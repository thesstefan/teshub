import argparse
import os
from typing import cast
from tqdm import tqdm
import evaluate

import torch
from PIL.Image import Image
from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.segmentation.weather2seg import Weather2SegDataset
from teshub.segmentation.weather_segformer import WeatherSegformer

parser = argparse.ArgumentParser(
    prog="teshub_seg",
    description="Provides tooling for CVAT interaction",
)

parser.add_argument(
    "--csv_path",
    type=str,
    help=(
        "CSV file where webcam metadata is stored. "
        "If not specified, `dataset_dir/webcams.csv` is used"
    ),
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=".",
    help=(
        "Directory where webcam streams are stored. "
        "If specified, local CVAT storage will be used. "
        "Otherwise, will attempt to use shared CVAT storage with "
        "image paths from the current directory"
    ),
)


def csv_path_from_args(args: argparse.Namespace) -> str:
    return os.path.abspath(cast(str, args.csv_path)) if args.csv_path else None

    # label:color_rgb:parts:actions


def main() -> None:
    args = parser.parse_args()

    webcam_dataset = WebcamDataset(
        cast(str, os.path.abspath(args.dataset_dir)), csv_path_from_args(args)
    )

    def feature_extractor(image: Image, segmentation: Image) -> torch.Tensor:
        encoded_inputs = SegformerImageProcessor()(
            image,
            segmentation,
            return_tensors="pt",
        )

        values = [
            Weather2SegDataset.color2id[tuple(label_color.tolist())]
            for label_color in encoded_inputs["labels"].view(-1, 3)
        ]
        encoded_inputs["labels"] = torch.tensor(values).view(512, 512, 1)

        for categories, values in encoded_inputs.items():
            values.squeeze_()

        return encoded_inputs

    weather2seg = Weather2SegDataset(webcam_dataset, feature_extractor)

    train_size = int(len(weather2seg) * 0.9)
    val_size = len(weather2seg) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        weather2seg, [train_size, val_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    model = WeatherSegformer("nvidia/mit-b0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

    metric = evaluate.load("mean_iou")
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(200):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs

            loss.backward()
            optimizer.step()

            # evaluate
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                predicted = upsampled_logits.argmax(dim=1)

                # note that the metric expects predictions + labels as numpy arrays
                metric.add_batch(
                    predictions=predicted.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy(),
                )

            # let's print loss and metrics every 100 batches
            if idx % 100 == 0:
                metrics = metric.compute(
                    num_labels=len(Weather2SegDataset.color2id),
                    ignore_index=255,
                    reduce_labels=False,  # we've already reduced the labels before)
                )

                print("Loss:", loss.item())
                print("Mean_iou:", metrics["mean_iou"])
                print("Mean accuracy:", metrics["mean_accuracy"])


if __name__ == "__main__":
    main()
