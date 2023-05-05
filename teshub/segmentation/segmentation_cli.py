import argparse
import os
from typing import cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from PIL.Image import Image
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.segmentation.weather2seg import Weather2SegDataset
from teshub.segmentation.weather_segformer_v2 import WeatherSegformerFinetuner

parser = argparse.ArgumentParser(
    prog="teshub_segmentation",
    description=(
        "Provides tooling for running weather"
        "segmentation/classification model",
    ),
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


def main() -> None:
    args = parser.parse_args()

    webcam_dataset = WebcamDataset(
        cast(str, os.path.abspath(args.dataset_dir)), csv_path_from_args(args)
    )

    # TODO: Move this somewhere else
    def feature_extractor(image: Image, segmentation: Image) -> torch.Tensor:
        encoded_inputs = SegformerImageProcessor()(
            image,
            segmentation,
            return_tensors="pt",
        )

        labels_1d = [
            Weather2SegDataset.color2id[tuple(label_color.tolist())]
            for label_color in encoded_inputs["labels"].view(-1, 3)
        ]
        encoded_inputs["labels"] = torch.tensor(labels_1d).view(512, 512, 1)

        for categories, values in encoded_inputs.items():
            values.squeeze_()

        return encoded_inputs

    weather2seg = Weather2SegDataset(webcam_dataset, feature_extractor)

    train_size = int(len(weather2seg) * 0.9)
    val_size = len(weather2seg) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        weather2seg, [train_size, val_size]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=6
    )
    val_dataloader = DataLoader(val_dataset, batch_size=2, num_workers=6)

    finetuner = WeatherSegformerFinetuner(
        pretrained_model="nvidia/mit-b0",
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        metrics_interval=5,
    )

    # TODO: Add CLI params for settings
    logger = TensorBoardLogger("tb_logs", name="weather_segformer")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=200,
        val_check_interval=len(train_dataloader),
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(finetuner)


if __name__ == "__main__":
    main()
