import argparse
import os
from dataclasses import dataclass

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.segmentation.weather2seg import Weather2SegDataset
from teshub.segmentation.predictor import SegmentationPredictor
from teshub.segmentation.trainer import SegmentationTrainer

parser = argparse.ArgumentParser(
    prog="teshub_segmentation",
    description=(
        "Provides tooling for running weather"
        "segmentation/classification model"
    ),
)
subparsers = parser.add_subparsers(
    required=True,
    title="subcommands",
    description="Valid subcommands",
    help="Additional help",
)

train_parser = subparsers.add_parser("train")
predict_parser = subparsers.add_parser("predict")

train_parser.add_argument(
    "--csv_path",
    type=str,
    help=(
        "CSV file where webcam metadata is stored. "
        "If not specified, `dataset_dir/webcams.csv` is used"
    ),
)
train_parser.add_argument(
    "--dataset_dir",
    type=str,
    default=".",
    help=(
        "Directory where webcam streams and metadata are stored. "
    ),
)

predict_parser.add_argument(
    "--image_path", type=str, help="Input image for inference", required=True
)
predict_parser.add_argument(
    "--model_checkpoint_path",
    type=str,
    help="Path to model checkpoint to be used",
    required=True,
)


@dataclass(kw_only=True)
class Arguments:
    dataset_dir: str | None = None
    csv_path: str | None = None
    image_path: str | None = None
    model_checkpoint_path: str | None = None


def csv_path_from_args(args: Arguments) -> str | None:
    return os.path.abspath(args.csv_path) if args.csv_path else None


def train(args: Arguments) -> None:
    assert (args.dataset_dir)

    webcam_dataset = WebcamDataset(
        os.path.abspath(args.dataset_dir), csv_path_from_args(args)
    )
    weather2seg = Weather2SegDataset(webcam_dataset)

    trainer = SegmentationTrainer(
        weather2seg,
        pretrained_model_name='nvidia/mit-b1',
        split_ratio=0.9,
        batch_size=2,
        metrics_interval=5,
        tb_log_dir="tb_logs",
        resume_checkpoint=None
    )

    trainer.fit()


def predict(args: Arguments) -> None:
    assert (args.model_checkpoint_path)
    assert (args.image_path)

    predictor = SegmentationPredictor(
        model_checkpoint_path=args.model_checkpoint_path,
        pretrained_model_name='nvidia/mit-b1'
    )

    prediction = predictor.predict(args.image_path)

    # TODO: Implement visualization module or something


def main() -> None:
    train_parser.set_defaults(func=train)
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
