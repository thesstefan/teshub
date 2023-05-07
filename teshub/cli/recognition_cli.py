from PIL import Image
import argparse
import os
from dataclasses import dataclass

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.recognition.weather2info import Weather2InfoDataset
from teshub.recognition.predictor import WeatherInFormerPredictor
from teshub.recognition.trainer import WeatherInFormerTrainer
from teshub.recognition.utils import DEFAULT_SEG_COLORS
from teshub.visualization.transforms import seg_mask_to_image

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog="teshub_recognition",
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
        "Directory where webcam streams and metadata are stored"
    ),
)
train_parser.add_argument(
    "--batch_size",
    type=int,
    default=2
)
train_parser.add_argument(
    "--pretrained_segformer_model",
    type=str,
    default="nvidia/mit-b1"
)
train_parser.add_argument(
    "--metrics_interval",
    type=int,
    default=5
)
train_parser.add_argument(
    "--train_val_split_ratio",
    type=float,
    default=0.9
)
train_parser.add_argument(
    "--lr",
    type=float,
    default=6 * 10e-05
)
train_parser.add_argument(
    "--resume_training_checkpoint_path",
    type=str
)
train_parser.add_argument(
    "--tb_logdir",
    type=str,
    default="tb_logs"
)
train_parser.add_argument(
    "--seg_loss_weight",
    type=float,
    default=0.5
)
train_parser.add_argument(
    "--reg_loss_weight",
    type=float,
    default=0.5
)
train_parser.add_argument(
    "--reg_loss_used",
    type=str,
    default='mse'
)
train_parser.add_argument(
    "--early_stop",
    action=argparse.BooleanOptionalAction,
    type=bool,
    default=True
)

predict_parser = subparsers.add_parser("predict")
predict_parser.add_argument(
    "--image_path",
    type=str,
    help="Path to image to be used for inference",
    required=True
)
predict_parser.add_argument(
    "--model_checkpoint_path",
    type=str,
    help="Path to model checkpoint to be used",
    required=True,
)


@dataclass(kw_only=True)
class Arguments:
    dataset_dir: str
    batch_size: int

    pretrained_segformer_model: str
    lr: float
    seg_loss_weight: float
    reg_loss_weight: float
    reg_loss_used: str

    train_val_split_ratio: float

    tb_logdir: str
    early_stop: bool

    metrics_interval: int

    csv_path: str | None = None
    resume_training_checkpoint_path: str | None = None

    image_path: str | None = None
    model_checkpoint_path: str | None = None


def csv_path_from_args(args: Arguments) -> str | None:
    return os.path.abspath(args.csv_path) if args.csv_path else None


def train(args: Arguments) -> None:
    webcam_dataset = WebcamDataset(
        os.path.abspath(args.dataset_dir), csv_path_from_args(args)
    )
    weather2seg = Weather2InfoDataset(webcam_dataset)

    trainer = WeatherInFormerTrainer(
        weather2seg,
        pretrained_segformer_model=args.pretrained_segformer_model,
        lr=args.lr,

        seg_loss_weight=args.seg_loss_weight,
        reg_loss_weight=args.reg_loss_weight,
        reg_loss_used=args.reg_loss_used,

        batch_size=args.batch_size,
        metrics_interval=args.metrics_interval,
        train_val_split_ratio=args.train_val_split_ratio,
        tb_log_dir=args.tb_logdir,
        resume_checkpoint=args.resume_training_checkpoint_path
    )

    trainer.fit()


def predict(args: Arguments) -> None:
    assert args.model_checkpoint_path
    assert args.image_path

    predictor = WeatherInFormerPredictor(
        model_checkpoint_path=args.model_checkpoint_path,
    )

    prediction = predictor.predict(args.image_path)
    predicted_img = seg_mask_to_image(prediction[0], DEFAULT_SEG_COLORS)

    # TODO: Create elaborate visualization tools in
    # visualization module
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axes[0].imshow(Image.open(args.image_path))
    axes[1].imshow(predicted_img)
    plt.show()


def main() -> None:
    train_parser.set_defaults(func=train)
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()

    args_dict = dict(vars(args))
    args_dict.pop('func')

    args.func(Arguments(**args_dict))


if __name__ == "__main__":
    main()
