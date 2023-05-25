# mypy: disable-error-code="misc, no-any-unimported"
import argparse
import os
from dataclasses import dataclass

import lightning.pytorch as pl  # type: ignore[import]
import torch
from lightning.pytorch.callbacks import Callback  # type: ignore[import]
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore[import]
from torch.utils.data import DataLoader, Subset

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.translation.att_pix2pix import AttentionPix2Pix
from teshub.translation.config.clear2cloudy_config import Clear2CloudyConfig
from teshub.translation.tensorboard_image_sampler import \
    TensorboardImageSampler
from teshub.translation.weather2weather import Weather2WeatherDataset
from teshub.translation.weather_morph import WeatherMorph

parser = argparse.ArgumentParser(
    prog="teshub_translation",
    description=(
        "Does weather translation?"
    )
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
        "Directory where webcam streams and metadata are stored"
    ),
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2
)
parser.add_argument(
    "--resume_training_checkpoint_path",
    type=str
)
parser.add_argument(
    "--tb_logdir",
    type=str,
    default="tb_logs"
)


@dataclass(kw_only=True)
class Arguments:
    batch_size: int

    dataset_dir: str = '.'
    csv_path: str | None = None

    resume_training_checkpoint_path: str | None
    max_epochs = 300
    tb_logdir: str = 'tb_logs'
    ckpt_save_top_k: int = -1
    ckpt_save_every_n_epochs: int = 10

    train_load_workers: int = 6


def csv_path_from_args(args: Arguments) -> str | None:
    return os.path.abspath(args.csv_path) if args.csv_path else None


def setup_trainer(
    args: Arguments,
    model_name: str,
    visualize_dataloader: DataLoader[tuple[tuple[torch.Tensor, ...],
                                           tuple[torch.Tensor, ...]]]
) -> pl.Trainer:
    logger = TensorBoardLogger(
        args.tb_logdir, name=model_name)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.ckpt_save_top_k,
        every_n_epochs=args.ckpt_save_every_n_epochs)

    callbacks: list[Callback] = [  # type: ignore[no-any-unimported]
        checkpoint_callback,
        TensorboardImageSampler(visualize_dataloader,
                                use_att=True, use_seg=True)
    ]

    return pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=5,
    )


def main() -> None:
    args = Arguments(**dict(vars(parser.parse_args())))  # type: ignore

    webcam_dataset = WebcamDataset(
        os.path.abspath(args.dataset_dir), csv_path_from_args(args)
    )

    weather2weather = Weather2WeatherDataset.from_translation_config(
        webcam_dataset,
        translation_config=Clear2CloudyConfig,
        max_pairs_per_webcam=2,
        return_att=True,
        return_seg=True,
        return_labels=True,
    )

    model = WeatherMorph(
        weather_informer_ckpt_path="INFORMER.ckpt",
        lr=0.0002,
        lambda_reconstruct=200
    )

    visualize_count: int = 5
    dataset_indices = list(range(len(weather2weather)))
    visualize_dataloader = DataLoader(
        Subset(weather2weather, dataset_indices[:visualize_count]),
        batch_size=args.batch_size,
        num_workers=1,
    )
    train_dataloader = DataLoader(
        Subset(weather2weather, dataset_indices[visualize_count:]),
        batch_size=args.batch_size,
        num_workers=args.train_load_workers,
    )

    trainer = setup_trainer(
        args, model_name="att_clear2cloudy",
        visualize_dataloader=visualize_dataloader
    )

    trainer.fit(
        model, train_dataloader,
        ckpt_path=args.resume_training_checkpoint_path
    )
