# mypy: disable-error-code="misc"

from dataclasses import dataclass

import lightning.pytorch as pl  # type: ignore[import]
import torch
from lightning.pytorch.callbacks import Callback  # type: ignore[import]
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore[import]
from torch.utils.data import DataLoader, Subset, random_split

from teshub.translation.teshub import Teshub
from teshub.translation.weather2weather import Weather2WeatherDataset


@dataclass
class TeshubTrainer:
    weather2weather: Weather2WeatherDataset

    lr: float
    lambda_recon: int

    batch_size: int
    metrics_interval: int

    train_val_split_ratio: float

    tb_log_dir: str
    resume_checkpoint: str | None = None

    train_load_workers: int = 6
    val_load_workers: int = 6

    model_name: str = "teshub"
    max_epochs: int = 300

    ckpt_save_top_k: int = 1
    ckpt_monitor: str = "val_loss"

    early_stop: bool = True

    def _setup_train_val(self) -> tuple[
        Subset[tuple[torch.Tensor, torch.Tensor]],
        Subset[tuple[torch.Tensor, torch.Tensor]]
    ]:
        train_size = int(
            len(self.weather2weather) * self.train_val_split_ratio
        )
        val_size = len(self.weather2weather) - train_size

        train_subset, val_subset = (
            random_split(self.weather2weather, [train_size, val_size])
        )

        return train_subset, val_subset

    def fit(self) -> None:
        train_dataset, val_dataset = self._setup_train_val()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.train_load_workers,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.val_load_workers
        )

        model = Teshub(
            lr=self.lr,
            lambda_recon=self.lambda_recon,

            train_loader=train_dataloader,
            val_loader=val_dataloader,

            metrics_interval=self.metrics_interval
        )

        logger = TensorBoardLogger(
            self.tb_log_dir, name=self.model_name)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=self.ckpt_save_top_k, monitor=self.ckpt_monitor)

        callbacks: list[Callback] = [  # type: ignore[no-any-unimported]
            checkpoint_callback
        ]

        if self.early_stop:
            early_stop_callback = EarlyStopping(
                monitor=self.ckpt_monitor,
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            val_check_interval=len(train_dataloader),
            logger=logger,
            callbacks=callbacks,
        )

        trainer.fit(
            model, ckpt_path=self.resume_checkpoint
        )
