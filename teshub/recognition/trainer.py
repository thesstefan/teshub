# mypy: disable-error-code="misc"

from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
from torch import nn
from typing import Type

from teshub.recognition.weather2info import Weather2InfoDataset
from teshub.recognition.weather_informer import WeatherInFormer

from lightning.pytorch.callbacks import (  # type: ignore[import]
    Callback, EarlyStopping, ModelCheckpoint
)
from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore[import]
import lightning.pytorch as pl  # type: ignore[import]


@dataclass
class WeatherInFormerTrainer:
    weather2info: Weather2InfoDataset

    pretrained_segformer_model: str
    lr: float

    seg_loss_weight: float
    reg_loss_weight: float
    reg_loss_used: str

    batch_size: int
    metrics_interval: int

    train_val_split_ratio: float
    tb_log_dir: str
    resume_checkpoint: str | None = None

    train_load_workers: int = 6
    val_load_workers: int = 6

    model_name: str = "weather_informer"
    max_epochs: int = 300

    ckpt_save_top_k: int = 1
    ckpt_monitor: str = "val_loss"

    early_stop: bool = True

    def _choose_reg_loss_criterion(
            self, name: str) -> Type[nn.modules.loss._Loss]:
        LOSS_DICT = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss
        }

        return LOSS_DICT[name]

    def fit(self) -> None:
        train_size = int(len(self.weather2info) * self.train_val_split_ratio)
        val_size = len(self.weather2info) - train_size

        train_dataset, val_dataset = random_split(
            self.weather2info, [train_size, val_size]
        )

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

        model = WeatherInFormer(
            pretrained_segformer_model=self.pretrained_segformer_model,
            lr=self.lr,
            seg_loss_weight=self.seg_loss_weight,
            reg_loss_weight=self.reg_loss_weight,
            reg_loss_criterion=self._choose_reg_loss_criterion(
                self.reg_loss_used),

            train_loader=train_dataloader,
            val_loader=val_dataloader,

            label_names=self.weather2info.label_names,
            seg_label_names=self.weather2info.seg_label_names,
            seg_label2id=self.weather2info.seg_label2id,

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
