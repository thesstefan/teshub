from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

from teshub.segmentation.weather2seg import Weather2SegDataset
from teshub.segmentation.weather_segformer import WeatherSegformer

from lightning.pytorch.callbacks import (  # type: ignore[import]
    EarlyStopping, ModelCheckpoint
)
from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore[import]
import lightning.pytorch as pl  # type: ignore[import]


@dataclass
class SegmentationTrainer:
    weather2seg: Weather2SegDataset
    pretrained_model_name: str
    split_ratio: float
    batch_size: int
    metrics_interval: int
    tb_log_dir: str
    lr: float
    resume_checkpoint: str | None = None

    def fit(self) -> None:
        train_size = int(len(self.weather2seg) * self.split_ratio)
        val_size = len(self.weather2seg) - train_size

        train_dataset, val_dataset = random_split(
            self.weather2seg, [train_size, val_size]
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=6,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, num_workers=6)

        model = WeatherSegformer(
            pretrained_model_name=self.pretrained_model_name,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            metrics_interval=self.metrics_interval,
            lr=self.lr
        )

        logger = TensorBoardLogger(  # type: ignore[misc]
            "tb_logs", name="weather_segformer")
        checkpoint_callback = ModelCheckpoint(  # type: ignore[misc]
            save_top_k=1, monitor="val_loss")
        early_stop_callback = EarlyStopping(  # type: ignore[misc]
            monitor="val_loss",
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="min",
        )

        trainer = pl.Trainer(  # type: ignore[misc]
            max_epochs=300,
            val_check_interval=len(train_dataloader),
            logger=logger,  # type: ignore[misc]
            callbacks=[
                early_stop_callback, checkpoint_callback  # type: ignore[misc]
            ],
        )

        trainer.fit(  # type: ignore[misc]
            model, ckpt_path=self.resume_checkpoint
        )
