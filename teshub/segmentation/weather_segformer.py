from dataclasses import dataclass, field
from typing import ClassVar

import lightning.pytorch as pl
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation


@dataclass(eq=False)
class WeatherSegformerFinetuner(pl.LightningModule):
    id2label: ClassVar[dict[int, str]] = {
        0: "background",
        1: "black_clouds",
        2: "white_clouds",
        3: "blue_sky",
        4: "gray_sky",
        5: "white_sky",
        6: "fog",
        7: "sun",
        8: "snow",
        9: "shadow",
        10: "wet_ground",
    }
    label2id: ClassVar[dict[str, int]] = {
        "background": 0,
        "black_clouds": 1,
        "white_clouds": 2,
        "blue_sky": 3,
        "gray_sky": 4,
        "white_sky": 5,
        "fog": 6,
        "sun": 7,
        "snow": 8,
        "shadow": 9,
        "wet_ground": 10,
    }

    train_loader: DataLoader
    val_loader: DataLoader

    metrics_interval: int = 100
    pretrained_model: str = "nvidia/mit-b0"

    model: nn.Module = field(init=False)
    train_metrics: torchmetrics.MetricCollection = field(init=False)
    val_metrics: torchmetrics.MetricCollection = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        self.model: nn.Module = (
            SegformerForSemanticSegmentation.from_pretrained(
                self.pretrained_model,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        )

        metrics = torchmetrics.MetricCollection(
            {
                "mean_iou": torchmetrics.classification.JaccardIndex(
                    task="multiclass",
                    num_classes=len(self.label2id),
                    average="macro",
                ),
                "mean_acc": torchmetrics.classification.Accuracy(
                    task="multiclass",
                    num_classes=len(self.label2id),
                    average="macro",
                ),
            }
        )

        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

    def forward(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output: tuple[torch.Tensor, ...] = self.model(
            pixel_values=images, labels=masks, return_dict=False
        )
        loss, logits = output

        return loss, logits

    def _get_predicted(
        self, logits: torch.Tensor, size: torch.Size
    ) -> torch.Tensor:
        upsampled_logits: torch.Tensor = nn.functional.interpolate(
            logits, size=size, mode="bilinear", align_corners=False
        )

        return upsampled_logits.argmax(dim=1)

    def _compute_loss_and_update_metrics(
        self,
        batch: dict[str, torch.Tensor],
        metrics: torchmetrics.MetricCollection,
    ) -> torch.Tensor:
        images, masks = batch["pixel_values"], batch["labels"]

        outputs = self.forward(images, masks)
        loss, logits = outputs

        predicted = self._get_predicted(logits, size=masks.shape[-2:])
        metrics.update(predicted, masks)

        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss: torch.Tensor = self._compute_loss_and_update_metrics(
            batch, self.train_metrics
        )

        self.log("train_loss", loss, prog_bar=True)

        if batch_idx % self.metrics_interval == 0:
            self.log_dict(self.train_metrics.compute(), prog_bar=True)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        loss = self._compute_loss_and_update_metrics(
            batch, metrics=self.val_metrics
        )

        self.log("val_loss", loss, prog_bar=True)

        if batch_idx % self.metrics_interval == 0:
            self.log_dict(self.val_metrics.compute(), prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=0.00006,
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
