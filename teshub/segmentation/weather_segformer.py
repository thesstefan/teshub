# mypy: disable-error-code="misc, no-any-unimported"
# lightning, torchmetrics and transformers don't have typing stubs

from dataclasses import dataclass, field

import lightning.pytorch as pl  # type: ignore[import]
import torch
from torchmetrics import (  # type: ignore[import]
    MetricCollection, JaccardIndex, Accuracy
)
from torch import nn
from torch.utils.data import DataLoader
from transformers import (  # type: ignore[import]
    SegformerForSemanticSegmentation
)
from teshub.segmentation.utils import (
    DEFAULT_LABEL2ID, DEFAULT_ID2LABEL, upsample_logits
)


@dataclass(eq=False)
class WeatherSegformer(pl.LightningModule):
    pretrained_model_name: str
    lr: float

    label2id: dict[str, int] = field(default_factory=lambda: DEFAULT_LABEL2ID)
    id2label: dict[int, str] = field(default_factory=lambda: DEFAULT_ID2LABEL)

    train_loader: DataLoader[dict[str, torch.Tensor]] | None = None
    val_loader: DataLoader[dict[str, torch.Tensor]] | None = None

    metrics_interval: int = 100

    batch_size: int | None = field(init=False, default=None)
    train_metrics: MetricCollection = field(init=False)
    val_metrics: MetricCollection = field(init=False)

    _segformer: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        if self.train_loader:
            self.batch_size = self.train_loader.batch_size

        self._segformer: nn.Module = (
            SegformerForSemanticSegmentation.from_pretrained(
                self.pretrained_model_name,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        )

        metrics = MetricCollection(
            {
                "mean_iou": JaccardIndex(
                    task="multiclass",
                    num_classes=len(self.label2id),
                    average="macro",
                ),
                "mean_acc": Accuracy(
                    task="multiclass",
                    num_classes=len(self.label2id),
                    average="macro",
                ),
            }
        )

        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

        self.save_hyperparameters("batch_size", "pretrained_model_name", "lr")


    def forward(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        output: tuple[torch.Tensor, ...] = self._segformer(
            pixel_values=images, labels=masks, return_dict=False
        )

        return output

    def _compute_loss_and_update_metrics(
        self,
        batch: dict[str, torch.Tensor],
        metrics: MetricCollection,
    ) -> torch.Tensor:
        images, masks = batch["pixel_values"], batch["labels"]

        outputs = self.forward(images, masks)
        loss, logits = outputs

        predicted = upsample_logits(logits, size=masks.shape[-2:])
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
            self.log_dict(self.train_metrics.compute(), on_epoch=True)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        loss = self._compute_loss_and_update_metrics(
            batch, metrics=self.val_metrics
        )

        self.log("val_loss", loss, prog_bar=True)

        if batch_idx % self.metrics_interval == 0:
            self.log_dict(self.val_metrics.compute(), on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.lr
        )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        return self.train_loader

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        return self.val_loader
