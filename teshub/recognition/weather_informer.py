# mypy: disable-error-code="misc, no-any-unimported"
# lightning, torchmetrics and transformers don't have typing stubs

from dataclasses import dataclass, field
import itertools

import lightning.pytorch as pl  # type: ignore[import]
import torch
from torchmetrics import (  # type: ignore[import]
    MetricCollection, JaccardIndex, Accuracy,
    MeanSquaredError, MeanAbsoluteError
)
from torch import nn
from torch.utils.data import DataLoader
from transformers import (  # type: ignore[import]
    SegformerForSemanticSegmentation
)
from teshub.recognition.utils import upsample_logits
from typing import Type


@dataclass(eq=False)
class WeatherInFormer(pl.LightningModule):
    pretrained_segformer_model: str
    lr: float

    seg_loss_weight: float
    reg_loss_weight: float
    reg_loss_criterion: Type[nn.modules.loss._Loss]

    label_names: list[str]
    seg_label_names: list[str]
    seg_label2id: dict[str, int]

    train_loader: DataLoader[dict[str, torch.Tensor]] | None = None
    val_loader: DataLoader[dict[str, torch.Tensor]] | None = None
    batch_size: int | None = field(init=False, default=None)

    metrics_interval: int = 100
    metrics: dict[str, dict[str, MetricCollection]] = field(
        init=False, default_factory=lambda: {'train': {}, 'val': {}})

    _segformer: nn.Module = field(init=False)
    _regression: nn.Module = field(init=False)
    _reg_loss_fn: nn.modules.loss._Loss = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        if self.train_loader:
            self.batch_size = self.train_loader.batch_size

        self._segformer = (
            SegformerForSemanticSegmentation.from_pretrained(
                self.pretrained_segformer_model,
                num_labels=len(self.seg_label_names),
                id2label=self.seg_label_names,
                label2id=self.seg_label2id,
            )
        )

        self._regression = nn.Sequential(
            nn.LazyLinear(out_features=len(self.label_names)),
            nn.Sigmoid()
        )
        self._reg_loss_fn = self.reg_loss_criterion()

        self._init_metrics()
        self.save_hyperparameters(
            "batch_size", "pretrained_segformer_model", "lr",
            "seg_loss_weight", "reg_loss_weight", "reg_loss_criterion"
        )

    def _construct_metrics(self, phase: str, task: str) -> MetricCollection:
        match task:
            case "seg":
                return MetricCollection(
                    {
                        "mean_iou": JaccardIndex(
                            task="multiclass",
                            num_classes=len(self.seg_label_names),
                            average="macro",
                        ),
                        "mean_acc": Accuracy(
                            task="multiclass",
                            num_classes=len(self.seg_label_names),
                            average="macro",
                        ),
                    },
                    prefix="seg",
                    postfix=phase
                )
            case "reg":
                return MetricCollection(
                    {
                        "mse": MeanSquaredError(),
                        "mae": MeanAbsoluteError()
                    },
                    prefix="reg",
                    postfix=phase
                )
            case _:
                raise RuntimeError(
                    "Invalid task for constructing "
                    f"metrics {task} (use seg or reg)")

    def _init_metrics(self) -> None:
        for phase, task in itertools.product(
                ['train', 'val'], ['seg', 'reg']):
            self.metrics[phase][task] = self._construct_metrics(phase, task)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        segformer_output: tuple[torch.Tensor, ...] = self._segformer(
            pixel_values=images, labels=masks, return_dict=False
        )

        # TODO: Find a cleaner way of doing this. Segformer output
        # contains loss only when training
        seg_mask: torch.Tensor = segformer_output[0]
        if len(segformer_output) == 2:
            seg_mask = segformer_output[1]

        batch_size = seg_mask.size(0)
        reg_output: tuple[torch.Tensor, ...] = (self._regression(
            seg_mask.reshape(batch_size, -1)))

        if labels:
            reg_loss = self._reg_loss_fn(reg_output[0], labels)
            reg_output = reg_loss + reg_output

        return segformer_output, reg_output

    def _compute_losses_and_update_metrics(
        self,
        batch: dict[str, torch.Tensor],
        phase: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, masks, labels = (
            batch["pixel_values"], batch["seg_labels"], batch["labels"]
        )

        seg_outputs, reg_outputs = self.forward(images, masks, labels)
        seg_loss, seg_logits = seg_outputs
        reg_loss, predicted_labels = reg_outputs

        predicted_masks = upsample_logits(seg_logits, size=masks.shape[-2:])

        self.metrics[phase]['seg'].update(predicted_masks, masks)
        self.metrics[phase]['reg'].update(predicted_labels, labels)

        return (
            seg_loss, reg_loss,
            seg_loss * self.seg_loss_weight + reg_loss * self.reg_loss_weight
        )

    def _log_losses(
        self,
        total_loss: torch.Tensor,
        seg_loss: torch.Tensor,
        reg_loss: torch.Tensor,
        phase: str
    ) -> None:
        self.log(f"{phase}_loss", total_loss, prog_bar=True)
        self.log(f"{phase}_seg_loss", seg_loss, prog_bar=True)
        self.log(f"{phase}_reg_loss", reg_loss, prog_bar=True)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        losses = self._compute_losses_and_update_metrics(batch, phase='train')
        self._log_losses(*losses, phase='train')

        if batch_idx and batch_idx % self.metrics_interval == 0:
            self.log_dict(self.metrics['train']['seg'].compute(),
                          on_epoch=True)
            self.log_dict(self.metrics['train']['reg'].compute(),
                          on_epoch=True)

        return losses[0]

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        losses = self._compute_losses_and_update_metrics(batch, phase='val')
        self._log_losses(*losses, phase='val')

        if batch_idx and batch_idx % self.metrics_interval == 0:
            self.log_dict(self.metrics['val']['seg'].compute(),
                          on_epoch=True)
            self.log_dict(self.metrics['val']['reg'].compute(),
                          on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.lr
        )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        return self.train_loader

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        return self.val_loader
