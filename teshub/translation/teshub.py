# mypy: disable-error-code="misc, no-any-unimported"
# lightning and torchmetrics don't have typing stubs
from dataclasses import dataclass, field

import lightning.pytorch as pl  # type: ignore[import]
import torch
from pl_bolts.models.gans.pix2pix.pix2pix_module import Pix2Pix
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


@dataclass(eq=False)
class Teshub(pl.LightningModule):
    lr: float
    lambda_recon: int = 200

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    batch_size: int | None = field(init=False, default=None)
    metrics_interval: int = 100

    _pix2pix: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False

        self._pix2pix = Pix2Pix(
            in_channels=4,
            out_channels=4,
            learning_rate=self.lr,
            lambda_recon=self.lambda_recon,
        )

        self.save_hyperparameters("lr", "lambda_recon")

    def forward(self, real: torch.Tensor) -> torch.Tensor:
        return self._pix2pix(real)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._pix2pix.training_step(
            batch, batch_idx, optimizer_idx=None)

    def configure_optimizers(self):
        return self._pix2pix.configure_optimizers()

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]] | None:
        return self.train_loader

    def val_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]] | None:
        return self.val_loader
