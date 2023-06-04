# mypy: disable-error-code="misc, no-any-unimported"
# lightning doesn't don't have typing stubs

# The pl_bolts module doesn't support PyTorch Lightning 2.0 out of the box.

from dataclasses import dataclass, field

import lightning.pytorch as pl  # type: ignore[import]
import torch
from pl_bolts.models.autoencoders.components import (  # type: ignore[import]
    resnet50_decoder, resnet50_encoder)
from torch import nn


@dataclass(eq=False, kw_only=True)
class WeatherMorphVAE(pl.LightningModule):
    input_height: int

    lr: float = 1e-4
    latent_dim: int = 256
    kl_weight: float = 0.1

    encoder: nn.Module = field(init=False)
    decoder: nn.Module = field(init=False)

    fc_mu: nn.Module = field(init=False)
    fc_var: nn.Module = field(init=False)

    first_conv: bool = field(init=False, default=True)
    maxpool1: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        super().__init__()

        self.encoder = resnet50_encoder(
            first_conv=self.first_conv,
            maxpool1=self.maxpool1)

        self.decoder = resnet50_decoder(
            self.latent_dim,
            self.input_height,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1)

        self.fc_mu = nn.LazyLinear(self.latent_dim)
        self.fc_var = nn.LazyLinear(self.latent_dim)

        self.save_hyperparameters(
            "input_height", "lr", "latent_dim", "kl_weight",
            "first_conv", "maxpool1"
        )

    # Re-check this
    def sample(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> tuple[
            torch.distributions.Distribution,
            torch.distributions.Distribution,
            torch.Tensor
    ]:
        std = torch.exp(log_var / 2)

        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return p, q, z

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(image)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        p, q, z = self.sample(mu, log_var)

        return self.decoder(z)

    def _run_step(
        self, image: torch.Tensor
    ) -> tuple[
            torch.distributions.Distribution,
            torch.distributions.Distribution,
            torch.Tensor
    ]:
        encoded = self.encoder(image)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def step(
        self,
        batch: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
        batch_idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x, y = batch
        x = x[0]
        y = y[0]
        z, x_hat, p, q = self._run_step(x)

        recon_loss = torch.nn.functional.mse_loss(x_hat, y, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_weight

        loss = recon_loss + kl

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }

        return loss, logs

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
        batch_idx: int
    ) -> torch.Tensor:
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()},
                      on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
