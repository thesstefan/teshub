# mypy: disable-error-code="misc, no-any-unimported"
# lightning doesn't don't have typing stubs

# The pl_bolts module doesn't support PyTorch Lightning 2.0 out of the box.

from dataclasses import dataclass, field

import lightning.pytorch as pl  # type: ignore[import]
import torch
from pl_bolts.models.gans.pix2pix.components import (  # type: ignore[import]
    Generator, PatchGAN)
from torch import nn


@dataclass(eq=False, kw_only=True)
class Pix2Pix(pl.LightningModule):
    lr: float = 0.0002
    lambda_reconstruct: int = 200

    in_channels: int = 3
    out_channels: int = 3

    generator: nn.Module = field(init=False)
    patch_gan: nn.Module = field(init=False)

    adversarial_criterion: nn.modules.loss._Loss = field(init=False)
    reconstruct_criterion: nn.modules.loss._Loss = field(init=False)

    def __post_init__(self, save_hparams: bool = True) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.generator = Generator(self.in_channels, self.out_channels)
        self.patch_gan = PatchGAN(self.in_channels + self.out_channels)

        self.generator = self.generator.apply(self._weights_init)
        self.patch_gan = self.patch_gan.apply(self._weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.reconstruct_criterion = nn.L1Loss()

        if save_hparams:
            self.save_hyperparameters(
                "lr", "lambda_reconstruct"
            )

    # TODO: Move this out of the class
    def _weights_init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
        if isinstance(module, nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
            torch.nn.init.constant_(module.bias, 0)

    # TODO: Move this out of the class
    def _set_requires_grad(
        self, module: nn.Module, requires_grad: bool
    ) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _generator_step(
        self,
        real_images: torch.Tensor,
        conditioned_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        patch_gan_logits = self.patch_gan(fake_images, conditioned_images)

        adversarial_loss: torch.Tensor = self.adversarial_criterion(
            patch_gan_logits, torch.ones_like(patch_gan_logits)
        )

        reconstruct_loss: torch.Tensor = self.reconstruct_criterion(
            fake_images, conditioned_images)

        return adversarial_loss + self.lambda_reconstruct * reconstruct_loss

    def _patch_gan_step(
        self,
        real_images: torch.Tensor,
        conditioned_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        fake_logits = self.patch_gan(fake_images, conditioned_images)
        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss: torch.Tensor = self.adversarial_criterion(
            fake_logits, torch.zeros_like(fake_logits))
        real_loss: torch.Tensor = self.adversarial_criterion(
            real_logits, torch.ones_like(real_logits))

        return (real_loss + fake_loss) / 2

    def _optimize_patch_gan(
        self,
        real_images: torch.Tensor,
        conditioned_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        patch_gan_opt, _ = self.optimizers()

        self._set_requires_grad(self.patch_gan, True)
        patch_gan_opt.zero_grad()

        patch_gan_loss = self._patch_gan_step(
            real_images, conditioned_images, fake_images)

        self.manual_backward(patch_gan_loss)
        patch_gan_opt.step()

        return patch_gan_loss

    def _optimize_generator(
        self,
        real_images: torch.Tensor,
        conditioned_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        _, generator_opt = self.optimizers()

        self._set_requires_grad(self.patch_gan, False)
        generator_opt.zero_grad()

        generator_loss = self._generator_step(
            real_images, conditioned_images, fake_images)

        self.manual_backward(generator_loss)
        generator_opt.step()

        return generator_loss

    def forward(self, real_image: torch.Tensor) -> torch.Tensor:
        fake_image: torch.Tensor = self.generator(real_image)

        return fake_image

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
        batch_idx: int,
    ) -> None:
        real_data, conditioned_data = batch
        real_images, conditioned_images = real_data[0], conditioned_data[0]

        fake_images = self.forward(real_images)

        patch_gan_loss = self._optimize_patch_gan(
            real_images, conditioned_images, fake_images.detach())

        generator_loss = self._optimize_generator(
            real_images, conditioned_images, fake_images)

        self.log_dict({
            "generator_loss": generator_loss,
            "patch_gan_loss": patch_gan_loss
        })

    def configure_optimizers(
        self
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        return (
            torch.optim.AdamW(self.patch_gan.parameters(), lr=self.lr),
            torch.optim.AdamW(self.generator.parameters(), lr=self.lr),
        )
