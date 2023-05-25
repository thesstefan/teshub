# mypy: disable-error-code="misc, no-any-unimported"

from dataclasses import dataclass

import torch

from teshub.translation.pix2pix import Pix2Pix


@dataclass(eq=False, kw_only=True)
class AttentionPix2Pix(Pix2Pix):
    def forward(
        self,
        real_images: torch.Tensor,
        attention_map: torch.Tensor | None = None
    ) -> torch.Tensor:
        fake_mask: torch.Tensor = self.generator(real_images)

        if attention_map is None:
            return fake_mask

        fake_image: torch.Tensor = (
            attention_map * fake_mask + (1 - attention_map) * real_images
        )

        return fake_image

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
        batch_idx: int,
    ) -> None:
        real_data, conditioned_data = batch
        real_image_data, conditioned_image_data = (
            real_data[0], conditioned_data[0]
        )
        real_images, real_attention = torch.split(
            real_image_data, [real_image_data.shape[1] - 1, 1], dim=1)

        # TODO: Find use for conditioned attention or just drop it
        conditioned_images, _ = torch.split(
            conditioned_image_data,
            [conditioned_image_data.shape[1] - 1, 1], dim=1)

        fake_images = self.forward(real_images, real_attention)

        patch_gan_loss = self._optimize_patch_gan(
            real_images, conditioned_images, fake_images)

        generator_loss = self._optimize_generator(
            real_images, conditioned_images, fake_images)

        self.log_dict({
            "generator_loss": generator_loss,
            "patch_gan_loss": patch_gan_loss
        })
