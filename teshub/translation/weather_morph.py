# mypy: disable-error-code="misc, no-any-unimported"
from dataclasses import dataclass, field

import torch
from torch import nn
from torchvision.transforms import InterpolationMode  # type: ignore[import]
from torchvision.transforms.functional import (  # type: ignore[import]
    normalize, resize)

from teshub.recognition.utils import (DEFAULT_LABELS, DEFAULT_SEG_LABEL2ID,
                                      DEFAULT_SEG_LABELS)
from teshub.recognition.weather_informer import WeatherInFormer
from teshub.translation.att_pix2pix import AttentionPix2Pix


@dataclass(eq=False, kw_only=True)
class WeatherMorph(AttentionPix2Pix):
    weather_informer_ckpt_path: str
    weather_informer: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.weather_informer = WeatherInFormer.load_from_checkpoint(
            self.weather_informer_ckpt_path,
            map_location=(
                torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu')
            ),
            label_names=DEFAULT_LABELS,
            seg_label_names=DEFAULT_SEG_LABELS,
            seg_label2id=DEFAULT_SEG_LABEL2ID,
        )

        for param in self.weather_informer.parameters():
            param.requires_grad = False

    # TODO: Move this in some utils package
    def _revert_normalization(self, img: torch.Tensor) -> torch.Tensor:
        reverted: torch.Tensor = normalize(
            img, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        reverted = normalize(reverted, (-0.5, -0.5, -0.5), (1., 1., 1.))

        return reverted

    def _transform_weather_informer_input(
        self, images: torch.Tensor, seg_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed_images = resize(
            images, (512, 512), InterpolationMode.BILINEAR, antialias=True)
        transformed_images = self._revert_normalization(transformed_images)
        transformed_images = transformed_images * 255

        transformed_masks = resize(
            seg_masks, (512, 512), InterpolationMode.NEAREST, antialias=False
        ).squeeze(dim=1).long()

        return transformed_images, transformed_masks

    def _get_weather_informer_losses(
        self,
        conditioned_seg_mask: torch.Tensor,
        conditioned_labels: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed_fake_images, transformed_conditioned_seg_masks = (
            self._transform_weather_informer_input(
                fake_images, conditioned_seg_mask)
        )

        weather_informer_outputs: tuple[
            tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
        ] = self.weather_informer(
            transformed_fake_images,
            masks=transformed_conditioned_seg_masks,
            labels=conditioned_labels
        )
        seg_output, reg_output = weather_informer_outputs
        seg_loss, seg_logits = seg_output
        reg_loss, fake_labels = reg_output

        return seg_loss, reg_loss

    def _optimize_discriminator(
        self,
        real_images: torch.Tensor,
        conditioned_images: torch.Tensor,
        conditioned_seg_masks: torch.Tensor,
        conditioned_labels: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        discriminator_opt, _ = self.optimizers()

        self._set_requires_grad(self.patch_gan, True)
        discriminator_opt.zero_grad()

        patch_gan_loss = self._patch_gan_step(
            real_images, conditioned_images, fake_images
        )
        weather_seg_loss, weather_reg_loss = (
            self._get_weather_informer_losses(
                conditioned_seg_masks, conditioned_labels, fake_images)
        )

        discriminator_loss = 0.5 * patch_gan_loss + \
            weather_seg_loss * 0.4 + weather_reg_loss * 0.1

        self.manual_backward(discriminator_loss)
        discriminator_opt.step()

        return (
            discriminator_loss,
            patch_gan_loss, weather_seg_loss, weather_seg_loss
        )

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
        batch_idx: int,
    ) -> None:
        real_data, conditioned_data = batch
        real_image_data, real_labels = real_data

        conditioned_image_data, conditioned_labels = conditioned_data

        real_images, real_att, _ = torch.split(
            real_image_data, [real_image_data.shape[1] - 2, 1, 1], dim=1)

        conditioned_images, _, conditioned_seg_masks = torch.split(
            conditioned_image_data,
            [conditioned_image_data.shape[1] - 2, 1, 1], dim=1)

        fake_images = self.forward(real_images, real_att)

        (discriminator_loss, patch_gan_loss,
            weather_seg_loss, weather_reg_loss) = (
            self._optimize_discriminator(
                real_images, conditioned_images,
                conditioned_seg_masks, conditioned_labels, fake_images.detach()
            )
        )

        generator_loss = self._optimize_generator(
            real_images, conditioned_images, fake_images)

        self.log_dict({
            "discriminator_loss": discriminator_loss,
            "patch_gan_loss": patch_gan_loss,
            "weather_reg_loss": weather_reg_loss,
            "weather_seg_loss": weather_seg_loss,
            "generator_loss": generator_loss,
        })
