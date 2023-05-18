# mypy: disable-error-code="misc, no-any-unimported"

import abc
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms  # type: ignore

from teshub.recognition.utils import DEFAULT_SEG_COLOR2ID
from teshub.translation.config.abc_config import TranslationConfig
from teshub.visualization.transforms import rgb_pixels_to_1d
from teshub.webcam.webcam_frame import WebcamFrame


@dataclass
class BaseTranslationConfig(TranslationConfig, abc.ABC):
    @classmethod
    def frame_transforms(
        cls,
        frame_size: tuple[int, int] = (512, 512),
    ) -> tuple[
        Callable[[Image.Image], torch.Tensor],
        Callable[[Image.Image], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor]
    ]:
        img_transform = transforms.Compose([
            transforms.Resize(
                frame_size,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            )
        ])

        seg_transform = transforms.Compose([
            transforms.Resize(
                frame_size,
                interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
            transforms.Lambda(
                lambda seg: rgb_pixels_to_1d(
                    seg, rgb_pixel_to_value=DEFAULT_SEG_COLOR2ID
                )),
        ])

        att_transform = transforms.Compose([
            transforms.Lambda(
                lambda seg: cls.attention_map(seg)
            ),
            transforms.GaussianBlur(kernel_size=101)
        ])

        return img_transform, seg_transform, att_transform

    @staticmethod
    def attention_map(seg: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(seg)

    @staticmethod
    @abc.abstractmethod
    def frame_selectors() -> tuple[
        Callable[[WebcamFrame], bool],
        Callable[[WebcamFrame], bool]
    ]:
        raise NotImplementedError
