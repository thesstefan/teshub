import abc
from typing import Callable

import torch
from PIL import Image

from teshub.webcam.webcam_frame import WebcamFrame


class TranslationConfig(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def frame_transforms() -> tuple[
        Callable[[Image.Image], torch.Tensor],
        Callable[[Image.Image], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
    ]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def frame_selectors() -> tuple[
        Callable[[WebcamFrame], bool],
        Callable[[WebcamFrame], bool]
    ]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def attention_map(seg: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
