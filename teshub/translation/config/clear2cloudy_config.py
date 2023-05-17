from dataclasses import dataclass
from typing import Callable

import torch

from teshub.recognition.utils import DEFAULT_SEG_LABEL2ID
from teshub.translation.config.base_config import BaseTranslationConfig
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus


@dataclass
class Clear2CloudyConfig(BaseTranslationConfig):
    @staticmethod
    def attention_map(seg: torch.Tensor) -> torch.Tensor:
        attention = torch.full_like(seg, 0.05, dtype=torch.float32)

        attention[
            seg == DEFAULT_SEG_LABEL2ID['blue_sky']
        ] = 1.0

        attention = torch.where(
            torch.logical_or(
                seg == DEFAULT_SEG_LABEL2ID['gray_sky'],
                seg == DEFAULT_SEG_LABEL2ID['white_sky']
            ), 0.75, attention)

        return attention

    @staticmethod
    def frame_selectors() -> tuple[
        Callable[[WebcamFrame], bool],
        Callable[[WebcamFrame], bool]
    ]:
        def is_plain_cloudy(frame: WebcamFrame) -> bool:
            assert frame.labels

            if frame.status not in [WebcamFrameStatus.MANUALLY_ANNOTATED,
                                    WebcamFrameStatus.AUTOMATICALLY_ANNOTATED]:
                return False

            if frame.labels['cloudy'] > 0.8 and frame.labels['foggy'] < 0.1:
                return True

            return False

        def is_clear(frame: WebcamFrame) -> bool:
            assert frame.labels

            if frame.status not in [WebcamFrameStatus.MANUALLY_ANNOTATED,
                                    WebcamFrameStatus.AUTOMATICALLY_ANNOTATED]:
                return False

            if all(rating < 0.05 for rating in frame.labels.values()):
                return True

            return False

        return is_clear, is_plain_cloudy
