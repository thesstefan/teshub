from dataclasses import dataclass
from typing import Callable

import torch

from teshub.recognition.utils import DEFAULT_SEG_LABEL2ID
from teshub.translation.config.base_config import BaseTranslationConfig
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus


@dataclass
class RemoveSnowConfig(BaseTranslationConfig):
    @staticmethod
    def attention_map(seg: torch.Tensor) -> torch.Tensor:
        attention = torch.full_like(seg, 0.075, dtype=torch.float32)

        attention = torch.where(
            torch.logical_or(
                seg == DEFAULT_SEG_LABEL2ID['snow'],
                seg == DEFAULT_SEG_LABEL2ID['shadow_snow'],
            ), 1.0, attention)

        return attention

    @ staticmethod
    def frame_selectors() -> tuple[
        Callable[[WebcamFrame], bool],
        Callable[[WebcamFrame], bool]
    ]:
        def is_snowy(frame: WebcamFrame) -> bool:
            assert frame.labels

            if frame.status not in [WebcamFrameStatus.MANUALLY_ANNOTATED,
                                    WebcamFrameStatus.AUTOMATICALLY_ANNOTATED]:
                return False

            if frame.labels['snowy'] > 0.8 and frame.labels['foggy'] < 0.1:
                return True

            return False

        def is_not_snowy(frame: WebcamFrame) -> bool:
            assert frame.labels

            if frame.status not in [WebcamFrameStatus.MANUALLY_ANNOTATED,
                                    WebcamFrameStatus.AUTOMATICALLY_ANNOTATED]:
                return False

            if frame.labels['snowy'] < 0.1 and frame.labels['foggy'] < 0.1:
                return True

            return False

        return is_snowy, is_not_snowy
