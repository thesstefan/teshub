from dataclasses import dataclass
from typing import Callable

import torch

from teshub.recognition.utils import DEFAULT_SEG_LABEL2ID
from teshub.translation.config.base_config import BaseTranslationConfig
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus


@dataclass
class AddFogConfig(BaseTranslationConfig):
    @staticmethod
    def attention_map(seg: torch.Tensor) -> torch.Tensor:
        attention = torch.full_like(seg, 1.0, dtype=torch.float32)

        attention = torch.where(
            torch.logical_or(
                seg == DEFAULT_SEG_LABEL2ID['snow'],
                seg == DEFAULT_SEG_LABEL2ID['shadow_snow'],
            ), 0.6, attention)

        attention = torch.where(
            torch.logical_or(
                seg == DEFAULT_SEG_LABEL2ID['background'],
                seg == DEFAULT_SEG_LABEL2ID['shadow']
            ), 0.6, attention)

        return attention

    @ staticmethod
    def frame_selectors() -> tuple[
        Callable[[WebcamFrame], bool],
        Callable[[WebcamFrame], bool]
    ]:
        def is_not_foggy(frame: WebcamFrame) -> bool:
            assert frame.labels

            if frame.status not in [WebcamFrameStatus.MANUALLY_ANNOTATED,
                                    WebcamFrameStatus.AUTOMATICALLY_ANNOTATED]:
                return False

            if frame.labels['foggy'] < 0.1:
                return True

            return False

        def is_foggy(frame: WebcamFrame) -> bool:
            assert frame.labels

            if frame.status not in [WebcamFrameStatus.MANUALLY_ANNOTATED,
                                    WebcamFrameStatus.AUTOMATICALLY_ANNOTATED]:
                return False

            if frame.labels['foggy'] > 0.5:
                return True

            return False

        return is_not_foggy, is_foggy
