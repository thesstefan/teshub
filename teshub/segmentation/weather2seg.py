from dataclasses import dataclass, field
from typing import Callable, ClassVar

import torch
from PIL.Image import Image
from torch.utils.data import Dataset

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus
from teshub.webcam.webcam_stream import WebcamStatus


@dataclass
class Weather2SegDataset(Dataset):
    webcam_dataset: WebcamDataset
    feature_extractor: Callable[[Image, Image], torch.Tensor]

    frames: list[WebcamFrame] = field(init=False, default_factory=list)

    color2id: ClassVar[dict[tuple[int], str]] = {
        (0, 0, 0): 0,
        (22, 21, 22): 1,
        (204, 204, 204): 2,
        (46, 6, 243): 3,
        (154, 147, 185): 4,
        (198, 233, 255): 5,
        (255, 53, 94): 6,
        (250, 250, 55): 7,
        (255, 255, 255): 8,
        (115, 51, 128): 9,
        (36, 179, 83): 10,
    }
    id2color: ClassVar[dict[tuple[int], str]] = {
        color: id for (id, color) in color2id.items()
    }

    def __post_init__(self) -> None:
        annotated_webcams = self.webcam_dataset.get_webcams_with_status(
            WebcamStatus.PARTIALLY_ANNOTATED
        )

        for webcam in annotated_webcams:
            self.frames.extend(
                filter(
                    lambda frame: frame.status
                    == WebcamFrameStatus.SYNTHETIC_ANNOTATION,
                    webcam.frames,
                )
            )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        encoded_inputs = self.feature_extractor(
            self.frames[idx].image,
            self.frames[idx].segmentation,
        )

        return encoded_inputs
