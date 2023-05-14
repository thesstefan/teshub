from dataclasses import dataclass, field
from itertools import product
from typing import Callable, cast

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor  # type: ignore[import]

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.extra_typing import Color

# TODO: Move this in a common directory
from teshub.recognition.utils import DEFAULT_SEG_COLOR2ID
from teshub.visualization.transforms import rgb_pixels_to_1d
from teshub.webcam.webcam_frame import WebcamFrame
from teshub.webcam.webcam_stream import WebcamStatus


@dataclass
class Weather2WeatherDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    webcam_dataset: WebcamDataset

    select_source: Callable[[WebcamFrame], bool]
    select_target: Callable[[WebcamFrame], bool]

    frame_mappings: list[tuple[WebcamFrame, WebcamFrame]] = field(
        init=False, default_factory=list
    )
    seg_color2id: dict[Color, int] = field(
        default_factory=lambda: DEFAULT_SEG_COLOR2ID
    )

    def __post_init__(self) -> None:
        annotated_webcams = self.webcam_dataset.get_webcams_with_status(
            WebcamStatus.ANNOTATED
        )

        for webcam in annotated_webcams:
            source_frames = filter(self.select_source, webcam.frames)
            target_frames = filter(self.select_target, webcam.frames)

            self.frame_mappings.extend(product(source_frames, target_frames))

    def __len__(self) -> int:
        return len(self.frame_mappings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_frame, target_frame = self.frame_mappings[idx]

        # TODO: Transposing here is awkward. Rewrite transform code
        # and use ChannelDimension.FIRST in ImageProcessor.

        source_pixels: torch.Tensor = pil_to_tensor(source_frame.image)
        target_pixels: torch.Tensor = pil_to_tensor(target_frame.image)

        source_seg_pixels: torch.Tensor = pil_to_tensor(
            source_frame.segmentation
        )
        target_seg_pixels: torch.Tensor = pil_to_tensor(
            target_frame.segmentation
        )

        source_seg = rgb_pixels_to_1d(
            source_seg_pixels, rgb_pixel_to_value=DEFAULT_SEG_COLOR2ID
        )
        target_seg = rgb_pixels_to_1d(
            target_seg_pixels, rgb_pixel_to_value=DEFAULT_SEG_COLOR2ID
        )

        source = torch.cat([source_pixels, source_seg], dim=0)
        target = torch.cat([target_pixels, target_seg], dim=0)

        return dict(source=source, target=target)
