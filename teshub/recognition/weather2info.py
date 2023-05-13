from dataclasses import dataclass, field

import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor  # type: ignore[import]

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.extra_typing import Color
from teshub.recognition.utils import (DEFAULT_FEATURE_EXTRACTOR_IMG_SIZE,
                                      DEFAULT_LABELS, DEFAULT_SEG_COLOR2ID,
                                      DEFAULT_SEG_COLORS, DEFAULT_SEG_LABEL2ID,
                                      DEFAULT_SEG_LABELS)
from teshub.visualization.transforms import rgb_pixels_to_1d
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus
from teshub.webcam.webcam_stream import WebcamStatus


@dataclass
class Weather2InfoDataset(Dataset[dict[str, torch.Tensor]]):
    webcam_dataset: WebcamDataset

    label_names: list[str] = field(
        default_factory=lambda: DEFAULT_LABELS)

    seg_label_names: list[str] = field(
        default_factory=lambda: DEFAULT_SEG_LABELS)
    seg_label2id: dict[str, int] = field(
        default_factory=lambda: DEFAULT_SEG_LABEL2ID)

    seg_colors: list[Color] = field(
        default_factory=lambda: DEFAULT_SEG_COLORS)
    seg_color2id: dict[Color, int] = field(
        default_factory=lambda: DEFAULT_SEG_COLOR2ID)

    frames: list[WebcamFrame] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        annotated_webcams = self.webcam_dataset.get_webcams_with_status(
            WebcamStatus.PARTIALLY_ANNOTATED
        )

        def is_manually_annotated_frame(frame: WebcamFrame) -> bool:
            return frame.status == WebcamFrameStatus.MANUALLY_ANNOTATED

        for webcam in annotated_webcams:
            self.frames.extend(
                filter(is_manually_annotated_frame, webcam.frames)
            )

    @staticmethod
    def feature_extractor(
        seg_color2id: dict[Color, int],
        image: Image,
        segmentation: Image | None = None,
    ) -> dict[str, torch.Tensor]:

        encoded_inputs: dict[str, torch.Tensor] = SegformerImageProcessor()(
            image,
            segmentation,
            size=DEFAULT_FEATURE_EXTRACTOR_IMG_SIZE,
            return_tensors="pt",
        )

        if segmentation:
            # Use shape (num_batches, num_channels, height, width)
            # for segmentation mask
            encoded_inputs["labels"] = (
                encoded_inputs["labels"].permute((0, 3, 1, 2))
            )

            encoded_inputs["seg_labels"] = rgb_pixels_to_1d(
                encoded_inputs["labels"],
                seg_color2id
            )
            # Reserve "labels" key for weather labels like "cloudy"
            del encoded_inputs["labels"]

        for categories, values in encoded_inputs.items():
            values.squeeze_()

        return encoded_inputs

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame = self.frames[idx]

        encoded_inputs = Weather2InfoDataset.feature_extractor(
            self.seg_color2id, frame.image, frame.segmentation,
        )

        if frame.labels:
            values = [frame.labels[label_name]
                      for label_name in self.label_names]
            encoded_inputs["labels"] = torch.FloatTensor(values)

        return encoded_inputs
