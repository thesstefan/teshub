from dataclasses import dataclass, field

import torch
from PIL.Image import Image
from torch.utils.data import Dataset

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus
from teshub.webcam.webcam_stream import WebcamStatus
from teshub.segmentation.utils import DEFAULT_ID2COLOR, DEFAULT_COLOR2ID

from transformers import SegformerImageProcessor  # type: ignore[import]


@dataclass
class Weather2SegDataset(Dataset[dict[str, torch.Tensor]]):
    webcam_dataset: WebcamDataset

    color2id: dict[tuple[int, ...], int] = field(
        default_factory=lambda: DEFAULT_COLOR2ID)
    id2color: dict[int, tuple[int, ...]] = field(
        default_factory=lambda: DEFAULT_ID2COLOR)

    frames: list[WebcamFrame] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        annotated_webcams = self.webcam_dataset.get_webcams_with_status(
            WebcamStatus.PARTIALLY_ANNOTATED
        )

        def is_manually_annotated_frame(frame: WebcamFrame) -> bool:
            # TODO: Change this to manual annotation
            return frame.status == WebcamFrameStatus.SYNTHETIC_ANNOTATION

        for webcam in annotated_webcams:
            self.frames.extend(
                filter(is_manually_annotated_frame, webcam.frames)
            )

    @staticmethod
    def feature_extractor(
        image: Image,
        segmentation: Image | None = None,
        color2id: dict[tuple[int, ...], int] = DEFAULT_COLOR2ID
    ) -> dict[str, torch.Tensor]:

        encoded_inputs: dict[str, torch.Tensor] = SegformerImageProcessor()(
            image,
            segmentation,
            return_tensors="pt",
        )

        # TODO: Find a better way of doing this and not break mypy
        labels_1d: list[int] = []
        color_tensor: torch.Tensor

        if segmentation:
            for color_tensor in encoded_inputs["labels"].view(-1, 3):
                color_list: list[int] = color_tensor.tolist()
                color_tuple = tuple(color_list)

                labels_1d.append(color2id[color_tuple])

            encoded_inputs["labels"] = torch.tensor(
                labels_1d).view(512, 512, 1)

        for categories, values in encoded_inputs.items():
            values.squeeze_()

        return encoded_inputs

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded_inputs = self.feature_extractor(
            self.frames[idx].image,
            self.frames[idx].segmentation,
        )

        return encoded_inputs
