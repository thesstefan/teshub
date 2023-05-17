import random
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Type

import torch
from PIL import Image
from torch.utils.data import Dataset

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.extra_typing import Color
# TODO: Move this in a common directory
from teshub.recognition.utils import DEFAULT_LABELS, DEFAULT_SEG_COLOR2ID
from teshub.translation.config.abc_config import TranslationConfig
from teshub.webcam.webcam_frame import WebcamFrame
from teshub.webcam.webcam_stream import WebcamStatus


@dataclass
class Weather2WeatherDataset(
        Dataset[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]):
    webcam_dataset: WebcamDataset

    select_source: Callable[[WebcamFrame], bool]
    select_target: Callable[[WebcamFrame], bool]

    img_transform: Callable[[Image.Image], torch.Tensor]
    seg_transform: Callable[[Image.Image], torch.Tensor] | None = None
    att_transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    max_pairs_per_webcam: int = 10
    return_att: bool = True
    return_seg: bool = True
    return_labels: bool = True

    frame_pairs: list[tuple[WebcamFrame, WebcamFrame]] = field(
        init=False, default_factory=list
    )
    seg_color2id: dict[Color, int] = field(
        default_factory=lambda: DEFAULT_SEG_COLOR2ID
    )
    label_names: list[str] = field(
        default_factory=lambda: DEFAULT_LABELS)

    def __post_init__(self) -> None:
        annotated_webcams = self.webcam_dataset.get_webcams_with_status(
            WebcamStatus.ANNOTATED
        )

        for webcam in annotated_webcams:
            source_frames = filter(self.select_source, webcam.frames)
            target_frames = filter(self.select_target, webcam.frames)

            frame_pairs = list(product(source_frames, target_frames))

            if len(frame_pairs) > self.max_pairs_per_webcam:
                self.frame_pairs.extend(random.sample(
                    frame_pairs, self.max_pairs_per_webcam))
            else:
                self.frame_pairs.extend(frame_pairs)

    def __len__(self) -> int:
        return len(self.frame_pairs)

    def _get_image_data(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_frame, target_frame = self.frame_pairs[idx]

        source_image, target_image = (
            self.img_transform(source_frame.image),
            self.img_transform(target_frame.image)
        )
        # TODO: Fix mypy error.
        if not isinstance(source_image, torch.Tensor):  # type: ignore[misc]
            raise RuntimeError(
                "Image transforms used in Weather2WeatherDataset "
                "must return tensors"
            )

        if not self.seg_transform or (
            not self.return_seg and (
                not self.att_transform or not self.return_att
            )
        ):
            return source_image, target_image

        source_seg, target_seg = (
            self.seg_transform(source_frame.segmentation),
            self.seg_transform(target_frame.segmentation)
        )
        # TODO: Fix mypy error.
        if not isinstance(source_image, torch.Tensor):  # type: ignore[misc]
            raise RuntimeError(
                "Image transforms used in Weather2WeatherDataset "
                "must return tensors"
            )

        if self.return_seg and (not self.att_transform or not self.return_att):
            return (
                torch.cat([source_image, source_seg], dim=0),
                torch.cat([target_image, target_seg], dim=0)
            )

        assert self.att_transform

        source_att, target_att = (
            self.att_transform(source_seg),
            self.att_transform(target_seg)
        )
        # TODO: Fix mypy error.
        if not isinstance(source_image, torch.Tensor):  # type: ignore[misc]
            raise RuntimeError(
                "Image transforms used in Weather2WeatherDataset "
                "must return tensors"
            )

        if not self.return_seg:
            return (
                torch.cat([source_image, source_att], dim=0),
                torch.cat([target_image, target_att], dim=0)
            )

        return (
            torch.cat([source_image, source_att, source_seg], dim=0),
            torch.cat([target_image, target_att, target_seg], dim=0)
        )

    def __getitem__(
        self,
        idx: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        source_frame, target_frame = self.frame_pairs[idx]

        source_image_data, target_image_data = self._get_image_data(idx)

        if not self.return_labels:
            return (source_image_data, ), (target_image_data, )

        assert source_frame.labels and target_frame.labels

        return (
            # TODO: Fix mypy errors.
            (source_image_data, torch.FloatTensor(
                [source_frame.labels[label_name]
                 for label_name in self.label_names]
            )),
            (target_image_data, torch.FloatTensor(
                [target_frame.labels[label_name]
                 for label_name in self.label_names]
            )),
        )

    @staticmethod
    def from_translation_config(
        webcam_dataset: WebcamDataset,
        translation_config: Type[TranslationConfig],
        max_pairs_per_webcam: int = 10,
        return_seg: bool = True,
        return_att: bool = True,
        return_labels: bool = True,
        seg_color2id: dict[Color, int] = DEFAULT_SEG_COLOR2ID
    ) -> 'Weather2WeatherDataset':
        select_source, select_target = translation_config.frame_selectors()
        img_transform, seg_transform, att_transform = (
            translation_config.frame_transforms()
        )

        return Weather2WeatherDataset(
            webcam_dataset,
            select_source=select_source,
            select_target=select_target,
            img_transform=img_transform,
            seg_transform=seg_transform,
            att_transform=att_transform,
            max_pairs_per_webcam=max_pairs_per_webcam,
            return_seg=return_seg,
            return_att=return_att,
            return_labels=return_labels,
            seg_color2id=seg_color2id
        )
