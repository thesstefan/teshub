import functools
import itertools 
import json
import os
from collections import defaultdict
from enum import Enum
from typing import Callable, List, Mapping, Tuple

import attrs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.io as sio  # type: ignore
import torch
import torchvision  # type: ignore
from torch.utils.data import Dataset


class WeatherType(Enum):
    CLOUDY = 0
    CLEAR = 1


# For now the default is resizing to 300x300 since segmentation and
# classification is ran this way. Could be made more flexible later.
IMAGE_SIZE: Tuple[int, int] = (300, 300)


@attrs.define
class WindyDataset(Dataset):
    """Windy dataset, initial version for Cloudy/Clear weather."""

    root_dir: str
    source_category: WeatherType
    target_category: WeatherType
    _img_pairs: List[Tuple[str, str]] = attrs.field(
        init=False, default=attrs.Factory(list)
    )
    _transform: Callable[[npt.NDArray], npt.NDArray] = attrs.field(
        init=False, default=torchvision.transforms.Resize(IMAGE_SIZE)
    )

    def _img_pairs_from_stream(
        self,
        stream_dir: str,
    ) -> List[Tuple[str, str]]:
        with open(
            os.path.join(stream_dir, "predictions.json")
        ) as json_prediction_data:
            prediction_data = json.load(json_prediction_data)
            img_by_weather: Mapping[WeatherType, List[str]] = defaultdict(list)

            for img_path, img_prediction in prediction_data.items():
                weather_type: WeatherType = WeatherType(
                    np.argmax(img_prediction["prob"])
                )
                img_by_weather[weather_type].append(
                    os.path.join(stream_dir, img_path)
                )

        return [
            img_pair
            for img_pair in itertools.product(
                img_by_weather[self.source_category],
                img_by_weather[self.target_category],
            )
        ]

    def __attrs_post_init__(self) -> None:
        for stream_dir in os.listdir(self.root_dir):
            self._img_pairs.extend(
                self._img_pairs_from_stream(
                    os.path.join(self.root_dir, stream_dir)
                )
            )

    def __len__(self) -> int:
        return len(self._img_pairs)

    def __getitem__(self, index: int):
        source_path, target_path = self._img_pairs[index]

        @functools.lru_cache(maxsize=512)
        def read_image(image_path: str):
            return torchvision.io.read_image(image_path).type(torch.float32)

        source_path, target_path = self._img_pairs[index]

        source_img = read_image(source_path)
        target_img = read_image(target_path)

        segmentation_mask = np.array(
            sio.loadmat(f"{source_path}_seg.mat").get("mask")
        )

        return {
            "source_img": self._transform(source_img),
            "segmentation_mask": segmentation_mask,
            "target_img": self._transform(target_img),
        }
