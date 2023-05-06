from dataclasses import field, dataclass
import torch
from torch import nn

from teshub.segmentation.weather2seg import Weather2SegDataset
from teshub.segmentation.weather_segformer import WeatherSegformer
from teshub.segmentation.utils import (
    upsample_logits, load_model_hyperparams_from_checkpoint, NestedTorchDict
)
from typing import cast

from PIL import Image


@dataclass
class SegmentationPredictor:
    model_checkpoint_path: str

    map_location: torch.device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    pretrained_model_name: str = field(init=False)
    model_batch_size: int = field(init=False)
    model: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        hparams: dict[str, NestedTorchDict] = (
            load_model_hyperparams_from_checkpoint(self.model_checkpoint_path)
        )
        assert hparams

        self.model_batch_size = cast(int, hparams['batch_size'])
        self.pretrained_model_name = cast(
            str, hparams['pretrained_model_name'])

        self.model = WeatherSegformer.load_from_checkpoint(  # type: ignore
            self.model_checkpoint_path,
            map_location=self.map_location,
            pretrained_model=self.pretrained_model_name,
        )

    def predict(self, image: str | Image.Image) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image)

        pixel_values = (
            Weather2SegDataset.feature_extractor(image)["pixel_values"]
        )

        pixel_values_batch = pixel_values.repeat(
            self.model_batch_size, 1, 1, 1
        ).to(self.map_location)

        outputs: tuple[torch.Tensor, ...] = self.model(pixel_values_batch)
        predicted = upsample_logits(
            outputs[0], size=torch.Size([image.size[1], image.size[0]]))

        return predicted.cpu()
