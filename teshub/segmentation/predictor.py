from dataclasses import field
import torch
from dataclasses import dataclass
from torch import nn

from teshub.segmentation.weather2seg import Weather2SegDataset
from teshub.segmentation.weather_segformer import WeatherSegformer

from transformers import SegformerImageProcessor  # type: ignore[import]

from PIL import Image


@dataclass
class SegmentationPredictor:
    model_checkpoint_path: str
    pretrained_model_name: str
    map_location: torch.device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    model: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        self.model = WeatherSegformer.load_from_checkpoint(  # type: ignore
            self.model_checkpoint_path,
            map_location=self.map_location,
            pretrained_model=self.pretrained_model_name
        )

    def predict(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        pixel_values = Weather2SegDataset.feature_extractor(image)[
            "pixel_values"]

        batch_size = 2
        pixel_values_batch = pixel_values.repeat(batch_size, 1, 1)

        outputs: tuple[torch.Tensor, ...] = self.model(pixel_values_batch)
        predicted: torch.Tensor = SegformerImageProcessor(  # type: ignore
        ).post_process_semantic_segmentation(outputs)

        return predicted
