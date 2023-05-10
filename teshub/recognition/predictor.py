from dataclasses import field, dataclass
import torch

from teshub.recognition.weather2info import Weather2InfoDataset
from teshub.recognition.weather_informer import WeatherInFormer
from teshub.recognition.utils import (
    upsample_logits, load_model_hyperparams_from_checkpoint, NestedTorchDict,
    DEFAULT_SEG_COLOR2ID, DEFAULT_LABELS, DEFAULT_SEG_LABELS,
    DEFAULT_SEG_LABEL2ID, DEFAULT_SEG_COLORS
)
from typing import cast
from teshub.visualization.transforms import seg_mask_to_image
from teshub.extra_typing import Color

from PIL import Image


@dataclass
class WeatherInFormerPredictor:
    model_checkpoint_path: str

    map_location: torch.device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    pretrained_segformer_model: str = field(init=False)
    model_batch_size: int = field(init=False)
    model: WeatherInFormer = field(init=False)

    def __post_init__(self) -> None:
        hparams: dict[str, NestedTorchDict] = (
            load_model_hyperparams_from_checkpoint(
                self.model_checkpoint_path, device=self.map_location)
        )
        assert hparams

        self.model_batch_size = cast(int, hparams['batch_size'])
        self.pretrained_segformer_model = cast(
            str, hparams['pretrained_segformer_model'])

        self.model = WeatherInFormer.load_from_checkpoint(  # type: ignore
            self.model_checkpoint_path,
            label_names=DEFAULT_LABELS,
            seg_label_names=DEFAULT_SEG_LABELS,
            seg_label2id=DEFAULT_SEG_LABEL2ID,
            map_location=self.map_location,
            pretrained_model=self.pretrained_segformer_model,

        )

    def predict(
        self,
        image: str | Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(image, str):
            image = Image.open(image)

        pixel_values = (
            Weather2InfoDataset.feature_extractor(
                seg_color2id=DEFAULT_SEG_COLOR2ID, image=image)["pixel_values"]
        )

        pixel_values_batch = pixel_values.repeat(
            self.model_batch_size, 1, 1, 1
        ).to(self.map_location)

        outputs: tuple[tuple[torch.Tensor], tuple[torch.Tensor]] = (
            self.model(pixel_values_batch)
        )
        seg_mask_output = outputs[0][0]
        predicted_labels = outputs[1][0]

        predicted_seg = upsample_logits(
            seg_mask_output, size=torch.Size([image.size[1], image.size[0]]))

        return predicted_seg.cpu(), predicted_labels.cpu()

    def predict_and_process(
        self,
        image: str | Image.Image,
    ) -> tuple[Image.Image, dict[str, float], dict[str, Color]]:
        seg_prediction: torch.Tensor
        labels_prediction: torch.Tensor

        seg_prediction, labels_prediction = self.predict(image)
        seg_image, used_colors = seg_mask_to_image(
            seg_prediction[0], DEFAULT_SEG_COLORS)

        color_map = {
            DEFAULT_SEG_LABELS[
                DEFAULT_SEG_COLOR2ID[
                    cast(tuple[int, int, int], tuple(color))
                ]
            ]:
                color for color in used_colors
        }
        label_dict = dict(
            zip(DEFAULT_LABELS,
                cast(list[float], labels_prediction[0].tolist()))
        )

        return seg_image, label_dict, color_map

