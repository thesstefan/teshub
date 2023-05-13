from dataclasses import dataclass, field
from typing import Generator, cast

import torch
from PIL import Image

from teshub.extra_typing import Color
from teshub.recognition.utils import (DEFAULT_FEATURE_EXTRACTOR_IMG_SIZE,
                                      DEFAULT_LABELS, DEFAULT_SEG_COLOR2ID,
                                      DEFAULT_SEG_COLORS, DEFAULT_SEG_LABEL2ID,
                                      DEFAULT_SEG_LABELS, NestedTorchDict,
                                      load_model_hyperparams_from_checkpoint,
                                      upsample_logits)
from teshub.recognition.weather2info import Weather2InfoDataset
from teshub.recognition.weather_informer import WeatherInFormer
from teshub.visualization.transforms import seg_mask_to_image


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

    def _predict_batch(
        self,
        image_batch: list[str | Image.Image],
        image_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(image_batch) <= self.model_batch_size

        # TODO: Not sure if we should use (height, width)
        # or (width, height) here.
        pixel_values_shape = (3, * DEFAULT_FEATURE_EXTRACTOR_IMG_SIZE.values())
        pixel_values_batch = torch.empty(
            (self.model_batch_size, *pixel_values_shape)
        )

        for idx, image in enumerate(image_batch):
            if isinstance(image, str):
                image = Image.open(image)

                if image.size != image_size:
                    raise RuntimeError(
                        "All images in batch must have the same size")

            pixel_values_batch[idx] = (
                Weather2InfoDataset.feature_extractor(
                    seg_color2id=DEFAULT_SEG_COLOR2ID, image=image
                )["pixel_values"]
            )

        pixel_values_batch = pixel_values_batch.to(self.map_location)

        outputs: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] = (
            self.model(pixel_values_batch)
        )
        seg_mask_output, predicted_labels = outputs

        predicted_seg_mask = upsample_logits(
            seg_mask_output[0], size=torch.Size(image_size[::-1])
        )

        return predicted_seg_mask, predicted_labels[0]

    def predict(
        self,
        image_list: list[str | Image.Image]
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        image_size = (
            image_list[0].size
            if isinstance(image_list[0], Image.Image)
            else Image.open(image_list[0]).size
        )

        for idx in range(0, len(image_list), self.model_batch_size):
            image_batch = image_list[idx:idx + self.model_batch_size]

            seg_mask, labels = self._predict_batch(image_batch, image_size)
            filler_image_count = self.model_batch_size - len(image_batch)

            if filler_image_count:
                seg_mask = seg_mask[:-filler_image_count]
                labels = labels[:-filler_image_count]

            batch_predictions: list[tuple[torch.Tensor, torch.Tensor]] = [
                (torch.squeeze(_seg_mask), torch.squeeze(_labels))
                for _seg_mask, _labels in zip(
                    torch.chunk(seg_mask, len(image_batch), dim=0),
                    torch.chunk(labels, len(image_batch), dim=0)
                )
            ]

            for prediction in batch_predictions:
                yield prediction

    def _process_output(
        self, outputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Image.Image, dict[str, float], dict[str, Color]]:
        seg_mask_prediction, labels_prediction = outputs

        seg_mask_image, used_colors = seg_mask_to_image(
            seg_mask_prediction, DEFAULT_SEG_COLORS)

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
                cast(list[float], labels_prediction.tolist()))
        )

        return seg_mask_image, label_dict, color_map

    def predict_and_process(
        self,
        image_list: list[str | Image.Image],
    ) -> Generator[
        tuple[Image.Image, dict[str, float], dict[str, Color]],
        None, None
    ]:
        for outputs in self.predict(image_list):
            yield self._process_output(outputs)
