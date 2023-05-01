from dataclasses import dataclass, field
from typing import ClassVar

import torch
from transformers import SegformerForSemanticSegmentation


@dataclass(eq=False)
class WeatherSegformer(torch.nn.Module):
    pretrained_model: str = "nvidia/mit-b0"
    model: torch.nn.Module = field(init=False)

    id2label: ClassVar[dict[int, str]] = {
        0: "background",
        1: "black_clouds",
        2: "white_clouds",
        3: "blue_sky",
        4: "gray_sky",
        5: "white_sky",
        6: "fog",
        7: "sun",
        8: "snow",
        9: "shadow",
        10: "wet_ground",
    }
    label2id: ClassVar[dict[str, int]] = {
        "background": 0,
        "black_clouds": 1,
        "white_clouds": 2,
        "blue_sky": 3,
        "gray_sky": 4,
        "white_sky": 5,
        "fog": 6,
        "sun": 7,
        "snow": 8,
        "shadow": 9,
        "wet_ground": 10,
    }

    def __post_init__(self) -> None:
        super().__init__()

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_model,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.segformer(
            pixel_values=pixel_values, labels=labels, return_dict=False
        )
