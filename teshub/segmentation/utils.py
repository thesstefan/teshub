import torch
from torch import nn
from typing import TypeAlias, cast

DEFAULT_ID2COLOR: dict[int, tuple[int, ...]] = {
    0: (0, 0, 0),
    1: (22, 21, 22),
    2: (204, 204, 204),
    3: (46, 6, 243),
    4: (154, 147, 185),
    5: (198, 233, 255),
    6: (255, 53, 94),
    7: (250, 250, 55),
    8: (255, 255, 255),
    9: (115, 51, 128),
    10: (36, 179, 83),
    11: (119, 119, 119),
}
DEFAULT_COLOR2ID = {
    color: id for (id, color) in DEFAULT_ID2COLOR.items()
}

DEFAULT_ID2LABEL: dict[int, str] = {
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
    11: "shadow_snow"
}
DEFAULT_LABEL2ID = {
    label: id for (id, label) in DEFAULT_ID2LABEL.items()
}


# Should this be moved to teshub.extra_typing?
# Not sure if introducing the torch dependency there is worth it
NestedTorchDict: TypeAlias = (
    dict[str, "NestedTorchDict"] | list["NestedTorchDict"] | str | int |
    float | bool | None | torch.Tensor
)


def upsample_logits(logits: torch.Tensor, size: torch.Size) -> torch.Tensor:
    upsampled_logits: torch.Tensor = nn.functional.interpolate(
        logits, size=size, mode="bilinear", align_corners=False
    )

    return upsampled_logits.argmax(dim=1)


def load_model_hyperparams_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> dict[str, NestedTorchDict]:
    checkpoint: dict[str, NestedTorchDict] = torch.load(
        checkpoint_path, map_location=device)

    return cast(dict[str, NestedTorchDict], checkpoint['hyper_parameters'])
