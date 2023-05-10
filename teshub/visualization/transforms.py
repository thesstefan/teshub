import torch
from PIL import Image
from teshub.extra_typing import Color
from typing import cast

# No stubs for transforms yet:
#   https://github.com/pytorch/vision/issues/2025
from torchvision.transforms.functional import (  # type: ignore
    to_pil_image
)


def seg_mask_to_image(
    seg_mask: torch.Tensor,
    colors: list[Color],
) -> tuple[Image.Image, list[Color]]:
    assert (len(seg_mask.shape) == 2)

    seg_mask_3d: torch.Tensor = (
        torch.clone(seg_mask).unsqueeze(dim=2).expand(-1, -1, 3).byte()
    )

    for id, color in enumerate(colors):
        seg_mask_3d[seg_mask == id] = torch.tensor(color, dtype=torch.uint8)

    image: Image.Image = to_pil_image(seg_mask_3d.permute(2, 0, 1))

    # TODO: This seems unsually slow. Investigate more and decide if it's needed.
    unique_color_tensors: torch.Tensor = torch.unique(
        seg_mask_3d.reshape(-1, 3), dim=0)
    colors_used: list[Color] = unique_color_tensors.tolist()

    return image, colors_used


def rgb_pixels_to_1d(
    pixel_values: torch.Tensor,
    rgb_pixel_to_value: dict[Color, int]
) -> torch.Tensor:
    values_1d: list[int] = []
    color_tensor: torch.Tensor

    for color_tensor in pixel_values.view(-1, 3):
        color_list: list[int] = color_tensor.tolist()
        color_tuple = cast(Color, tuple(color_list))

        values_1d.append(rgb_pixel_to_value[color_tuple])

    return torch.tensor(values_1d).view(
        *pixel_values.shape[-2:], 1
    )
