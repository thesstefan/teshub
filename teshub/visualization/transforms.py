from typing import cast

import torch
from PIL import Image
# No stubs for transforms yet:
#   https://github.com/pytorch/vision/issues/2025
from torchvision.transforms.functional import to_pil_image  # type: ignore

from teshub.extra_typing import Color


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

    # Expected shapes: (batch_size, num_channels=3, height, width)
    # or (num_channels=3, height, width)
    assert pixel_values.shape[-3] == 3 and len(pixel_values.shape) in [3, 4]

    # Move color channel to last dimension
    for color_tensor in pixel_values.transpose(-3, -1).reshape(-1, 3):
        color_list: list[int] = color_tensor.tolist()
        color_tuple = cast(Color, tuple(color_list))

        values_1d.append(rgb_pixel_to_value[color_tuple])

    shape_1d: tuple[int, ...] = (
        pixel_values.shape[0], 1, *pixel_values.shape[-2:])

    # If batch size is not used, remove placeholder from shape
    if len(pixel_values.shape) == 3:
        shape_1d = shape_1d[1:]

    return torch.tensor(values_1d).view(shape_1d)
