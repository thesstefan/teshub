import torch
from PIL import Image
from teshub.extra_typing import Color

# No stubs for transforms yet:
#   https://github.com/pytorch/vision/issues/2025
from torchvision.transforms.functional import (  # type: ignore
    to_pil_image
)


def seg_mask_to_image(
    seg_mask: torch.Tensor, colors: list[Color]
) -> Image.Image:
    assert (len(seg_mask.shape) == 2)

    seg_mask_3d: torch.Tensor = (
        torch.clone(seg_mask).unsqueeze(dim=2).expand(-1, -1, 3).byte()
    )

    for id, color in enumerate(colors):
        seg_mask_3d[seg_mask == id] = torch.tensor(color, dtype=torch.uint8)

    image: Image.Image = to_pil_image(seg_mask_3d.permute(2, 0, 1))

    return image
