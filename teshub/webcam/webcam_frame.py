import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PIL import Image


class WebcamFrameStatus(str, Enum):
    NONE: str = "NONE"
    DOWNLOADED: str = "DOWNLOADED"
    DELETED: str = "DELETED"
    MANUAL_ANNOTATION: str = "MANUAL_ANNOTATION"
    SYNTHETIC_ANNOTATION: str = "SYNTHETIC_ANNOTATION"

    def __str__(self) -> str:
        return self.name


@dataclass
class WebcamFrame:
    file_name: str
    status: WebcamFrameStatus = WebcamFrameStatus.NONE

    webcam_id: Optional[str] = None

    labels: Optional[dict[str, float]] = None
    segmentation_path: Optional[str] = None

    url: Optional[str] = None
    load_dir: Optional[str] = None

    @property
    def image(self) -> Image.Image:
        if not self.load_dir:
            raise RuntimeError(
                f"Image not available for frame {self.file_name}"
            )

        return Image.open(os.path.join(self.load_dir, self.file_name))

    @property
    def segmentation(self) -> Image.Image:
        if not self.load_dir or not self.segmentation_path:
            raise RuntimeError(
                f"Segmentation not available for frame {self.file_name}"
            )

        return Image.open(
            os.path.join(self.load_dir, self.segmentation_path)
        )

    def __str__(self) -> str:
        return f"WebcamFrame(file_name={self.file_name}, status={self.status})"
