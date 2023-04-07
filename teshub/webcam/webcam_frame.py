from dataclasses import dataclass
from enum import Enum
from typing import Optional


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

    labels: Optional[dict[str, float]] = None
    segmentation_path: Optional[str] = None

    url: Optional[str] = None

    @property
    def image(self) -> None:
        raise NotImplementedError

    @property
    def segmentation(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"WebcamFrame(file_name={self.file_name}, status={self.status})"
