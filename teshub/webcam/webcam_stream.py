from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from teshub.webcam.webcam_frame import WebcamFrame


class WebcamStatus(str, Enum):
    NONE: str = "NONE"
    DOWNLOADED: str = "DOWNLOADED"
    TASK_CREATED: str = "TASK_CREATED"
    PARTIALLY_ANNOTATED: str = "PARTIALLY_ANNOTATED"
    DELETED: str = "DELETED"
    ANNOTATED: str = "ANNOTATED"
    VERIFIED: str = "VERIFIED"
    RUN_INFERENCE: str = "RUN_INFERENCE"
    VERIFIED_INFERENCE: str = "VERIFIED_WITH_INFERENCE"

    def __str__(self) -> str:
        return self.name


@dataclass
class WebcamStream:
    id: str
    status: WebcamStatus

    image_count: Optional[int] = None
    categories: Optional[list[str]] = None

    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    continent: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    frames: list[WebcamFrame] = field(init=False, default_factory=list)

    def __str__(self) -> str:
        return (
            f"WebcamStream(id={self.id}, status={self.status}, "
            f"image_count={self.image_count})"
        )
