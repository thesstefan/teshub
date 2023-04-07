from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class WebcamStatus(Enum):
    NONE: str = "NONE"
    DOWNLOADED: str = "DOWNLOADED"
    TASK_CREATED: str = "TASK_CREATED"
    PARTIALLY_ADNOTATED: str = "PARTIALLY_ADNOTATED"
    ADNOTATED: str = "ADNOTATED"
    VERIFIED: str = "VERIFIED"
    RUN_INFERENCE: str = "RUN_INFERENCE"
    VERIFIED_INFERENCE: str = "VERIFIED_WITH_INFERENCE"


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

    image_urls: list[str] = field(init=False, default_factory=list)
    image_paths: list[str] = field(init=False, default_factory=list)
