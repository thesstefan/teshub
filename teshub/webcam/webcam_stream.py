from dataclasses import dataclass, field
from enum import Enum
from typing import List


class WebcamStatus(Enum):
    NONE: str = "NONE"
    DOWNLOADED: str = "DOWNLOADED"
    VERIFIED: str = "VERIFIED"
    RUN_INFERENCE: str = "RUN_INFERENCE"
    VERIFIED_INFERENCE: str = "VERIFIED_WITH_INFERENCE"


@dataclass
class WebcamLocation:
    city: str
    region: str
    country: str
    continent: str
    latitude: float
    longitude: float


@dataclass
class WebcamStream:
    id: int
    categories: List[str]
    location: WebcamLocation
    status: WebcamStatus
    image_urls: List[str] = field(init=False, default_factory=list)
