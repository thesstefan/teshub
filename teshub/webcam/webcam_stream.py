from dataclasses import dataclass, field
from enum import Enum
from typing import List


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
class WebcamLocation:
    city: str
    region: str
    country: str
    continent: str
    latitude: float
    longitude: float


@dataclass
class WebcamStream:
    id: str
    categories: List[str]
    location: WebcamLocation
    status: WebcamStatus
    image_urls: List[str] = field(init=False, default_factory=list)
    image_paths: List[str] = field(init=False, default_factory=list)
