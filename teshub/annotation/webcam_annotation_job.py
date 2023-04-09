from dataclasses import dataclass


@dataclass
class WebcamAnnotationJob:
    webcam_id: str
    job_state: str
    deleted_frames: list[str]
    segmentation_masks: dict[str, bytes]
    labels: dict[str, dict[str, float]]
