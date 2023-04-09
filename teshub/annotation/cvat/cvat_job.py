# cvat-sdk does not have typing stubs :(
# mypy: disable_error_code = "misc, import, no-any-unimported"

from dataclasses import dataclass

from cvat_sdk.api_client.model.annotations_read import AnnotationsRead
from cvat_sdk.api_client.model.data_meta_read import DataMetaRead
from cvat_sdk.api_client.model.job_read import JobRead


@dataclass
class CVATJob:
    job: JobRead
    job_media_data: DataMetaRead
    annotation_metadata: AnnotationsRead
    raw_job_annotations: bytes
    task_name: str
    project_labels: dict[str, str]
