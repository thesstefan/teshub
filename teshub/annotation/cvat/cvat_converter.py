# cvat-sdk does not have typing stubs :(
# mypy: disable_error_code = "misc, import, no-any-unimported, no-any-return"

import io
import os
import zipfile
from collections import defaultdict
from typing import Callable, DefaultDict

from cvat_sdk.api_client.model.annotations_read import AnnotationsRead
from cvat_sdk.api_client.model.data_meta_read import DataMetaRead

from teshub.annotation.cvat.cvat_job import CVATJob
from teshub.annotation.webcam_annotation_job import WebcamAnnotationJob


class CVATConverter:
    # TODO: Private static methods don't feel right.
    # Should a simple function be used instead of this class?
    @staticmethod
    def _get_deleted_frames(job_media_data: DataMetaRead) -> list[str]:
        deleted_ids: list[int] = job_media_data.deleted_frames

        return [job_media_data.frames[id].name for id in deleted_ids]

    @staticmethod
    def _get_segmentation_masks(
        raw_job_annotations: bytes, task_name: str
    ) -> dict[str, bytes]:
        zip_file = zipfile.ZipFile(io.BytesIO(raw_job_annotations), "r")

        is_segmentation: Callable[[str], bool] = lambda path: path.startswith(
            f"SegmentationClass/{task_name}/"
        )

        return {
            os.path.basename(path).replace(".png", "_seg.png"): zip_file.read(
                path
            )
            for path in filter(is_segmentation, zip_file.namelist())
        }

    @staticmethod
    def _get_labels(
        job_annotation_metadata: AnnotationsRead,
        project_labels: dict[str, str],
        job_media_data: DataMetaRead,
    ) -> dict[str, dict[str, float]]:
        labels: DefaultDict[str, dict[str, float]] = defaultdict(
            lambda: {label_name: 0.0 for label_name in project_labels.values()}
        )

        # Find frames with segmentation masks available
        # Tags are not used in CVAT if the value of the label is 0.0,
        # so this allows labelling images with no tags
        annotated_frames = [
            job_media_data.frames[shape.frame].name
            for shape in job_annotation_metadata.shapes
        ]
        for frame in annotated_frames:
            _ = labels[frame]

        for tag in job_annotation_metadata.tags:
            frame_name = job_media_data.frames[tag.frame].name

            labels[frame_name][project_labels[tag.label_id]] = (
                float(tag.attributes[0].value) if tag.attributes else 0.0
            )

        return dict(labels)

    @staticmethod
    def to_webcam_job(cvat_job: CVATJob) -> WebcamAnnotationJob:
        return WebcamAnnotationJob(
            cvat_job.task_name,
            cvat_job.job.state,
            CVATConverter._get_deleted_frames(cvat_job.job_media_data),
            CVATConverter._get_segmentation_masks(
                cvat_job.raw_job_annotations, cvat_job.task_name
            ),
            CVATConverter._get_labels(
                cvat_job.annotation_metadata,
                cvat_job.project_labels,
                cvat_job.job_media_data,
            ),
        )
