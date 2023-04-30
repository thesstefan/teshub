import logging
import os
from dataclasses import dataclass
from typing import ClassVar

from teshub.annotation.cvat.cvat_converter import CVATConverter
from teshub.annotation.cvat.cvat_manager import CVATManager
from teshub.annotation.webcam_annotation_job import WebcamAnnotationJob
from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.webcam.webcam_stream import WebcamStatus


@dataclass
class WebcamAnnotationManager:
    # TODO: May define a generic annotation manager class to allow easy
    # integration with other annotation tools
    annotation_manager: CVATManager
    webcam_dataset: WebcamDataset

    # TODO: Make these adjustable
    MAX_LABEL_COUNT: ClassVar[int] = 50
    JOB_EXPORT_FORMAT: ClassVar[str] = "Segmentation mask 1.1"

    def create_tasks(
        self,
        task_count: int,
        project_id: int,
        owner_id: int,
        assignee_id: int,
    ):
        webcams = self.webcam_dataset.get_webcams_with_status(
            WebcamStatus.DOWNLOADED, task_count
        )

        logging.info(f"Creating {len(webcams)} annotation jobs...")

        for webcam in webcams:
            try:
                self.annotation_manager.create_task(
                    webcam.id,
                    project_id,
                    owner_id,
                    assignee_id,
                    [
                        os.path.join(webcam.id, frame.file_name)
                        for frame in webcam.frames
                    ],
                )
            except Exception as e:
                # TODO: See what errors are common and handle
                # them accordingly. Stop when a fatal one is encountered.
                logging.error(e)

                continue

            self.webcam_dataset.update_webcam_status(
                webcam.id, WebcamStatus.TASK_CREATED
            )

            logging.info(
                f"Successfully created task {webcam.id} "
                f"with {len(webcam.frames)} images\n"
            )

    def get_job(self, webcam_id: str, project_id: int) -> WebcamAnnotationJob:
        logging.info(f"Fetching annotation job {webcam_id}...")

        return CVATConverter.to_webcam_job(
            self.annotation_manager.get_cvat_job(
                webcam_id,
                project_id,
                WebcamAnnotationManager.JOB_EXPORT_FORMAT,
                WebcamAnnotationManager.MAX_LABEL_COUNT,
            )
        )

    def sync(self, project_id: int):
        logging.info("Synchronizing annotation jobs with dataset...")

        webcams = self.webcam_dataset.get_webcams_with_status(
            WebcamStatus.TASK_CREATED
        )
        synchronized_jobs_count: int = 0

        for webcam in webcams:
            job = self.get_job(webcam.id, project_id)

            if job.job_state in ["new", "in_progress"]:
                continue

            if job.job_state == "rejected":
                logging.info(
                    f"Annotation job for webcam {webcam.id} is rejected. "
                    "Removing webcam...\n"
                )

                self.webcam_dataset.delete_webcam(webcam.id)
                synchronized_jobs_count += 1

                continue

            if job.job_state == "completed":
                logging.info(
                    f"Annotation job for webcam {webcam.id} is completed. "
                    "Starting sync...\n"
                )

                self.webcam_dataset.delete_frames(
                    webcam.id, job.deleted_frames
                )

                self.webcam_dataset.save_annotations(
                    webcam.id,
                    job.segmentation_masks,
                    job.labels,
                    synthetic=True,
                )
                synchronized_jobs_count += 1

                logging.info(
                    f"Annotation job for webcam {webcam.id} "
                    "synchronized succesfully.\n"
                )

        logging.info(f"Synchronized {synchronized_jobs_count} job(s)...")
