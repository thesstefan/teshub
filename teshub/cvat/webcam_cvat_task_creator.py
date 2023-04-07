import logging
from dataclasses import dataclass

from teshub.cvat.cvat_task_creator import CVATTaskCreator
from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.webcam.webcam_stream import WebcamStatus


@dataclass
class WebcamAdnotationTaskCreator:
    task_creator: CVATTaskCreator
    webcam_dataset: WebcamDataset

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

        for webcam in webcams:
            try:
                self.task_creator.create_task(
                    webcam.id,
                    project_id,
                    owner_id,
                    assignee_id,
                    [frame.file_name for frame in webcam.frames],
                )
            except Exception as e:
                logging.error(e)

                continue

            self.webcam_dataset.update_webcam_status(
                webcam.id, WebcamStatus.TASK_CREATED
            )

            logging.info(
                f"Successfully created task {webcam.id} "
                f"with {len(webcam.frames)} images\n"
            )
