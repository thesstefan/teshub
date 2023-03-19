import logging
from dataclasses import dataclass
from typing import List

from cvat_sdk.api_client import ApiClient, Configuration, exceptions
from cvat_sdk.api_client.model.data_request import DataRequest
from cvat_sdk.api_client.model.storage_type import StorageType
from cvat_sdk.api_client.model.task_read import TaskRead
from cvat_sdk.api_client.model.task_write_request import TaskWriteRequest


@dataclass
class CVATTaskCreator:
    config: Configuration
    use_shared_storage: bool = True

    def _create_empty_task(
        self, task_name: str, project_id: int, owner_id: int, assignee_id: int
    ) -> int:
        with ApiClient(self.config) as api_client:
            task_write_request = TaskWriteRequest(
                name=task_name,
                project_id=project_id,
                owner_id=owner_id,
                assignee_id=assignee_id,
            )

            data: TaskRead
            (data, repsponse) = api_client.tasks_api.create(task_write_request)

            return data.id

    def _create_task_data(self, task_id: int, image_paths: List[str]) -> None:
        with ApiClient(self.config) as api_client:
            data_request = DataRequest(
                image_quality=100,
                storage=StorageType(
                    "share" if self.use_shared_storage else "local"
                ),
                server_files=image_paths,
            )

            api_client.tasks_api.create_data(task_id, data_request)

    def create_task(
        self,
        task_name: str,
        project_id: int,
        owner_id: int,
        assignee_id: int,
        image_paths: List[str],
    ) -> None:
        logging.info(
            f"Creating CVAT task {task_name} with given configuration..."
        )

        task_id = self._create_empty_task(
            task_name, project_id, owner_id, assignee_id
        )

        logging.info(
            f"Adding {len(image_paths)} images to task "
            f"{task_name} (ID: {task_id})..."
        )

        self._create_task_data(task_id, image_paths)
