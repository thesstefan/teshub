# cvat-sdk does not have typing stubs :(
# mypy: disable_error_code = "misc, import, no-any-unimported, no-any-return"

import logging
from dataclasses import dataclass
from http import HTTPStatus
from time import sleep
from typing import Optional

from cvat_sdk.api_client import ApiClient, Configuration
from cvat_sdk.api_client.model.annotations_read import AnnotationsRead
from cvat_sdk.api_client.model.data_meta_read import DataMetaRead
from cvat_sdk.api_client.model.data_request import DataRequest
from cvat_sdk.api_client.model.job_read import JobRead
from cvat_sdk.api_client.model.paginated_job_read_list import \
    PaginatedJobReadList
from cvat_sdk.api_client.model.paginated_label_list import PaginatedLabelList
from cvat_sdk.api_client.model.paginated_task_read_list import \
    PaginatedTaskReadList
from cvat_sdk.api_client.model.storage_type import StorageType
from cvat_sdk.api_client.model.task_read import TaskRead
from cvat_sdk.api_client.model.task_write_request import TaskWriteRequest
from urllib3 import HTTPResponse

from teshub.annotation.cvat.cvat_job import CVATJob


@dataclass
class CVATManager:
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

    def _create_task_data(self, task_id: int, image_paths: list[str]) -> None:
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
        image_paths: list[str],
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

    def get_task(self, task_name: str, project_id: int) -> TaskRead:
        with ApiClient(self.config) as api_client:
            data: PaginatedTaskReadList
            (data, response) = api_client.tasks_api.tasks_list(
                name=task_name, project_id=project_id
            )

            if data.count == 0:
                raise RuntimeError(
                    f"Task {task_name} not found in project "
                    f"with ID {project_id}"
                )

            if data.count > 1:
                raise RuntimeError(
                    f"Multiple tasks with name {task_name} found in "
                    f"project with ID {project_id}. The task name "
                    "should be unique!"
                )

            return data.results[0]

    def get_job(self, task_name: str, project_id: int) -> JobRead:
        with ApiClient(self.config) as api_client:
            data: PaginatedJobReadList
            (data, response) = api_client.jobs_api.list(
                task_name=task_name, project_id=project_id
            )

            if data.count == 0:
                raise RuntimeError(
                    f"No Job found in task {task_name} from project "
                    f"with ID {project_id}"
                )

            if data.count > 1:
                raise RuntimeError(
                    f"Multiple Jobs found with Task name {task_name}, "
                    f"project with ID {project_id}. We should have "
                    "only one Job per Task and unique Task names."
                )

            return data.results[0]

    def get_job_media_metadata(self, job_id: int) -> DataMetaRead:
        with ApiClient(self.config) as api_client:
            data: DataMetaRead
            (data, response) = api_client.jobs_api.retrieve_data_meta(job_id)

            return data

    def get_job_annotation_metadata(self, job_id: int) -> DataMetaRead:
        with ApiClient(self.config) as api_client:
            data: AnnotationsRead
            (data, response) = api_client.jobs_api.retrieve_annotations(job_id)

            return data

    def get_raw_job_annotations_with_format(
        self,
        job_id: int,
        export_format: Optional[str] = None,
        max_retries: int = 20,
        interval: float = 0.1,
    ) -> bytes:
        response: HTTPResponse
        with ApiClient(self.config) as api_client:
            for _ in range(max_retries):
                (_, response) = api_client.jobs_api.retrieve_annotations(
                    job_id, format=export_format, _parse_response=False
                )

                if response.status == HTTPStatus.CREATED:
                    break
                assert response.status == HTTPStatus.ACCEPTED
                sleep(interval)

        assert response.status == HTTPStatus.CREATED
        (_, response) = api_client.jobs_api.retrieve_annotations(
            job_id,
            format=export_format,
            action="download",
            location="local",
            use_default_location=False,
            _parse_response=False,
        )
        assert response.status == HTTPStatus.OK

        return response.read()

    def get_project_labels(
        self, project_id: int, label_count: int, type: str
    ) -> dict[str, str]:
        with ApiClient(self.config) as api_client:
            data: PaginatedLabelList
            (data, response) = api_client.labels_api.list(
                project_id=project_id, page_size=label_count, type=type
            )

            return {label.id: label.name for label in data.results}

    def get_cvat_job(
        self,
        task_name: str,
        project_id: int,
        export_format: str,
        label_count: int,
    ) -> CVATJob:
        job = self.get_job(task_name, project_id)

        return CVATJob(
            job,
            self.get_job_media_metadata(job.id),
            self.get_job_annotation_metadata(job.id),
            self.get_raw_job_annotations_with_format(job.id, export_format),
            task_name,
            self.get_project_labels(project_id, label_count, "tag"),
        )
