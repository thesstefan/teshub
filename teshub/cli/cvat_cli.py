import argparse
import os
from typing import cast

from cvat_sdk.api_client import Configuration as CVATConfiguration

from teshub.annotation.cvat.cvat_manager import CVATManager
from teshub.annotation.webcam_annotation_manager import WebcamAnnotationManager
from teshub.dataset.webcam_dataset import WebcamDataset

parser = argparse.ArgumentParser(
    prog="teshub_cvat",
    description="Provides tooling for CVAT interaction",
)

parser.add_argument(
    "--cvat_host",
    type=str,
    default="localhost:8080",
    help="CVAT host endpoint",
)
parser.add_argument(
    "--cvat_username",
    required=True,
    type=str,
    help=("CVAT username"),
)
parser.add_argument(
    "--cvat_password",
    required=True,
    type=str,
    help=("CVAT password"),
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=".",
    help=(
        "Directory where webcam streams are stored. "
        "If specified, local CVAT storage will be used. "
        "Otherwise, will attempt to use shared CVAT storage with "
        "image paths from the current directory"
    ),
)
parser.add_argument(
    "--project_id",
    required=True,
    type=int,
    help=("The ID of the project to create the jobs in"),
)
parser.add_argument(
    "--csv_path",
    type=str,
    help=(
        "CSV file where webcam metadata is stored. "
        "If not specified, `dataset_dir/webcams.csv` is used"
    ),
)


subparsers = parser.add_subparsers(
    required=True,
    title="subcommands",
    description="Valid subcommands",
    help="Additional help",
)

job_create_parser = subparsers.add_parser("job_create")
job_sync_parser = subparsers.add_parser("job_sync")

job_create_parser.add_argument(
    "--job_count",
    required=True,
    type=int,
    help=("Upper bound of number of jobs to create."),
)
job_create_parser.add_argument(
    "--owner_id",
    required=True,
    type=int,
    help=("The ID of the job owner"),
)
job_create_parser.add_argument(
    "--assignee_id",
    required=True,
    type=int,
    help=("The ID of the user to assign the jobs to"),
)


def csv_path_from_args(args: argparse.Namespace) -> str:
    return os.path.abspath(cast(str, args.csv_path)) if args.csv_path else None


def sync_jobs(
    annotation_manager: WebcamAnnotationManager, args: argparse.Namespace
) -> None:
    annotation_manager.sync(cast(int, args.project_id))


def create_jobs(
    annotation_manager: WebcamAnnotationManager, args: argparse.Namespace
) -> None:
    annotation_manager.create_tasks(
        cast(int, args.job_count),
        cast(int, args.project_id),
        cast(int, args.owner_id),
        cast(int, args.assignee_id),
    )


def main() -> None:
    job_create_parser.set_defaults(func=create_jobs)
    job_sync_parser.set_defaults(func=sync_jobs)

    args = parser.parse_args()

    cvat_config = CVATConfiguration(
        host=cast(str, args.cvat_host),
        username=cast(str, args.cvat_username),
        password=cast(str, args.cvat_password),
    )

    webcam_dataset = WebcamDataset(
        cast(str, os.path.abspath(args.dataset_dir)), csv_path_from_args(args)
    )

    cvat_manager = CVATManager(
        cvat_config, use_shared_storage=(cast(str, args.dataset_dir) == ".")
    )

    annotation_manager = WebcamAnnotationManager(cvat_manager, webcam_dataset)

    args.func(annotation_manager, args)


if __name__ == "__main__":
    main()
