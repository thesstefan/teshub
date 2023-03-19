import argparse
import logging
import os
from typing import cast

from cvat_sdk.api_client import Configuration as CVATConfiguration

from teshub.cvat.cvat_task_creator import CVATTaskCreator
from teshub.cvat.webcam_cvat_task_creator import WebcamAdnotationTaskCreator
from teshub.dataset.webcam_csv import WebcamCSV
from teshub.dataset.webcam_dataset import WebcamDataset

logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cvat_task_creator.log"),
        logging.StreamHandler(),
    ],
)

parser = argparse.ArgumentParser(
    prog="teshub_cvat",
    description="Provides tooling for CVAT interaction",
)
subparsers = parser.add_subparsers(
    required=True,
    title="subcommands",
    description="Valid subcommands",
    help="Additional help",
)

task_creator_parser = subparsers.add_parser("task_creator")

task_creator_parser.add_argument(
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
task_creator_parser.add_argument(
    "--cvat_host",
    type=str,
    default="localhost:8080",
    help="CVAT host endpoint",
)
task_creator_parser.add_argument(
    "--csv_path",
    type=str,
    required=True,
    help=(
        "CSV file where webcam metadata is stored. "
        "If not specified, `dataset_dir/webcam_metadata.csv` is used"
    ),
)
task_creator_parser.add_argument(
    "--task_count",
    required=True,
    type=int,
    help=("Upper bound of number of tasks to create."),
)
task_creator_parser.add_argument(
    "--cvat_username",
    required=True,
    type=str,
    help=("CVAT username"),
)
task_creator_parser.add_argument(
    "--cvat_password",
    required=True,
    type=str,
    help=("CVAT password"),
)
task_creator_parser.add_argument(
    "--owner_id",
    required=True,
    type=int,
    help=("The ID of the task owner"),
)
task_creator_parser.add_argument(
    "--project_id",
    required=True,
    type=int,
    help=("The ID of the project to create the tasks in"),
)
task_creator_parser.add_argument(
    "--assignee_id",
    required=True,
    type=int,
    help=("The ID of the user to assign the tasks to"),
)


def csv_path_from_args(args: argparse.Namespace) -> str:
    return (
        os.path.abspath(cast(str, args.csv_path))
        if args.csv_path
        else os.path.join(
            os.path.abspath(cast(str, args.webcam_dir)), "webcam_metadata.csv"
        )
    )


def create_tasks(args: argparse.Namespace) -> None:
    cvat_config = CVATConfiguration(
        host=cast(str, args.cvat_host),
        username=cast(str, args.cvat_username),
        password=cast(str, args.cvat_password),
    )
    cvat_task_creator = CVATTaskCreator(
        cvat_config, use_shared_storage=(cast(str, args.dataset_dir) == ".")
    )

    webcam_csv = WebcamCSV(csv_path_from_args(args))
    webcam_csv.load()
    dataset = WebcamDataset(webcam_csv, cast(str, args.dataset_dir))

    task_creator = WebcamAdnotationTaskCreator(cvat_task_creator, dataset)

    task_creator.create_tasks(
        cast(int, args.task_count),
        cast(int, args.project_id),
        cast(int, args.owner_id),
        cast(int, args.assignee_id),
    )


def main() -> None:
    task_creator_parser.set_defaults(func=create_tasks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
