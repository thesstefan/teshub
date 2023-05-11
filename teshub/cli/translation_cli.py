import argparse
import io
import logging
import os
from dataclasses import dataclass

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.translation.weather2weather import Weather2WeatherDataset
from teshub.webcam.webcam_frame import WebcamFrame

parser = argparse.ArgumentParser(
    prog="teshub_translation",
    description=(
        "Does weather translation?"
    )
)
parser.add_argument(
    "--csv_path",
    type=str,
    help=(
        "CSV file where webcam metadata is stored. "
        "If not specified, `dataset_dir/webcams.csv` is used"
    ),
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=".",
    help=(
        "Directory where webcam streams and metadata are stored"
    ),
)


@dataclass(kw_only=True)
class Arguments:
    dataset_dir: str = '.'
    csv_path: str | None = None


def csv_path_from_args(args: Arguments) -> str | None:
    return os.path.abspath(args.csv_path) if args.csv_path else None


def main() -> None:
    args = Arguments(**dict(vars(parser.parse_args())))  # type: ignore

    webcam_dataset = WebcamDataset(
        os.path.abspath(args.dataset_dir), csv_path_from_args(args)
    )

    def is_cloudy(frame: WebcamFrame) -> bool:
        assert frame.labels
        return frame.labels['cloudy'] >= 0.4

    def is_clear(frame: WebcamFrame) -> bool:
        assert frame.labels
        return frame.labels['cloudy'] < 0.4

    weather2weather = Weather2WeatherDataset(
        webcam_dataset,
        select_source=is_clear,
        select_target=is_cloudy
    )
    print(weather2weather[0][0].shape)


if __name__ == '__main__':
    main()
