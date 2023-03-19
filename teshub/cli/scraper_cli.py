import argparse
import logging
import os
from typing import cast

from teshub.dataset.webcam_csv import WebcamCSV
from teshub.scraping.webcam_scraper import WebcamScraper
from teshub.scraping.webcam_scraper_config import WebcamScraperConfig

logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)

parser = argparse.ArgumentParser(
    prog="windy_scraper",
    description="Scrapes webcams from windy.com.",
)
parser.add_argument(
    "--api_key",
    type=str,
    required=True,
    help="Windy API key used for webcam list requests",
)
parser.add_argument(
    "--webcam_count",
    type=int,
    default=10,
    help="Amount of webcams to scrape",
)
parser.add_argument(
    "--webcam_dir",
    type=str,
    required=True,
    help="Directory where webcam data is stored",
)
parser.add_argument(
    "--csv_path",
    type=str,
    default=None,
    help=(
        "CSV file where webcam metadata is stored. "
        "If not specified, `webcam_dir/webcam_metadata.csv` is used"
    ),
)
parser.add_argument(
    "--async_download",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "If set, images in a webcam are downloaded "
        "sequentially instead of asynchronously."
    ),
)
parser.add_argument(
    "--page_offset",
    type=int,
    default=0,
    help=(
        "Offset to use for webcam page requests. Default to 0."
        "Could check logs and use a better value on new runs."
        "May automate this in the future."
    ),
)


def scraper_config_from_args(args: argparse.Namespace) -> WebcamScraperConfig:
    return WebcamScraperConfig(
        webcam_count=cast(int, args.webcam_count),
        api_key=cast(str, args.api_key),
        dst_dir=cast(str, args.webcam_dir),
        async_download=cast(bool, args.async_download),
        request_offset=cast(int, args.page_offset),
    )


def csv_path_from_args(args: argparse.Namespace) -> str:
    return (
        os.path.abspath(cast(str, args.csv_path))
        if args.csv_path
        else os.path.join(
            os.path.abspath(cast(str, args.webcam_dir)), "webcam_metadata.csv"
        )
    )


def main() -> None:
    args = parser.parse_args()
    scraper_config = scraper_config_from_args(args)

    webcam_csv = WebcamCSV(csv_path_from_args(args))
    webcam_csv.load()

    WebcamScraper(scraper_config, webcam_csv).scrape_webcams()


if __name__ == "__main__":
    main()
