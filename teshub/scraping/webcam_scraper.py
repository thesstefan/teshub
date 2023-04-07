import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type, Union, cast

import requests
from bs4 import BeautifulSoup

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.scraping.webcam_downloader import (AsyncWebcamDownloader,
                                               SequentialWebcamDownloader,
                                               WebcamDownloader)
from teshub.scraping.webcam_scraper_config import WebcamScraperConfig
from teshub.typing import JSON
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus
from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream

WINDY_API_URL = "https://api.windy.com/api/webcams/v2/list"
EMBED_WINDY_WEBCAM_URL = (
    "https://webcams.windy.com/webcams/public/embed/player"
)


@dataclass
class WebcamScraper:
    config: WebcamScraperConfig
    webcam_dataset: WebcamDataset
    webcam_downloader: Type[WebcamDownloader] = field(
        init=False, default=AsyncWebcamDownloader
    )
    selected_webcams: List[WebcamStream] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        if not self.config.async_download:
            self.webcam_downloader = SequentialWebcamDownloader

    def _build_api_request_url(self, offset: Optional[int]) -> str:
        WINDY_API_URL = "https://api.windy.com/api/webcams/v2/list"
        show_params = "?show=webcams:location,category"

        offset = offset or self.config.request_offset

        api_request_url = (
            f"{WINDY_API_URL}"
            f"/limit={self.config.request_limit},{offset}/"
            f"{show_params}"
        )

        return api_request_url

    def _request_webcam_list(self, request_url: str) -> List[WebcamStream]:
        logging.info("Requesting webcam list with given configuration...")
        logging.info(f"\tGET {request_url}")

        headers = {"x-windy-key": self.config.api_key}

        response = requests.get(request_url, headers=headers)
        response.raise_for_status()

        response_json: Dict[str, Dict[str, JSON]] = json.loads(response.text)
        webcam_json_list = cast(
            List[Dict[str, JSON]], response_json["result"]["webcams"]
        )

        webcam_list = []
        for webcam_json in webcam_json_list:
            location_json = cast(
                Dict[str, Union[str, float]], webcam_json["location"]
            )

            webcam_list.append(
                WebcamStream(
                    cast(str, webcam_json["id"]),
                    WebcamStatus.NONE,
                    None,
                    [
                        category_json["id"]
                        for category_json in cast(
                            List[Dict[str, str]], webcam_json["category"]
                        )
                    ],
                    cast(str, location_json["city"]),
                    cast(str, location_json["region"]),
                    cast(str, location_json["country"]),
                    cast(str, location_json["continent"]),
                    cast(float, location_json["latitude"]),
                    cast(float, location_json["longitude"]),
                )
            )

        return webcam_list

    def _get_webcam_image_urls(self, webcam_id: str) -> List[str]:
        logging.info(f"Requesting webcam {webcam_id} metadata...")
        embed_webcam_link = (
            f"{EMBED_WINDY_WEBCAM_URL}/{webcam_id}"
            f"/{self.config.webcam_timeframe}"
        )

        request = requests.get(embed_webcam_link)
        request.raise_for_status()

        soup = BeautifulSoup(request.text, "lxml")
        pattern = re.compile("(_slideFull = )(.+)((?:\n.+)+)(])")
        script = soup.find("script", string=pattern)

        if not script:
            raise RuntimeError("Failed to extract webcam image list!")

        result = pattern.search(script.text)

        if not result:
            raise RuntimeError("Failed to extract webcam image list!")

        image_list_str = (
            cast(str, result.group(2)).removesuffix(",").replace("'", '"')
        )

        return cast(List[str], json.loads(image_list_str))

    def _select_webcams(self) -> None:
        offset = self.config.request_offset
        request_index = 0

        while len(self.selected_webcams) < self.config.webcam_count:
            request_url = self._build_api_request_url(offset)

            new_webcams = list(
                filter(
                    cast(
                        Callable[[WebcamStream], bool],
                        lambda webcam: not self.webcam_dataset.webcam_exists(
                            webcam.id
                        ),
                    ),
                    self._request_webcam_list(request_url),
                )
            )

            offset += self.config.request_limit
            self.selected_webcams.extend(new_webcams)

            request_index += 1

        self.selected_webcams = self.selected_webcams[
            : self.config.webcam_count
        ]

        logging.info(
            f"Successfully selected {len(self.selected_webcams)} "
            f"webcams after {request_index} requests\n"
        )

    def scrape_webcams(self) -> None:
        self._select_webcams()

        if not os.path.exists(self.config.dst_dir):
            os.makedirs(self.config.dst_dir)

        for index, webcam in enumerate(self.selected_webcams):
            webcam_dir = os.path.join(self.config.dst_dir, str(webcam.id))

            image_urls = self._get_webcam_image_urls(webcam.id)

            webcam.frames = [
                WebcamFrame(
                    os.path.basename(url),
                    # Nothing is persisted if download is not succesful, so
                    # it's safe to assign status beforehand
                    status=WebcamFrameStatus.DOWNLOADED,
                    url=url,
                )
                for url in image_urls
            ]
            webcam.image_count = len(image_urls)

            download_succesful = self.webcam_downloader(
                webcam, webcam_dir, self.config
            ).download()

            if not download_succesful:
                continue

            webcam.status = WebcamStatus.DOWNLOADED
            self.webcam_dataset.add_webcam(webcam)

            logging.info(
                f"Scraped {index + 1}/{self.config.webcam_count} streams\n"
            )
