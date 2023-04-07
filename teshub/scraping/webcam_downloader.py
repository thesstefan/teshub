import abc
import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Awaitable, List, Optional, cast

import aiofiles
import requests
from aiohttp.client import ClientSession

from teshub.scraping.webcam_scraper_config import WebcamScraperConfig
from teshub.webcam.webcam_stream import WebcamStream


@dataclass
class WebcamDownloader(abc.ABC):
    webcam: WebcamStream
    dst_dir: str
    config: Optional[WebcamScraperConfig] = None

    @abc.abstractmethod
    def _download_webcam_images(self) -> None:
        raise NotImplementedError

    def download(self) -> bool:
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

        logging.info(f"Downloading images from webcam {self.webcam.id}...")

        try:
            self._download_webcam_images()
        except Exception as error:
            logging.error(f"Error on webcam download: {error}")
            shutil.rmtree(self.dst_dir)
            logging.error(f"Removed webcam dir {self.dst_dir}. Continuing...")

            return False

        logging.info(
            f"Images from webcam {self.webcam.id} downloaded successfully."
        )

        return True


@dataclass
class AsyncWebcamDownloader(WebcamDownloader):
    webcam: WebcamStream
    dst_dir: str
    config: Optional[WebcamScraperConfig] = None

    default_max_concurrent_requests = 5
    default_request_timeout = 30

    async def _download_image(
        self,
        image_url: str,
        image_dst_path: str,
        http_session: ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            logging.info(f"\t Downloading {image_dst_path}...")

            async with http_session.get(image_url) as response:
                content = await response.read()
                response.raise_for_status()

                async with aiofiles.open(image_dst_path, "+wb") as image:
                    await image.write(content)

    async def _download_webcam_images_async(self) -> None:
        download_tasks: List[Awaitable[None]] = []
        semaphore = asyncio.Semaphore(
            self.config.download_max_concurrent_requests
            if self.config
            else self.default_max_concurrent_requests
        )

        async with ClientSession() as http_session:
            for frame in self.webcam.frames:
                if not frame.url:
                    raise RuntimeError(
                        f"Frame {frame} from {self.webcam} does "
                        "not have an associated URL!"
                    )

                image_dst_path = os.path.join(
                    self.dst_dir, os.path.basename(frame.url)
                )

                download_tasks.append(
                    asyncio.wait_for(
                        self._download_image(
                            frame.url,
                            image_dst_path,
                            http_session,
                            semaphore,
                        ),
                        timeout=(
                            self.config.download_request_timeout
                            if self.config
                            else self.default_request_timeout
                        ),
                    )
                )

            await cast(Awaitable[List[None]], asyncio.gather(*download_tasks))

    def _download_webcam_images(self) -> None:
        asyncio.run(self._download_webcam_images_async())


@dataclass
class SequentialWebcamDownloader(WebcamDownloader):
    webcam: WebcamStream
    dst_dir: str
    config: Optional[WebcamScraperConfig] = None

    def _download_image(self, image_url: str, image_dst_path: str) -> None:
        logging.info(f"\t Downloading {image_dst_path}...")

        response = requests.get(image_url)
        response.raise_for_status()

        with open(image_dst_path, "wb") as image_file:
            image_file.write(response.content)

    def _download_webcam_images(self) -> None:
        for frame in self.webcam.frames:
            if not frame.url:
                raise RuntimeError(
                    f"Frame {frame} from {self.webcam} does "
                    "not have an associated URL!"
                )

            image_dst_path = os.path.join(
                self.dst_dir, os.path.basename(frame.url)
            )

            self._download_image(frame.url, image_dst_path)
