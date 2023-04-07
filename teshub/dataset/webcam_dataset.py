import glob
import os
from dataclasses import dataclass
from typing import List, Optional

from teshub.dataset.webcam_csv import WebcamCSV
from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream


@dataclass
class WebcamDataset:
    webcam_csv: WebcamCSV
    data_dir: str

    def fill_webcam_image_paths(self, webcam: WebcamStream) -> None:
        webcam_dir = os.path.join(self.data_dir, str(webcam.id))

        if not os.path.isdir(webcam_dir):
            raise RuntimeError(f"Webcam directory {webcam_dir} doesn't exist")

        webcam.image_paths = glob.glob(f"{webcam_dir}/*.jpg")

    def get_webcam(
        self, webcam_id: str, fill_images: bool = True
    ) -> WebcamStream:
        webcam = self.webcam_csv.get_webcam(webcam_id)

        if fill_images:
            self.fill_webcam_image_paths(webcam)

        return webcam

    def query_webcams(
        self,
        df_query: Optional[str],
        count: Optional[int],
        fill_images: bool = True,
    ) -> List[WebcamStream]:
        webcams = self.webcam_csv.query_records(df_query, count)

        if fill_images:
            for webcam in webcams:
                self.fill_webcam_image_paths(webcam)

        return webcams

    def get_webcams_with_status(
        self, status: WebcamStatus, count: Optional[int]
    ) -> List[WebcamStream]:
        return self.query_webcams(f"status == '{status.value}'", count)
