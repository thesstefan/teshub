import os
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

from teshub.dataset.webcam_csv import WebcamCSV
from teshub.dataset.webcam_frame_csv import WebcamFrameCSV
from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream


@dataclass
class WebcamDataset:
    data_dir: str
    webcam_csv_path: Optional[str] = None

    _webcam_csv: WebcamCSV = field(init=False)

    DEFAULT_WEBCAM_CSV_NAME: ClassVar[str] = "webcams.csv"
    DEFAULT_FRAME_CSV_NAME: ClassVar[str] = "frames.csv"

    def __post_init__(self) -> None:
        self._webcam_csv = WebcamCSV(
            os.path.join(
                self.data_dir,
                self.webcam_csv_path or self.DEFAULT_WEBCAM_CSV_NAME,
            )
        )

    def fill_webcam_image_paths(self, webcam: WebcamStream) -> None:
        webcam_dir = os.path.join(self.data_dir, webcam.id)

        if not os.path.isdir(webcam_dir):
            raise RuntimeError(f"Webcam directory {webcam_dir} doesn't exist")

        frame_csv = WebcamFrameCSV(
            os.path.join(webcam_dir, self.DEFAULT_FRAME_CSV_NAME)
        )

        webcam.frames = frame_csv.query_records()

    def get_webcam(
        self, webcam_id: str, fill_images: bool = True
    ) -> WebcamStream:
        webcam = self._webcam_csv.get_webcam(webcam_id)

        if fill_images:
            self.fill_webcam_image_paths(webcam)

        return webcam

    def query_webcams(
        self,
        df_query: Optional[str],
        count: Optional[int],
        fill_images: bool = True,
    ) -> List[WebcamStream]:
        webcams = self._webcam_csv.query_records(df_query, count)

        if fill_images:
            for webcam in webcams:
                self.fill_webcam_image_paths(webcam)

        return webcams

    def webcam_exists(self, webcam_id: str) -> bool:
        return self._webcam_csv.exists(webcam_id)

    def get_webcams_with_status(
        self, status: WebcamStatus, count: Optional[int]
    ) -> List[WebcamStream]:
        return self.query_webcams(f"status == '{status.value}'", count)

    def add_webcam_frames(self, webcam: WebcamStream) -> None:
        webcam_dir = os.path.join(self.data_dir, webcam.id)

        if not os.path.isdir(webcam_dir):
            raise RuntimeError(f"Webcam directory {webcam_dir} doesn't exist")

        webcam_frame_csv = WebcamFrameCSV(
            os.path.join(webcam_dir, "frames.csv")
        )

        for frame in webcam.frames:
            webcam_frame_csv.add_record(frame, log=False)

        webcam_frame_csv.save()

    def add_webcam(self, webcam: WebcamStream) -> None:
        self.add_webcam_frames(webcam)
        self._webcam_csv.add_record(webcam)

    def update_webcam_status(
        self, webcam_id: str, status: WebcamStatus
    ) -> None:
        self._webcam_csv.update_record(
            webcam_id, {"status": status}, persist=True
        )
