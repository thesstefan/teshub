import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

from teshub.dataset.webcam_csv import WebcamCSV
from teshub.dataset.webcam_frame_csv import WebcamFrameCSV
from teshub.webcam.webcam_frame import WebcamFrameStatus
from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream


@dataclass
class WebcamDataset:
    data_dir: str
    webcam_csv_path: Optional[str] = None

    _webcam_csv: WebcamCSV = field(init=False)

    DEFAULT_WEBCAM_CSV_NAME: ClassVar[str] = "webcams.csv"
    DEFAULT_FRAME_CSV_NAME: ClassVar[str] = "frames.csv"

    def __post_init__(self) -> None:
        logging.info("Loading webcam dataset...")

        self._webcam_csv = WebcamCSV(
            os.path.join(
                self.data_dir,
                self.webcam_csv_path or self.DEFAULT_WEBCAM_CSV_NAME,
            )
        )

        logging.info("Successfully loaded dataset!\n")

    def fill_webcam_frames(self, webcam: WebcamStream) -> None:
        webcam_dir = os.path.join(self.data_dir, webcam.id)

        if not os.path.isdir(webcam_dir):
            raise RuntimeError(f"Webcam directory {webcam_dir} doesn't exist")

        frame_csv = WebcamFrameCSV(
            os.path.join(webcam_dir, self.DEFAULT_FRAME_CSV_NAME)
        )

        webcam.frames = frame_csv.query_records()

        for frame in webcam.frames:
            frame.webcam_id = webcam.id
            frame.load_dir = webcam_dir

    def get_webcam(
        self, webcam_id: str, fill_frames: bool = True
    ) -> WebcamStream:
        webcam = self._webcam_csv.get_webcam(webcam_id)

        if fill_frames:
            self.fill_webcam_frames(webcam)

        return webcam

    def query_webcams(
        self,
        df_query: Optional[str],
        count: Optional[int],
        fill_frames: bool = True,
    ) -> List[WebcamStream]:
        webcams = self._webcam_csv.query_records(df_query, count)

        if fill_frames:
            for webcam in webcams:
                self.fill_webcam_frames(webcam)

        return webcams

    def webcam_exists(self, webcam_id: str) -> bool:
        return self._webcam_csv.exists(webcam_id)

    def get_webcams_with_status(
        self, status: WebcamStatus, count: Optional[int] = None
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

    def delete_webcam(self, webcam_id: str, soft_rm: bool = True) -> None:
        webcam_dir = os.path.join(self.data_dir, webcam_id)

        self.update_webcam_status(webcam_id, WebcamStatus.DELETED)

        if soft_rm:
            shutil.move(webcam_dir, f"{webcam_id}_DELETED")

            return

        shutil.rmtree(webcam_dir)

    def delete_frames(self, webcam_id: str, deleted_frames: list[str]) -> None:
        webcam_dir = os.path.join(self.data_dir, webcam_id)

        frame_csv = WebcamFrameCSV(
            os.path.join(webcam_dir, self.DEFAULT_FRAME_CSV_NAME)
        )

        deleted_count: int = 0
        for frame_file_name in deleted_frames:
            frame_file_name = os.path.basename(frame_file_name)
            frame = frame_csv.get(frame_file_name)

            if frame.status != WebcamFrameStatus.DELETED:
                frame_csv.update_record(
                    frame_file_name,
                    {"status": WebcamFrameStatus.DELETED},
                    persist=False,
                )

                deleted_count += 1

        if deleted_count:
            frame_csv.save()

            webcam = self._webcam_csv.get(webcam_id)
            self._webcam_csv.update_record(
                webcam_id,
                {"image_count": webcam.image_count - deleted_count},
            )

    def _save_seg_masks(
        self, webcam_id: str, seg_masks: dict[str, bytes]
    ) -> None:
        webcam_dir = os.path.join(self.data_dir, webcam_id)

        for seg_file_name, seg_bytes in seg_masks.items():
            seg_path = os.path.join(webcam_dir, seg_file_name)

            with open(seg_path, "wb") as seg_file:
                seg_file.write(seg_bytes)

    def save_annotations(
        self,
        webcam_id: str,
        seg_masks: dict[str, bytes],
        labels: dict[dict[str, float]],
        synthetic: bool,
    ) -> None:
        webcam_dir = os.path.join(self.data_dir, webcam_id)
        frame_csv = WebcamFrameCSV(
            os.path.join(webcam_dir, self.DEFAULT_FRAME_CSV_NAME)
        )

        self._save_seg_masks(webcam_id, seg_masks)

        for seg_file_name in seg_masks.keys():
            seg_file_name = os.path.basename(seg_file_name)

            frame_csv.update_record(
                seg_file_name.replace("_seg.png", ".jpg"),
                {
                    "status": WebcamFrameStatus.SYNTHETIC_ANNOTATION
                    if synthetic
                    else WebcamFrameStatus.MANUAL_ANNOTATION,
                    "segmentation_path": seg_file_name,
                },
                persist=False,
            )

        for frame_file_name, frame_labels in labels.items():
            frame_file_name = os.path.basename(frame_file_name)

            frame_csv.update_record(
                frame_file_name,
                {
                    "status": WebcamFrameStatus.SYNTHETIC_ANNOTATION
                    if synthetic
                    else WebcamFrameStatus.MANUAL_ANNOTATION,
                    "labels": str(frame_labels),
                },
                persist=False,
            )

        frame_csv.save()

        # TODO: Know when fully annotated
        self.update_webcam_status(webcam_id, WebcamStatus.PARTIALLY_ANNOTATED)

    def get_by_iloc(self, iloc: int) -> WebcamStream:
        webcam = self.webcam_csv.get_by_iloc(iloc)
        self.fill_webcam_frames(webcam)

        return webcam
