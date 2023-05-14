import argparse
import io
import logging
import os
from dataclasses import dataclass

from teshub.dataset.webcam_dataset import WebcamDataset
from teshub.recognition.predictor import WeatherInFormerPredictor
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus
from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream

parser = argparse.ArgumentParser(
    prog="teshub_auto_annotation",
    description=(
        "Batch annotates webcam streams in dataset"
        "using pretrained WeatherInformer model."
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
parser.add_argument(
    "--model_checkpoint_path",
    type=str,
    help="Path to WeatherInfo model checkpoint to be used",
    required=True,
)


@dataclass(kw_only=True)
class Arguments:
    model_checkpoint_path: str

    dataset_dir: str = '.'
    csv_path: str | None = None


def csv_path_from_args(args: Arguments) -> str | None:
    return os.path.abspath(args.csv_path) if args.csv_path else None


def load_predictor(model_checkpoint_path: str) -> WeatherInFormerPredictor:
    return WeatherInFormerPredictor(model_checkpoint_path)


def annotate_webcam_dataset(
    webcam_dataset: WebcamDataset,
    weather_informer_predictor: WeatherInFormerPredictor
) -> None:
    webcams: list[WebcamStream] = webcam_dataset.query_webcams(
        f"status != '{WebcamStatus.ANNOTATED}'", count=None
    )

    logging.info(f"Automatically annotating {len(webcams)} webcams...")

    for webcam in webcams:
        def not_annotated_frame(frame: WebcamFrame) -> bool:
            return frame.status not in [
                WebcamFrameStatus.MANUALLY_ANNOTATED,
                WebcamFrameStatus.AUTOMATICALLY_ANNOTATED
            ]

        frames = list(filter(not_annotated_frame, webcam.frames))
        output_gen = weather_informer_predictor.predict_and_process(
            [frame.image for frame in frames]
        )

        webcam_raw_seg_masks: dict[str, bytes] = {}
        webcam_labels: dict[str, dict[str, float]] = {}

        for idx, (frame, output) in enumerate(zip(frames, output_gen)):
            seg_image, labels, _ = output

            # TODO: Needed for compatibility with WebcamDataset.
            seg_file_name = frame.file_name.replace(".jpg", "_seg.png")

            # TODO: Needed for compatibility with WebcamDataset.
            seg_png_bytes = io.BytesIO()
            seg_image.save(seg_png_bytes, format='PNG')
            webcam_raw_seg_masks[seg_file_name] = seg_png_bytes.getvalue()

            webcam_labels[frame.file_name] = labels

        webcam_dataset.save_annotations(
            webcam.id, webcam_raw_seg_masks, webcam_labels, synthetic=True
        )
        webcam_dataset.update_webcam_status(webcam.id, WebcamStatus.ANNOTATED)


def main() -> None:
    args = Arguments(**dict(vars(parser.parse_args())))  # type: ignore

    webcam_dataset = WebcamDataset(
        os.path.abspath(args.dataset_dir), csv_path_from_args(args)
    )

    weather_informer_predictor = load_predictor(args.model_checkpoint_path)
    annotate_webcam_dataset(webcam_dataset, weather_informer_predictor)
