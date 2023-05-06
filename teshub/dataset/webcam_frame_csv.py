import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type, cast

import dacite
import pandera as pa
import pandera.typing as pat

from teshub.dataset.csv_manager import CSVManager
from teshub.extra_typing import DataClassT
from teshub.webcam.webcam_frame import WebcamFrame, WebcamFrameStatus


class WebcamFrameRecordSchema(pa.SchemaModel):
    file_name: pat.Index[str] = pa.Field(unique=True, coerce=True)
    status: pat.Series[str] = pa.Field(
        isin=cast(list[str], [status.value for status in WebcamFrameStatus])
    )
    segmentation_path: pat.Series[str] = pa.Field(nullable=True, coerce=True)
    labels: pat.Series[str] = pa.Field(nullable=True, coerce=True)


@dataclass
class WebcamFrameCSV(CSVManager[WebcamFrame]):
    data_class: Type[DataClassT] = field(init=False, default=WebcamFrame)

    df_index: Optional[str] = field(init=False, default="file_name")
    df_schema: Optional[Type[pa.SchemaModel]] = field(
        init=False, default=WebcamFrameRecordSchema
    )
    df_read_converter: Optional[dict[str, Callable[[Any], Any]]] = field(
        init=False,
        default_factory=lambda: {
            "labels": lambda label_dict: ast.literal_eval(label_dict)
            if label_dict
            else None
        },
    )
    df_columns: list[str] = field(
        init=False,
        default_factory=lambda: [
            "file_name",
            "status",
            "segmentation_path",
            "labels",
        ],
    )
    dacite_config: Optional[dacite.Config] = field(
        init=False, default_factory=lambda: dacite.Config(cast=[Enum])
    )
