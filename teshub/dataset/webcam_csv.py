from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type, cast

import dacite
import pandera as pa
import pandera.typing as pat

from teshub.dataset.csv_manager import CSVManager
from teshub.typing import DataClassT
from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream


class WebcamRecordSchema(pa.SchemaModel):
    id: pat.Index[str] = pa.Field(unique=True, coerce=True)
    status: pat.Series[str] = pa.Field(
        isin=cast(list[str], [status.value for status in WebcamStatus])
    )

    image_count: pat.Series[int] = pa.Field(ge=0, nullable=True, coerce=True)
    categories: Optional[pat.Series[str]] = pa.Field(nullable=True)

    city: Optional[pat.Series[str]] = pa.Field(nullable=True)
    region: Optional[pat.Series[str]] = pa.Field(nullable=True)
    country: Optional[pat.Series[str]] = pa.Field(nullable=True)
    continent: Optional[pat.Series[str]] = pa.Field(nullable=True)
    latitude: Optional[pat.Series[float]] = pa.Field(
        nullable=True, coerce=True
    )
    longitude: Optional[pat.Series[float]] = pa.Field(
        nullable=True, coerce=True
    )


@dataclass
class WebcamCSV(CSVManager[WebcamStream]):
    data_class: Type[DataClassT] = field(init=False, default=WebcamStream)

    df_index: Optional[str] = field(init=False, default="id")
    df_schema: Optional[Type[pa.SchemaModel]] = field(
        init=False, default=WebcamRecordSchema
    )
    df_read_converter: Optional[dict[str, Callable[[Any], Any]]] = field(
        init=False,
        default_factory=lambda: {
            "categories": lambda lst: lst.strip("[]")
            .replace("'", "")
            .split(", "),
        },
    )
    df_dtype: Optional[dict[str, Type[str | int | float]]] = field(
        init=False, default_factory=lambda: {"id": str}
    )
    df_columns: list[str] = field(
        init=False,
        default_factory=lambda: [
            "id",
            "status",
            "image_count",
            "categories",
            "city",
            "region",
            "country",
            "continent",
            "latitude",
            "longitude",
        ],
    )
    dacite_config: Optional[dacite.Config] = field(
        init=False, default=dacite.Config(cast=[Enum])
    )
