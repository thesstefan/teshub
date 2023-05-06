import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type, cast

import dacite
import pandera as pa
import pandera.typing as pat

from teshub.dataset.csv_manager import CSVManager
from teshub.extra_typing import DataClassT
from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream


class WebcamStreamRecordSchema(pa.SchemaModel):
    id: pat.Index[str] = pa.Field(unique=True, coerce=True)
    status: pat.Series[str] = pa.Field(
        isin=cast(list[str], [status.value for status in WebcamStatus])
    )

    image_count: pat.Series[int] = pa.Field(ge=0, nullable=True, coerce=True)
    # TODO: Check if labels string can be evaluated by
    # ast.literal_eval as a list[str]
    categories: pat.Series[str] | None = pa.Field(nullable=True, coerce=True)

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
        init=False, default=WebcamStreamRecordSchema
    )
    df_read_converter: Optional[dict[str, Callable[[Any], Any]]] = field(
        init=False,
        default_factory=lambda: {
            "categories": lambda list_str: ast.literal_eval(list_str)
            if list_str
            else None
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
        init=False, default_factory=lambda: dacite.Config(cast=[Enum])
    )
