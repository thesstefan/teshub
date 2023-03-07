import logging
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
from pandera.typing import Index, Series

from teshub.webcam.webcam_stream import WebcamStatus, WebcamStream


class WebcamRecordSchema(pa.SchemaModel):
    id: Index[int] = pa.Field(unique=True)
    categories: Optional[Series[str]] = pa.Field(nullable=True)
    city: Optional[Series[str]] = pa.Field(nullable=True)
    region: Optional[Series[str]] = pa.Field(nullable=True)
    country: Optional[Series[str]] = pa.Field(nullable=True)
    continent: Optional[Series[str]] = pa.Field(nullable=True)
    latitude: Optional[Series[float]] = pa.Field(nullable=True)
    longitude: Optional[Series[float]] = pa.Field(nullable=True)
    image_count: Series[int] = pa.Field(gt=0)
    status: Series[str] = pa.Field(
        isin=cast(List[str], [status.value for status in WebcamStatus])
    )


@dataclass
class WebcamRecordKeeper:
    csv_path: str
    _webcam_df: pd.DataFrame = field(init=False)

    record_items: ClassVar[List[str]] = [
        "id",
        "categories",
        "city",
        "region",
        "country",
        "continent",
        "latitude",
        "longitude",
        "image_count",
        "status",
    ]

    def load(self) -> None:
        try:
            self._webcam_df = pd.read_csv(self.csv_path).set_index("id")

            if not self._webcam_df.empty:
                WebcamRecordSchema.validate(self._webcam_df)

            logging.info(
                f"Succesfully read webcam records from {self.csv_path}"
            )
        except FileNotFoundError:
            logging.warning(
                "Record file not found. "
                f"Creating new one at {self.csv_path} and continuing..."
            )

            self._webcam_df = pd.DataFrame(
                columns=self.record_items
            ).set_index("id")

    def save(self) -> None:
        self._webcam_df.to_csv(self.csv_path, index=True)

    def exists(self, webcam: WebcamStream) -> bool:
        return int(webcam.id) in cast(
            npt.NDArray[np.int64], self._webcam_df.index.values
        )

    def add_record(self, webcam: WebcamStream, persist: bool = True) -> None:
        new_record = pd.DataFrame(
            [
                [
                    webcam.id,
                    ",".join(webcam.categories),
                    webcam.location.city,
                    webcam.location.region,
                    webcam.location.country,
                    webcam.location.continent,
                    webcam.location.latitude,
                    webcam.location.longitude,
                    len(webcam.image_urls),
                    webcam.status.value,
                ]
            ],
            columns=self.record_items,
        ).set_index("id")

        WebcamRecordSchema.validate(new_record)
        self._webcam_df = pd.concat([self._webcam_df, new_record])

        if persist:
            self.save()
