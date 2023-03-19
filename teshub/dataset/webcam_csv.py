import logging
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
from pandera.typing import Index, Series

from teshub.webcam.webcam_stream import (WebcamLocation, WebcamStatus,
                                         WebcamStream)


class WebcamRecordSchema(pa.SchemaModel):
    id: Index[str] = pa.Field(unique=True)
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
class WebcamCSV:
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

    def _from_record_df(self, record_df: pd.DataFrame) -> List[WebcamStream]:
        WebcamRecordSchema.validate(record_df)

        return [
            WebcamStream(
                str(id),
                cast(str, record["categories"]).split(","),
                WebcamLocation(
                    cast(str, record["city"]),
                    cast(str, record["region"]),
                    cast(str, record["country"]),
                    cast(str, record["continent"]),
                    cast(float, record["latitude"]),
                    cast(float, record["longitude"]),
                ),
                WebcamStatus(cast(str, record["status"])),
            )
            for id, record in record_df.iterrows()
        ]

    def _to_record_df(self, webcams: List[WebcamStream]) -> pd.DataFrame:
        record_df = pd.DataFrame(
            [
                [
                    webcam.id,
                    ",".join(webcam.categories) if webcam.categories else None,
                    webcam.location.city,
                    webcam.location.region,
                    webcam.location.country,
                    webcam.location.continent,
                    float(webcam.location.latitude),
                    float(webcam.location.longitude),
                    len(webcam.image_urls),
                    webcam.status.value,
                ]
                for webcam in webcams
            ],
            columns=self.record_items,
        ).set_index("id")

        WebcamRecordSchema.validate(record_df)

        return record_df

    def load(self) -> None:
        try:
            self._webcam_df = pd.read_csv(
                self.csv_path, dtype={"id": str}
            ).set_index("id")

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

    def query_webcams(
        self, df_query: Optional[str], count: Optional[int]
    ) -> List[WebcamStream]:
        query_df = (
            self._webcam_df.query(df_query) if df_query else self._webcam_df
        )

        if count:
            query_df = query_df.head(count)

        return self._from_record_df(query_df)

    def add_record(self, webcam: WebcamStream, persist: bool = True) -> None:
        new_record = self._to_record_df([webcam])
        self._webcam_df = pd.concat([self._webcam_df, new_record])

        if persist:
            self.save()

            logging.info(
                f"Persisted webcam record {webcam.id} in "
                f"{self.csv_path} successfully."
            )

    def update_record_status(
        self, webcam_id: str, status: WebcamStatus, persist: bool = True
    ) -> None:
        webcam_record = self._webcam_df.loc[[webcam_id]]
        old_status = cast(str, webcam_record[["status"]].values[0][0])
        webcam_record["status"] = status.value

        WebcamRecordSchema.validate(webcam_record)
        self._webcam_df.loc[[webcam_id]] = webcam_record

        if persist:
            self.save()

            logging.info(
                f"Changed status of webcam {webcam_id} "
                f"from {old_status} to {status.value}. Persisted changes in "
                f"{self.csv_path} successfully."
            )

    def exists(self, webcam: WebcamStream) -> bool:
        return webcam.id in cast(
            npt.NDArray[np.str_], self._webcam_df.index.values
        )
