from dataclasses import asdict
from typing import ItemsView, Optional, Type, cast

import dacite
import numpy as np
import pandas as pd
import pandera as pa

from teshub.typing import JSON, DataClassT


class CSVConverter:
    @staticmethod
    def from_df(
        df: pd.DataFrame,
        data_cls: Type[DataClassT],
        df_index: Optional[str] = None,
        df_schema: Optional[Type[pa.SchemaModel]] = None,
        dacite_config: Optional[dacite.Config] = None,
    ) -> list[DataClassT]:
        if not len(df):
            return []

        if df_schema:
            df_schema.validate(df)

        df_dict: dict[str, dict[str, JSON]]
        records: list[dict[str, JSON]]

        if df_index:
            df_dict = df.to_dict("index")
            items: ItemsView[str, dict[str, JSON]] = df_dict.items()

            records = [columns | {df_index: index} for index, columns in items]
        else:
            records = df.to_dict("records")

        # Remove None/NaN columns so that dacite can properly
        # fill Optional values
        records = [
            {
                name: value
                for name, value in record.items()
                if value not in [None, np.nan]
            }
            for record in records
        ]

        return [
            dacite.from_dict(
                data_class=data_cls,
                data=record,
                config=dacite_config,
            )
            for record in records
        ]

    @staticmethod
    def to_df(
        data: list[DataClassT],
        df_columns: list[str],
        df_index: Optional[str] = None,
        df_schema: Optional[Type[pa.SchemaModel]] = None,
    ) -> pd.DataFrame:
        records: list[dict[str, JSON]] = [
            cast(
                dict[str, JSON],
                asdict(item)
                | {
                    # Also get property values
                    name: prop.__get__(item)
                    for name, prop in vars(type(item)).items()
                    if isinstance(prop, property)
                },
            )
            for item in data
        ]

        # Filter df to contain only specified column names
        records = [
            {name: record[name] for name in df_columns} for record in records
        ]

        df = pd.DataFrame(records)

        if df_index:
            df.set_index(df_index, inplace=True)

        if df_schema:
            df_schema.validate(df)

        return df
