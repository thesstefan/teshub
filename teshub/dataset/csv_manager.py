import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, Type

import pandas as pd
import pandera as pa

from teshub.dataset.csv_converter import CSVConverter
from teshub.typing import DataClassT


@dataclass
class CSVManager(Generic[DataClassT]):
    csv_path: str
    data_class: Type[DataClassT]

    df_index: Optional[str] = None
    df_schema: Optional[Type[pa.SchemaModel]] = None
    df_read_converter: Optional[dict[str, Callable[[Any], Any]]] = None
    df_dtype: Optional[dict[str, Type[str | int | float]]] = None

    df_columns: list[str] = field(default_factory=list)
    _df: pd.DataFrame = field(init=False)

    def load(self) -> None:
        try:
            self._df = pd.read_csv(
                self.csv_path,
                dtype=self.df_dtype,
                converters=self.df_read_converter,
            )

            if self.df_index:
                self._df.set_index(self.df_index, inplace=True)

            if not self._df.empty and self.df_schema:
                self.df_schema.validate(self._df)

            logging.info(
                f"Succesfully read {len(self._df)} "
                f"{self.data_class.__name__} record(s) from {self.csv_path}!"
            )
        except FileNotFoundError:
            logging.warning(
                "CSV file not found. "
                f"Will create new one at {self.csv_path} and continue..."
            )

            self._df = pd.DataFrame(columns=self.df_columns)

            if self.df_index:
                self._df.set_index(self.df_index, inplace=True)

    def save(self) -> None:
        csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)

        self._df.to_csv(self.csv_path, index=self.df_index is not None)

    def exists(self, id: str) -> bool:
        return self.df_index and id in self._df.index

    def query_records(
        self, df_query: Optional[str] = None, count: Optional[int] = None
    ) -> list[DataClassT]:
        query_df = self._df.query(df_query) if df_query else self._df

        if count:
            query_df = query_df.head(count)

        return CSVConverter.from_df(
            query_df,
            self.data_class,
            self.df_index,
            self.df_schema,
            self.dacite_config,
        )

    def add_record(self, item: DataClassT, persist: bool = True) -> None:
        item_df = CSVConverter.to_df(
            [item], self.df_columns, self.df_index, self.df_schema
        )

        assert len(item_df) == 1

        if self.df_index and item_df.index.values[0] in self._df.index:
            raise RuntimeError(
                f"{self.data_class.__name__}({item_df.index.values[0]}) "
                "already exists in CSV file!"
            )

        self._df = pd.concat([self._df, item_df])

        if persist:
            self.save()

            logging.info(f"Persisted {item} in {self.csv_path} successfully.")

    def update_record(
        self,
        id: str,
        # TODO: Use some generic type (numpy dtype?)
        update_dict: dict[str, str | int | float | bool],
        persist: bool = True,
    ) -> None:
        if id in update_dict.keys():
            raise RuntimeError("Can't update record ID!")

        record = self._df.loc[[id]].copy()
        record[list(update_dict.keys())] = list(update_dict.values())

        self.df_schema.validate(record)

        self._df.loc[[id]] = record

        if persist:
            self.save()

            logging.info(
                f"Updated {self.data_class.__name__}({id}) "
                f"with new values in {self.csv_path}:"
            )
            for name, value in update_dict.items():
                logging.info(f"\t\t{name} -> {value}")
