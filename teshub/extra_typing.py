from typing import Mapping, Sequence, TypeAlias, TypeVar

JSON: TypeAlias = (
    Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
)

DataClassT = TypeVar("DataClassT")
