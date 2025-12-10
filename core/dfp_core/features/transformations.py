"""Shared pandas/pyarrow/polars transforms for feature engineering."""
from typing import Any, Callable

Transform = Callable[[Any], Any]

def normalize_numeric(df: Any, columns: list[str]) -> Any:
    for col in columns:
        if col in df:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df
