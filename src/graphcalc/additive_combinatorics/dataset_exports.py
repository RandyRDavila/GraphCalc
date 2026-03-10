from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from graphcalc.additive_combinatorics.dataset_generators import (
    additive_dataset_column_definitions,
)

__all__ = [
    "additive_rows_to_dataframe",
    "additive_package_to_dataframe",
    "save_additive_package_csv",
    "save_additive_column_definitions_json",
    "save_additive_metadata_json",
]


def additive_rows_to_dataframe(rows: Iterable[dict]) -> pd.DataFrame:
    r"""
    Convert additive-combinatorics dataset rows to a pandas DataFrame.

    Parameters
    ----------
    rows : iterable of dict
        Dataset rows, typically produced by
        :func:`graphcalc.additive_combinatorics.dataset_generators.generate_additive_set_dataset`
        or one of the snapshot helpers.

    Returns
    -------
    pandas.DataFrame
        DataFrame whose columns are determined by the row keys.

    Notes
    -----
    This function does not require any specific schema beyond standard
    dictionary-like dataset rows, but it is intended primarily for use with the
    additive-combinatorics conjecturing records defined by this package.
    """
    return pd.DataFrame(list(rows))


def additive_package_to_dataframe(package: dict) -> pd.DataFrame:
    r"""
    Convert a dataset package dictionary to a pandas DataFrame.

    Parameters
    ----------
    package : dict
        Dataset package containing a ``"rows"`` entry.

    Returns
    -------
    pandas.DataFrame
        DataFrame built from ``package["rows"]``.

    Raises
    ------
    KeyError
        If the package does not contain a ``"rows"`` entry.
    TypeError
        If ``package["rows"]`` is not iterable in the expected way.
    """
    if "rows" not in package:
        raise KeyError("package must contain a 'rows' entry.")
    rows = package["rows"]
    return additive_rows_to_dataframe(rows)


def save_additive_package_csv(package: dict, path: str | Path, *, index: bool = False) -> Path:
    r"""
    Save a dataset package as a CSV file.

    Parameters
    ----------
    package : dict
        Dataset package containing a ``"rows"`` entry.
    path : str or pathlib.Path
        Output CSV path.
    index : bool, default=False
        Whether to write the pandas index column.

    Returns
    -------
    pathlib.Path
        The resolved output path that was written.
    """
    output_path = Path(path)
    df = additive_package_to_dataframe(package)
    df.to_csv(output_path, index=index)
    return output_path


def save_additive_column_definitions_json(path: str | Path, *, indent: int = 2) -> Path:
    r"""
    Save the additive dataset column definitions as JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Output JSON path.
    indent : int, default=2
        Indentation level passed to :func:`json.dump`.

    Returns
    -------
    pathlib.Path
        The resolved output path that was written.
    """
    output_path = Path(path)
    definitions = additive_dataset_column_definitions()
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(definitions, fh, indent=indent, sort_keys=True)
    return output_path


def save_additive_metadata_json(metadata: dict, path: str | Path, *, indent: int = 2) -> Path:
    r"""
    Save dataset metadata as JSON.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to serialize.
    path : str or pathlib.Path
        Output JSON path.
    indent : int, default=2
        Indentation level passed to :func:`json.dump`.

    Returns
    -------
    pathlib.Path
        The resolved output path that was written.
    """
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=indent, sort_keys=True)
    return output_path
