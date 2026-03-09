from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

__all__ = [
    "quantum_rows_to_dataframe",
    "quantum_package_to_dataframe",
    "save_quantum_package_csv",
    "save_quantum_column_definitions_json",
    "save_quantum_metadata_json",
]


def quantum_rows_to_dataframe(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Convert a sequence of dataset rows into a pandas DataFrame.

    Parameters
    ----------
    rows : sequence of mappings
        Row dictionaries, typically produced by a quantum dataset generator.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the supplied rows.

    Notes
    -----
    This function preserves the row dictionaries as-is, allowing tuple-valued
    columns such as subsystem patterns or measurement-probability tuples to
    remain available in the resulting DataFrame.
    """
    return pd.DataFrame(list(rows))


def quantum_package_to_dataframe(package: Mapping[str, Any]) -> pd.DataFrame:
    """
    Convert a quantum dataset package into a pandas DataFrame.

    Parameters
    ----------
    package : mapping
        Dataset package expected to contain a ``"rows"`` entry.

    Returns
    -------
    pandas.DataFrame
        DataFrame built from ``package["rows"]``.

    Raises
    ------
    ValueError
        If the package does not contain a ``"rows"`` key.
    """
    if "rows" not in package:
        raise ValueError('package must contain a "rows" entry.')

    rows = package["rows"]
    return quantum_rows_to_dataframe(rows)


def save_quantum_package_csv(
    package: Mapping[str, Any],
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """
    Save the rows of a quantum dataset package as a CSV file.

    Parameters
    ----------
    package : mapping
        Dataset package expected to contain a ``"rows"`` entry.
    path : str or pathlib.Path
        Output CSV path.
    index : bool, default=False
        Whether to write the DataFrame index to the CSV file.

    Returns
    -------
    pathlib.Path
        The resolved output path.

    Notes
    -----
    Non-scalar objects such as tuples and dictionaries are serialized by pandas
    using their standard string representations in the CSV output.
    """
    out_path = Path(path)
    df = quantum_package_to_dataframe(package)
    df.to_csv(out_path, index=index)
    return out_path


def save_quantum_column_definitions_json(
    package: Mapping[str, Any],
    path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """
    Save the column-definition dictionary of a quantum dataset package as JSON.

    Parameters
    ----------
    package : mapping
        Dataset package expected to contain a ``"column_definitions"`` entry.
    path : str or pathlib.Path
        Output JSON path.
    indent : int, default=2
        JSON indentation level.
    sort_keys : bool, default=True
        Whether to sort keys in the output JSON.

    Returns
    -------
    pathlib.Path
        The resolved output path.

    Raises
    ------
    ValueError
        If the package does not contain a ``"column_definitions"`` key.
    """
    if "column_definitions" not in package:
        raise ValueError('package must contain a "column_definitions" entry.')

    out_path = Path(path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            package["column_definitions"],
            f,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )
        f.write("\n")

    return out_path


def save_quantum_metadata_json(
    package: Mapping[str, Any],
    path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """
    Save the metadata dictionary of a quantum dataset package as JSON.

    Parameters
    ----------
    package : mapping
        Dataset package expected to contain a ``"metadata"`` entry.
    path : str or pathlib.Path
        Output JSON path.
    indent : int, default=2
        JSON indentation level.
    sort_keys : bool, default=True
        Whether to sort keys in the output JSON.

    Returns
    -------
    pathlib.Path
        The resolved output path.

    Raises
    ------
    ValueError
        If the package does not contain a ``"metadata"`` key.
    """
    if "metadata" not in package:
        raise ValueError('package must contain a "metadata" entry.')

    out_path = Path(path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            package["metadata"],
            f,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )
        f.write("\n")

    return out_path
