import json

import pandas as pd
import pytest

from graphcalc.additive_combinatorics.dataset_exports import (
    additive_package_to_dataframe,
    additive_rows_to_dataframe,
    save_additive_column_definitions_json,
    save_additive_metadata_json,
    save_additive_package_csv,
)
from graphcalc.additive_combinatorics.dataset_generators import (
    additive_dataset_column_definitions,
    additive_set_to_record,
    small_additive_snapshot,
)
from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
from graphcalc.additive_combinatorics.sets import AdditiveSet


def test_additive_rows_to_dataframe_returns_dataframe():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)
    row = additive_set_to_record(A)

    df = additive_rows_to_dataframe([row])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert set(row.keys()).issubset(df.columns)


def test_additive_rows_to_dataframe_preserves_row_count():
    rows = small_additive_snapshot()

    df = additive_rows_to_dataframe(rows)

    assert len(df) == len(rows)


def test_additive_package_to_dataframe_reads_rows_entry():
    rows = small_additive_snapshot()
    package = {"rows": rows, "metadata": {"name": "example"}}

    df = additive_package_to_dataframe(package)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(rows)


def test_additive_package_to_dataframe_requires_rows_entry():
    with pytest.raises(KeyError, match="must contain a 'rows' entry"):
        additive_package_to_dataframe({"metadata": {}})


def test_save_additive_package_csv_writes_file(tmp_path):
    rows = small_additive_snapshot()
    package = {"rows": rows, "metadata": {"name": "snapshot"}}
    output = tmp_path / "snapshot.csv"

    written_path = save_additive_package_csv(package, output)

    assert written_path == output
    assert output.exists()

    df = pd.read_csv(output)
    assert len(df) == len(rows)


def test_save_additive_package_csv_respects_index_false(tmp_path):
    rows = small_additive_snapshot()
    package = {"rows": rows}
    output = tmp_path / "snapshot.csv"

    save_additive_package_csv(package, output, index=False)

    df = pd.read_csv(output)
    assert "Unnamed: 0" not in df.columns


def test_save_additive_column_definitions_json_writes_expected_content(tmp_path):
    output = tmp_path / "column_definitions.json"

    written_path = save_additive_column_definitions_json(output)

    assert written_path == output
    assert output.exists()

    with output.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload == additive_dataset_column_definitions()
    assert "cardinality" in payload
    assert "description" in payload["cardinality"]


def test_save_additive_metadata_json_writes_expected_content(tmp_path):
    metadata = {
        "dataset_name": "toy snapshot",
        "num_rows": 3,
        "notes": {"deterministic": True},
    }
    output = tmp_path / "metadata.json"

    written_path = save_additive_metadata_json(metadata, output)

    assert written_path == output
    assert output.exists()

    with output.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload == metadata
