import json

import pandas as pd
import pytest

from graphcalc.quantum.dataset_exports import (
    quantum_package_to_dataframe,
    quantum_rows_to_dataframe,
    save_quantum_column_definitions_json,
    save_quantum_metadata_json,
    save_quantum_package_csv,
)
from graphcalc.quantum.dataset_generators import (
    generate_parameter_grid,
    small_quantum_snapshot,
)


def test_quantum_rows_to_dataframe_basic():
    rows = [
        {"state_family": "plus", "purity": 1.0},
        {"state_family": "bell", "purity": 1.0},
    ]

    df = quantum_rows_to_dataframe(rows)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["state_family", "purity"]
    assert len(df) == 2
    assert df.loc[0, "state_family"] == "plus"
    assert df.loc[1, "purity"] == 1.0


def test_quantum_package_to_dataframe_basic():
    package = generate_parameter_grid(
        state_specs=[{"family": "plus"}],
        channel_specs=[],
        subsystem_patterns=[()],
        measurement_families=["pauli_x"],
        include_measurements=True,
    )

    df = quantum_package_to_dataframe(package)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "state_family" in df.columns
    assert "measurement_family" in df.columns


def test_quantum_package_to_dataframe_missing_rows_raises():
    with pytest.raises(ValueError, match='package must contain a "rows" entry'):
        quantum_package_to_dataframe({})


def test_save_quantum_package_csv(tmp_path):
    package = small_quantum_snapshot()
    out_path = tmp_path / "snapshot.csv"

    returned = save_quantum_package_csv(package, out_path)

    assert returned == out_path
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert len(df) == package["metadata"]["num_rows"]
    assert "state_family" in df.columns


def test_save_quantum_column_definitions_json(tmp_path):
    package = small_quantum_snapshot()
    out_path = tmp_path / "column_definitions.json"

    returned = save_quantum_column_definitions_json(package, out_path)

    assert returned == out_path
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "state_family" in data
    assert "purity" in data


def test_save_quantum_column_definitions_json_missing_key_raises(tmp_path):
    out_path = tmp_path / "column_definitions.json"

    with pytest.raises(ValueError, match='package must contain a "column_definitions" entry'):
        save_quantum_column_definitions_json({}, out_path)


def test_save_quantum_metadata_json(tmp_path):
    package = small_quantum_snapshot()
    out_path = tmp_path / "metadata.json"

    returned = save_quantum_metadata_json(package, out_path)

    assert returned == out_path
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "num_rows" in data
    assert data["num_rows"] == package["metadata"]["num_rows"]


def test_save_quantum_metadata_json_missing_key_raises(tmp_path):
    out_path = tmp_path / "metadata.json"

    with pytest.raises(ValueError, match='package must contain a "metadata" entry'):
        save_quantum_metadata_json({}, out_path)
