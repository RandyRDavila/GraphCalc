import numpy as np

from graphcalc.quantum.dataset_generators import (
    generate_parameter_grid,
    generate_quantum_state_dataset,
    large_quantum_snapshot,
    medium_quantum_snapshot,
    quantum_dataset_column_definitions,
    small_quantum_snapshot,
)


def test_column_definitions_contains_core_keys():
    defs = quantum_dataset_column_definitions()

    assert "state_family" in defs
    assert "purity" in defs
    assert "negativity" in defs
    assert "measurement_probabilities" in defs
    assert "channel_is_quantum_channel" in defs


def test_generate_dataset_for_base_state_without_noise():
    rows = generate_quantum_state_dataset(
        state_specs=[{"family": "plus"}],
        channel_specs=[],
        subsystem_patterns=[()],
        measurement_families=["pauli_x"],
        include_measurements=True,
    )

    assert len(rows) == 1
    row = rows[0]

    assert row["state_family"] == "plus"
    assert row["state_is_pure"] is True
    assert np.isclose(row["purity"], 1.0)
    assert row["measurement_family"] == "pauli_x"
    assert np.allclose(row["measurement_probabilities"], (1.0, 0.0))


def test_generate_dataset_for_bell_state_with_local_phase_damping():
    rows = generate_quantum_state_dataset(
        state_specs=[{"family": "bell", "which": 0}],
        channel_specs=[{"family": "phase_damping", "gamma": 0.5}],
        subsystem_patterns=[(0,)],
        measurement_families=["computational_basis"],
        include_measurements=True,
    )

    assert len(rows) == 1
    row = rows[0]

    assert row["state_family"] == "bell"
    assert row["channel_family"] == "phase_damping"
    assert row["channel_subsystems"] == (0,)
    assert row["channel_is_quantum_channel"] is True
    assert row["measurement_family"] == "computational_basis"
    assert len(row["measurement_probabilities"]) == 4
    assert np.isclose(sum(row["measurement_probabilities"]), 1.0)
    assert row["state_is_valid"] is True


def test_generate_dataset_with_multiple_measurements_expands_rows():
    rows = generate_quantum_state_dataset(
        state_specs=[{"family": "plus"}],
        channel_specs=[],
        subsystem_patterns=[()],
        measurement_families=["pauli_x", "pauli_z"],
        include_measurements=True,
    )

    assert len(rows) == 2
    families = {row["measurement_family"] for row in rows}
    assert families == {"pauli_x", "pauli_z"}


def test_generate_dataset_without_measurements():
    rows = generate_quantum_state_dataset(
        state_specs=[{"family": "ghz", "n": 3}],
        channel_specs=[{"family": "identity", "dim": 2}],
        subsystem_patterns=[(0,), (1,)],
        include_measurements=False,
    )

    assert len(rows) == 2
    for row in rows:
        assert "measurement_family" not in row
        assert row["state_family"] == "ghz"
        assert row["dims"] == (2, 2, 2)


def test_generate_dataset_for_multiple_subsystem_local_noise():
    rows = generate_quantum_state_dataset(
        state_specs=[{"family": "ghz", "n": 3}],
        channel_specs=[{"family": "amplitude_damping", "gamma": 0.25}],
        subsystem_patterns=[(0, 2)],
        measurement_families=["computational_basis"],
        include_measurements=True,
    )

    assert len(rows) == 1
    row = rows[0]

    assert row["channel_subsystems"] == (0, 2)
    assert row["channel_family"] == "amplitude_damping"
    assert row["state_is_valid"] is True
    assert np.isclose(sum(row["measurement_probabilities"]), 1.0)


def test_generate_dataset_for_werner_state_records_parameter():
    rows = generate_quantum_state_dataset(
        state_specs=[{"family": "werner", "p": 0.75}],
        channel_specs=[],
        subsystem_patterns=[()],
        measurement_families=["bell_basis"],
        include_measurements=True,
    )

    assert len(rows) == 1
    row = rows[0]

    assert row["state_family"] == "werner"
    assert np.isclose(row["state_parameter"], 0.75)
    assert row["state_parameters"]["p"] == 0.75
    assert row["measurement_family"] == "bell_basis"
    assert len(row["measurement_probabilities"]) == 4


def test_generate_dataset_rejects_empty_state_specs():
    try:
        generate_quantum_state_dataset(state_specs=[])
    except ValueError as exc:
        assert "state_specs must be nonempty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty state_specs.")


def test_generate_parameter_grid_returns_rows_definitions_and_metadata():
    package = generate_parameter_grid(
        state_specs=[{"family": "plus"}],
        channel_specs=[],
        subsystem_patterns=[()],
        measurement_families=["pauli_x"],
        include_measurements=True,
    )

    assert "rows" in package
    assert "column_definitions" in package
    assert "metadata" in package
    assert package["metadata"]["num_rows"] == 1
    assert isinstance(package["rows"], list)
    assert isinstance(package["column_definitions"], dict)


def test_small_quantum_snapshot_structure():
    package = small_quantum_snapshot()

    assert "rows" in package
    assert "column_definitions" in package
    assert "metadata" in package
    assert package["metadata"]["num_rows"] == len(package["rows"])
    assert len(package["rows"]) > 0


def test_medium_quantum_snapshot_structure():
    package = medium_quantum_snapshot()

    assert "rows" in package
    assert "column_definitions" in package
    assert "metadata" in package
    assert package["metadata"]["num_rows"] == len(package["rows"])
    assert len(package["rows"]) > 0


def test_large_quantum_snapshot_structure():
    package = large_quantum_snapshot()

    assert "rows" in package
    assert "column_definitions" in package
    assert "metadata" in package
    assert package["metadata"]["num_rows"] == len(package["rows"])
    assert len(package["rows"]) > 0


def test_snapshot_sizes_are_monotone():
    small = small_quantum_snapshot()
    medium = medium_quantum_snapshot()
    large = large_quantum_snapshot()

    assert len(small["rows"]) <= len(medium["rows"])
    assert len(medium["rows"]) <= len(large["rows"])


def test_snapshot_column_definitions_include_expected_keys():
    package = small_quantum_snapshot()
    defs = package["column_definitions"]

    assert "state_family" in defs
    assert "channel_family" in defs
    assert "measurement_family" in defs
    assert "purity" in defs
    assert "negativity" in defs
