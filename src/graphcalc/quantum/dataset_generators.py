from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from graphcalc.quantum.channel_generators import (
    amplitude_damping_channel,
    bit_flip_channel,
    bit_phase_flip_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
    phase_flip_channel,
)
from graphcalc.quantum.channel_properties import (
    is_completely_positive as channel_is_completely_positive,
    is_quantum_channel,
    is_trace_preserving as channel_is_trace_preserving,
    is_unital as channel_is_unital,
    is_unitary_channel,
)
from graphcalc.quantum.generators import (
    bell_state,
    computational_basis_state,
    ghz_state,
    maximally_mixed_state,
    plus_state,
    w_state,
    werner_state,
)
from graphcalc.quantum.invariants import (
    entanglement_entropy,
    linear_entropy,
    logarithmic_negativity,
    mutual_information,
    negativity,
    purity,
    rank,
    von_neumann_entropy,
)
from graphcalc.quantum.local_channels import (
    apply_channel_to_subsystem,
    apply_channels_to_subsystems,
)
from graphcalc.quantum.measurement_generators import (
    bell_basis_measurement,
    computational_basis_measurement,
    pauli_x_measurement,
    pauli_y_measurement,
    pauli_z_measurement,
)
from graphcalc.quantum.measurement_properties import (
    is_povm,
    is_projective_measurement,
    is_rank_one_measurement,
)
from graphcalc.quantum.properties import (
    has_positive_partial_transpose,
    is_entangled,
    is_mixed,
    is_product_state,
    is_pure,
    is_valid_state,
)
from graphcalc.quantum.states import QuantumState

__all__ = [
    "quantum_dataset_column_definitions",
    "generate_quantum_state_dataset",
    "generate_parameter_grid",
    "small_quantum_snapshot",
    "medium_quantum_snapshot",
    "large_quantum_snapshot",
]


def quantum_dataset_column_definitions() -> Dict[str, str]:
    """
    Return a dictionary mapping dataset column names to mathematical meanings.

    Notes
    -----
    The returned dictionary is intended to describe the columns produced by
    ``generate_quantum_state_dataset``. Some columns may be absent from a
    particular dataset row if the corresponding quantity is not applicable.
    """
    return {
        "state_family": "Name of the base state family used to generate the row.",
        "state_label": "Human-readable label for the generated state instance.",
        "state_parameter": "Primary scalar parameter used for the state family, if any.",
        "state_parameters": "Dictionary of state-family parameters used to generate the row.",
        "num_subsystems": "Number of tensor-factor subsystems in the state.",
        "dims": "Tuple of subsystem dimensions.",
        "dimension": "Total Hilbert-space dimension, equal to the product of subsystem dimensions.",
        "channel_family": "Name of the local channel family applied to the state.",
        "channel_parameter": "Primary scalar parameter used for the channel family, if any.",
        "channel_parameters": "Dictionary of channel-family parameters used to generate the row.",
        "channel_subsystems": "Tuple of subsystem indices on which local channels were applied.",
        "measurement_family": "Name of the measurement family used for summary statistics, if any.",
        "measurement_probabilities": "Tuple of measurement-outcome probabilities for the selected measurement family.",
        "measurement_num_outcomes": "Number of outcomes in the selected measurement.",
        "measurement_is_povm": "Whether the selected measurement is a valid POVM.",
        "measurement_is_projective": "Whether the selected measurement is projective.",
        "measurement_is_rank_one": "Whether every nonzero effect of the selected measurement has rank one.",
        "state_is_valid": "Whether the final density operator is Hermitian, positive semidefinite, and trace one.",
        "state_is_pure": "Whether the final state is pure, equivalently Tr(rho^2)=1.",
        "state_is_mixed": "Whether the final state is mixed.",
        "state_is_product": "Whether the final state is product under the default bipartition convention of the properties module.",
        "state_is_entangled": "Whether the final state is entangled under the default bipartition convention of the properties module.",
        "state_has_ppt": "Whether the final state has positive partial transpose with respect to the selected PPT subsystem set.",
        "purity": "Purity Tr(rho^2) of the final state.",
        "rank": "Rank of the final density operator.",
        "linear_entropy": "Linear entropy 1 - Tr(rho^2) of the final state.",
        "von_neumann_entropy": "Von Neumann entropy of the final state.",
        "negativity": "Entanglement negativity with respect to the selected subsystem set for partial transpose.",
        "logarithmic_negativity": "Logarithmic negativity with respect to the selected subsystem set for partial transpose.",
        "entanglement_entropy": "Entropy of a reduced state for a pure-state bipartition; omitted when not defined.",
        "mutual_information": "Quantum mutual information between the selected subsystem sets A and B.",
        "ppt_subsystems": "Subsystem set used for PPT and negativity calculations.",
        "entanglement_entropy_subsystems": "Subsystem set used for entanglement-entropy calculation.",
        "mutual_information_subsystems_a": "First subsystem set used for mutual-information calculation.",
        "mutual_information_subsystems_b": "Second subsystem set used for mutual-information calculation.",
        "channel_is_cp": "Whether the selected channel is completely positive.",
        "channel_is_tp": "Whether the selected channel is trace preserving.",
        "channel_is_unital": "Whether the selected channel is unital.",
        "channel_is_quantum_channel": "Whether the selected channel is completely positive and trace preserving.",
        "channel_is_unitary": "Whether the selected channel is a unitary channel.",
    }


def _default_ppt_subsystems(state: QuantumState) -> tuple[int, ...]:
    if state.num_subsystems <= 1:
        return (0,)
    return (0,)


def _default_entanglement_entropy_subsystems(state: QuantumState) -> tuple[int, ...]:
    if state.num_subsystems <= 1:
        return (0,)
    return (0,)


def _default_mutual_information_subsystems(
    state: QuantumState,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if state.num_subsystems < 2:
        return None
    return ((0,), (1,))


def _state_from_spec(spec: Mapping[str, Any]) -> QuantumState:
    family = spec["family"]

    if family == "plus":
        return plus_state()
    if family == "bell":
        which = int(spec.get("which", 0))
        return bell_state(which)
    if family == "ghz":
        n = int(spec["n"])
        return ghz_state(n)
    if family == "w":
        n = int(spec["n"])
        return w_state(n)
    if family == "werner":
        p = float(spec["p"])
        return werner_state(p)
    if family == "maximally_mixed":
        dim = int(spec.get("dim", 2))
        return maximally_mixed_state(dim)
    if family == "computational_basis":
        bits = tuple(int(b) for b in spec["bits"])
        return computational_basis_state(bits)

    raise ValueError(f"Unknown state family: {family!r}")


def _state_label_from_spec(spec: Mapping[str, Any]) -> str:
    family = spec["family"]

    if family == "plus":
        return "plus"
    if family == "bell":
        return f"bell_{int(spec.get('which', 0))}"
    if family == "ghz":
        return f"ghz_{int(spec['n'])}"
    if family == "w":
        return f"w_{int(spec['n'])}"
    if family == "werner":
        return f"werner_{float(spec['p'])}"
    if family == "maximally_mixed":
        return f"maximally_mixed_{int(spec.get('dim', 2))}"
    if family == "computational_basis":
        bits = "".join(str(int(b)) for b in spec["bits"])
        return f"basis_{bits}"

    return str(family)


def _state_primary_parameter(spec: Mapping[str, Any]) -> float | int | None:
    family = spec["family"]
    if family in {"ghz", "w"}:
        return int(spec["n"])
    if family == "werner":
        return float(spec["p"])
    if family == "bell":
        return int(spec.get("which", 0))
    if family == "maximally_mixed":
        return int(spec.get("dim", 2))
    return None


def _channel_from_spec(spec: Mapping[str, Any]):
    family = spec["family"]

    if family == "identity":
        dim = int(spec.get("dim", 2))
        return identity_channel(dim)
    if family == "depolarizing":
        return depolarizing_channel(float(spec["p"]), dim=int(spec.get("dim", 2)))
    if family == "bit_flip":
        return bit_flip_channel(float(spec["p"]))
    if family == "phase_flip":
        return phase_flip_channel(float(spec["p"]))
    if family == "bit_phase_flip":
        return bit_phase_flip_channel(float(spec["p"]))
    if family == "phase_damping":
        return phase_damping_channel(float(spec["gamma"]))
    if family == "amplitude_damping":
        return amplitude_damping_channel(float(spec["gamma"]))

    raise ValueError(f"Unknown channel family: {family!r}")


def _channel_primary_parameter(spec: Mapping[str, Any]) -> float | int | None:
    if "p" in spec:
        return float(spec["p"])
    if "gamma" in spec:
        return float(spec["gamma"])
    if "dim" in spec:
        return int(spec["dim"])
    return None


def _measurement_from_name(name: str, state: QuantumState):
    if name == "computational_basis":
        return computational_basis_measurement(dim=state.dimension)
    if name == "pauli_x":
        if state.dimension != 2:
            raise ValueError("pauli_x measurement is currently defined only for qubits.")
        return pauli_x_measurement()
    if name == "pauli_y":
        if state.dimension != 2:
            raise ValueError("pauli_y measurement is currently defined only for qubits.")
        return pauli_y_measurement()
    if name == "pauli_z":
        if state.dimension != 2:
            raise ValueError("pauli_z measurement is currently defined only for qubits.")
        return pauli_z_measurement()
    if name == "bell_basis":
        if state.dimension != 4:
            raise ValueError("bell_basis measurement requires total dimension 4.")
        return bell_basis_measurement()

    raise ValueError(f"Unknown measurement family: {name!r}")


def _serialize_scalar_dict(spec: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in spec.items():
        if isinstance(value, tuple):
            out[key] = tuple(value)
        else:
            out[key] = value
    return out


def generate_quantum_state_dataset(
    *,
    state_specs: Sequence[Mapping[str, Any]],
    channel_specs: Sequence[Mapping[str, Any]] | None = None,
    subsystem_patterns: Sequence[Sequence[int]] | None = None,
    measurement_families: Sequence[str] | None = None,
    include_measurements: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate a conjecturing-oriented dataset of quantum-state snapshots.

    Parameters
    ----------
    state_specs : sequence of mappings
        Specifications describing base-state families to generate.
    channel_specs : sequence of mappings | None, default=None
        Specifications describing local channel families. If omitted, only the
        base states are used.
    subsystem_patterns : sequence of sequences of int | None, default=None
        Subsystem index patterns on which the selected local channels are
        applied. If omitted, the empty pattern is used, meaning no noise.
    measurement_families : sequence of str | None, default=None
        Optional measurement families for probability summaries. Supported
        values currently include:
        ``"computational_basis"``, ``"pauli_x"``, ``"pauli_y"``, ``"pauli_z"``,
        and ``"bell_basis"`` when dimension permits.
    include_measurements : bool, default=True
        Whether to include measurement summary columns.

    Returns
    -------
    list of dict
        Dataset rows suitable for conversion to a pandas DataFrame.

    Notes
    -----
    Each row corresponds to:

    - one base state specification,
    - one local channel specification (or none),
    - one subsystem pattern,
    - one optional measurement family.

    The resulting rows are intentionally denormalized so that each row is
    self-contained for downstream conjecturing code.
    """
    if not state_specs:
        raise ValueError("state_specs must be nonempty.")

    if channel_specs is None:
        channel_specs = [{"family": "identity", "dim": 2}]
    if subsystem_patterns is None:
        subsystem_patterns = [()]
    if measurement_families is None:
        measurement_families = ["computational_basis"] if include_measurements else []

    rows: List[Dict[str, Any]] = []

    for state_spec in state_specs:
        base_state = _state_from_spec(state_spec)

        compatible_channel_specs = list(channel_specs)
        if not compatible_channel_specs:
            compatible_channel_specs = [{"family": "identity", "dim": base_state.dimension}]

        for channel_spec in compatible_channel_specs:
            for subsystem_pattern in subsystem_patterns:
                subsystem_pattern = tuple(int(i) for i in subsystem_pattern)

                if any(i < 0 or i >= base_state.num_subsystems for i in subsystem_pattern):
                    continue

                if subsystem_pattern:
                    channel = _channel_from_spec(channel_spec)
                    if len(subsystem_pattern) == 1:
                        final_state = apply_channel_to_subsystem(
                            base_state, channel, subsystem_pattern[0]
                        )
                    else:
                        channels = [channel for _ in subsystem_pattern]
                        final_state = apply_channels_to_subsystems(
                            base_state, channels, subsystem_pattern
                        )
                    channel_family = str(channel_spec["family"])
                    channel_parameter = _channel_primary_parameter(channel_spec)
                    channel_parameters = _serialize_scalar_dict(channel_spec)
                    channel_cp = channel_is_completely_positive(channel)
                    channel_tp = channel_is_trace_preserving(channel)
                    channel_unital = channel_is_unital(channel)
                    channel_q = is_quantum_channel(channel)
                    channel_unitary = is_unitary_channel(channel)
                else:
                    final_state = base_state.copy()
                    channel_family = "none"
                    channel_parameter = None
                    channel_parameters = {}
                    channel_cp = None
                    channel_tp = None
                    channel_unital = None
                    channel_q = None
                    channel_unitary = None

                ppt_subsystems = _default_ppt_subsystems(final_state)
                ee_subsystems = _default_entanglement_entropy_subsystems(final_state)
                mi_subsystems = _default_mutual_information_subsystems(final_state)

                row_base: Dict[str, Any] = {
                    "state_family": str(state_spec["family"]),
                    "state_label": _state_label_from_spec(state_spec),
                    "state_parameter": _state_primary_parameter(state_spec),
                    "state_parameters": _serialize_scalar_dict(state_spec),
                    "num_subsystems": final_state.num_subsystems,
                    "dims": tuple(int(d) for d in final_state.dims),
                    "dimension": final_state.dimension,
                    "channel_family": channel_family,
                    "channel_parameter": channel_parameter,
                    "channel_parameters": channel_parameters,
                    "channel_subsystems": tuple(subsystem_pattern),
                    "state_is_valid": is_valid_state(final_state),
                    "state_is_pure": is_pure(final_state),
                    "state_is_mixed": is_mixed(final_state),
                    "state_is_product": is_product_state(final_state),
                    "state_is_entangled": is_entangled(final_state),
                    "state_has_ppt": has_positive_partial_transpose(
                        final_state, subsystems=ppt_subsystems
                    ),
                    "purity": purity(final_state),
                    "rank": rank(final_state),
                    "linear_entropy": linear_entropy(final_state),
                    "von_neumann_entropy": von_neumann_entropy(final_state),
                    "negativity": negativity(final_state, subsystems=ppt_subsystems),
                    "logarithmic_negativity": logarithmic_negativity(
                        final_state, subsystems=ppt_subsystems
                    ),
                    "ppt_subsystems": tuple(ppt_subsystems),
                    "entanglement_entropy_subsystems": tuple(ee_subsystems),
                    "mutual_information_subsystems_a": None if mi_subsystems is None else tuple(mi_subsystems[0]),
                    "mutual_information_subsystems_b": None if mi_subsystems is None else tuple(mi_subsystems[1]),
                    "channel_is_cp": channel_cp,
                    "channel_is_tp": channel_tp,
                    "channel_is_unital": channel_unital,
                    "channel_is_quantum_channel": channel_q,
                    "channel_is_unitary": channel_unitary,
                }

                if final_state.is_pure:
                    row_base["entanglement_entropy"] = entanglement_entropy(
                        final_state,
                        subsystems=ee_subsystems,
                    )
                else:
                    row_base["entanglement_entropy"] = None

                if mi_subsystems is None:
                    row_base["mutual_information"] = None
                else:
                    row_base["mutual_information"] = mutual_information(
                        final_state,
                        subsystems_a=mi_subsystems[0],
                        subsystems_b=mi_subsystems[1],
                    )

                if not include_measurements:
                    rows.append(row_base)
                    continue

                if not measurement_families:
                    rows.append(row_base)
                    continue

                for measurement_family in measurement_families:
                    try:
                        measurement = _measurement_from_name(measurement_family, final_state)
                    except ValueError:
                        continue

                    row = dict(row_base)
                    row["measurement_family"] = measurement_family
                    row["measurement_probabilities"] = tuple(
                        float(x) for x in measurement.outcome_probabilities(final_state)
                    )
                    row["measurement_num_outcomes"] = measurement.num_outcomes
                    row["measurement_is_povm"] = is_povm(measurement)
                    row["measurement_is_projective"] = is_projective_measurement(measurement)
                    row["measurement_is_rank_one"] = is_rank_one_measurement(measurement)
                    rows.append(row)

    return rows

def generate_parameter_grid(
    *,
    state_specs: Sequence[Mapping[str, Any]],
    channel_specs: Sequence[Mapping[str, Any]] | None = None,
    subsystem_patterns: Sequence[Sequence[int]] | None = None,
    measurement_families: Sequence[str] | None = None,
    include_measurements: bool = True,
) -> Dict[str, Any]:
    """
    Generate a conjecturing-ready dataset package together with metadata.

    Parameters
    ----------
    state_specs : sequence of mappings
        Specifications describing base-state families.
    channel_specs : sequence of mappings | None, default=None
        Specifications describing local channel families.
    subsystem_patterns : sequence of sequences of int | None, default=None
        Target subsystem patterns for local channel application.
    measurement_families : sequence of str | None, default=None
        Optional measurement families for probability summaries.
    include_measurements : bool, default=True
        Whether to include measurement summary columns.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"rows"`` : list of row dictionaries,
        - ``"column_definitions"`` : dictionary mapping columns to meanings,
        - ``"metadata"`` : summary information about the generated dataset.
    """
    rows = generate_quantum_state_dataset(
        state_specs=state_specs,
        channel_specs=channel_specs,
        subsystem_patterns=subsystem_patterns,
        measurement_families=measurement_families,
        include_measurements=include_measurements,
    )

    return {
        "rows": rows,
        "column_definitions": quantum_dataset_column_definitions(),
        "metadata": {
            "num_rows": len(rows),
            "num_state_specs": len(state_specs),
            "num_channel_specs": 0 if channel_specs is None else len(channel_specs),
            "num_subsystem_patterns": 0 if subsystem_patterns is None else len(subsystem_patterns),
            "num_measurement_families": 0 if measurement_families is None else len(measurement_families),
            "include_measurements": include_measurements,
        },
    }


def small_quantum_snapshot() -> Dict[str, Any]:
    """
    Return a small snapshot dataset for quick experimentation.

    Notes
    -----
    This snapshot is intentionally compact and includes a small variety of
    state families, a few low-complexity noise settings, and basic measurement
    summaries.
    """
    state_specs = [
        {"family": "plus"},
        {"family": "bell", "which": 0},
        {"family": "ghz", "n": 3},
        {"family": "werner", "p": 0.25},
    ]

    channel_specs = [
        {"family": "identity", "dim": 2},
        {"family": "phase_damping", "gamma": 0.2},
        {"family": "amplitude_damping", "gamma": 0.2},
    ]

    subsystem_patterns = [
        (),
        (0,),
    ]

    measurement_families = [
        "computational_basis",
    ]

    return generate_parameter_grid(
        state_specs=state_specs,
        channel_specs=channel_specs,
        subsystem_patterns=subsystem_patterns,
        measurement_families=measurement_families,
        include_measurements=True,
    )


def medium_quantum_snapshot() -> Dict[str, Any]:
    """
    Return a medium-sized snapshot dataset for broader conjecturing sweeps.

    Notes
    -----
    This snapshot expands both the family diversity and the parameter grid,
    while remaining small enough to inspect interactively.
    """
    state_specs = [
        {"family": "plus"},
        {"family": "bell", "which": 0},
        {"family": "bell", "which": 2},
        {"family": "ghz", "n": 3},
        {"family": "w", "n": 3},
        {"family": "werner", "p": 0.25},
        {"family": "werner", "p": 0.5},
        {"family": "werner", "p": 0.75},
        {"family": "computational_basis", "bits": (0, 0)},
        {"family": "computational_basis", "bits": (0, 1)},
    ]

    channel_specs = [
        {"family": "identity", "dim": 2},
        {"family": "bit_flip", "p": 0.2},
        {"family": "phase_flip", "p": 0.2},
        {"family": "phase_damping", "gamma": 0.2},
        {"family": "amplitude_damping", "gamma": 0.2},
        {"family": "depolarizing", "p": 0.2, "dim": 2},
    ]

    subsystem_patterns = [
        (),
        (0,),
        (1,),
    ]

    measurement_families = [
        "computational_basis",
        "pauli_x",
        "pauli_z",
        "bell_basis",
    ]

    return generate_parameter_grid(
        state_specs=state_specs,
        channel_specs=channel_specs,
        subsystem_patterns=subsystem_patterns,
        measurement_families=measurement_families,
        include_measurements=True,
    )


def large_quantum_snapshot() -> Dict[str, Any]:
    """
    Return a larger snapshot dataset for more substantial conjecturing runs.

    Notes
    -----
    This snapshot is still deliberately bounded to standard low-dimensional
    families, but it creates a much richer Cartesian-product sweep over
    states, local noise models, subsystem patterns, and measurements.
    """
    state_specs = [
        {"family": "plus"},
        {"family": "bell", "which": 0},
        {"family": "bell", "which": 1},
        {"family": "bell", "which": 2},
        {"family": "bell", "which": 3},
        {"family": "ghz", "n": 3},
        {"family": "ghz", "n": 4},
        {"family": "w", "n": 3},
        {"family": "w", "n": 4},
        {"family": "werner", "p": 0.1},
        {"family": "werner", "p": 0.25},
        {"family": "werner", "p": 0.5},
        {"family": "werner", "p": 0.75},
        {"family": "werner", "p": 0.9},
        {"family": "computational_basis", "bits": (0,)},
        {"family": "computational_basis", "bits": (1,)},
        {"family": "computational_basis", "bits": (0, 0)},
        {"family": "computational_basis", "bits": (0, 1)},
        {"family": "computational_basis", "bits": (1, 0)},
        {"family": "computational_basis", "bits": (1, 1)},
    ]

    channel_specs = [
        {"family": "identity", "dim": 2},
        {"family": "bit_flip", "p": 0.1},
        {"family": "bit_flip", "p": 0.3},
        {"family": "phase_flip", "p": 0.1},
        {"family": "phase_flip", "p": 0.3},
        {"family": "bit_phase_flip", "p": 0.2},
        {"family": "phase_damping", "gamma": 0.1},
        {"family": "phase_damping", "gamma": 0.3},
        {"family": "amplitude_damping", "gamma": 0.1},
        {"family": "amplitude_damping", "gamma": 0.3},
        {"family": "depolarizing", "p": 0.1, "dim": 2},
        {"family": "depolarizing", "p": 0.3, "dim": 2},
    ]

    subsystem_patterns = [
        (),
        (0,),
        (1,),
        (0, 1),
        (0, 2),
    ]

    measurement_families = [
        "computational_basis",
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "bell_basis",
    ]

    return generate_parameter_grid(
        state_specs=state_specs,
        channel_specs=channel_specs,
        subsystem_patterns=subsystem_patterns,
        measurement_families=measurement_families,
        include_measurements=True,
    )
