"""
Tools for finite-dimensional quantum information in graphcalc.

The ``graphcalc.quantum`` package provides lightweight, mathematically
oriented support for finite-dimensional quantum states, channels, and
measurements. The design follows the same general philosophy as the rest of
graphcalc: a small number of core object classes together with functional
submodules for standard generators, numerical invariants, Boolean properties,
and dataset-construction helpers.

Scope
-----
This package currently focuses on:

- quantum states represented by density operators,
- standard state families used in quantum information and entanglement theory,
- quantum channels represented by Choi matrices,
- standard noisy channel families,
- finite measurements represented by measurement operators,
- local channel actions on multipartite states,
- dataset generation and export helpers for conjecturing workflows.

Core object classes
-------------------
``QuantumState``
    Finite-dimensional multipartite quantum state stored canonically as a
    density operator together with explicit subsystem dimensions.

``QuantumChannel``
    Finite-dimensional quantum channel stored canonically by its Choi matrix,
    with explicit input and output dimensions.

``QuantumMeasurement``
    Finite quantum measurement stored by a family of measurement operators,
    with effects derived as ``M_a^dagger M_a``.

Main submodule families
-----------------------
``states``
    Core quantum-state representation and state-level structural operations.

``generators``
    Named quantum-state constructors such as computational basis states, Bell
    states, GHZ states, W states, maximally mixed states, and Werner states.

``invariants``
    Numerical quantities attached to states, including purity, entropy,
    negativity, fidelity, entanglement entropy, and mutual information.

``properties``
    Boolean predicates for states, including validity, purity, PPT, product
    structure, and entanglement.

``channels``
    Core quantum-channel representation, alternate constructors, and channel
    composition/tensor operations.

``channel_generators``
    Standard channel families such as depolarizing, bit-flip, phase-flip,
    phase-damping, and amplitude-damping channels.

``channel_invariants``
    Numerical quantities attached to channels, especially Choi- and
    Kraus-rank-related data.

``channel_properties``
    Boolean predicates for channels, including complete positivity, trace
    preservation, unitality, and unitary-channel tests.

``measurements``
    Core measurement representation together with probabilities and
    post-measurement state updates.

``measurement_generators``
    Standard measurement families such as computational-basis, Pauli-basis,
    and Bell-basis measurements.

``measurement_properties``
    Boolean predicates for measurements, including POVM validity,
    projectiveness, and rank-one structure.

``local_channels``
    Helpers for applying local channels to specified subsystems of a
    multipartite quantum state.

``dataset_generators``
    Helpers for generating conjecturing-oriented datasets from combinations of
    state families, channel families, subsystem patterns, and measurements.

``dataset_exports``
    Helpers for exporting generated datasets to pandas DataFrames, CSV files,
    and JSON metadata.

Notes
-----
This package is intended for finite-dimensional computational experimentation,
small- to medium-scale dataset generation, and mathematically transparent
prototyping. It is not intended to be a full symbolic quantum-circuit
framework or a high-performance simulation library.
"""

from graphcalc.quantum.states import QuantumState
from graphcalc.quantum.generators import (
    basis_state,
    bell_state,
    computational_basis_state,
    ghz_state,
    maximally_mixed_state,
    minus_state,
    plus_state,
    w_state,
    werner_state,
)
from graphcalc.quantum.invariants import (
    entanglement_entropy,
    fidelity,
    linear_entropy,
    logarithmic_negativity,
    mutual_information,
    negativity,
    purity,
    rank,
    von_neumann_entropy,
)
from graphcalc.quantum.properties import (
    has_positive_partial_transpose,
    is_entangled,
    is_mixed,
    is_product_state,
    is_pure,
    is_valid_state,
)

from graphcalc.quantum.channels import QuantumChannel
from graphcalc.quantum.channel_generators import (
    amplitude_damping_channel,
    bit_flip_channel,
    bit_phase_flip_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
    phase_flip_channel,
)
from graphcalc.quantum.channel_invariants import (
    choi_eigenvalues,
    choi_rank,
    input_dimension,
    kraus_rank,
    output_dimension,
)
from graphcalc.quantum.channel_properties import (
    is_completely_positive,
    is_quantum_channel,
    is_trace_preserving,
    is_unital,
    is_unitary_channel,
)

from graphcalc.quantum.measurements import QuantumMeasurement
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

from graphcalc.quantum.local_channels import (
    apply_channel_to_subsystem,
    apply_channels_to_subsystems,
)

from graphcalc.quantum.dataset_generators import (
    generate_parameter_grid,
    generate_quantum_state_dataset,
    large_quantum_snapshot,
    medium_quantum_snapshot,
    quantum_dataset_column_definitions,
    small_quantum_snapshot,
)
from graphcalc.quantum.dataset_exports import (
    quantum_package_to_dataframe,
    quantum_rows_to_dataframe,
    save_quantum_column_definitions_json,
    save_quantum_metadata_json,
    save_quantum_package_csv,
)

__all__ = [
    # Core classes
    "QuantumState",
    "QuantumChannel",
    "QuantumMeasurement",
    # State generators
    "basis_state",
    "computational_basis_state",
    "plus_state",
    "minus_state",
    "bell_state",
    "ghz_state",
    "w_state",
    "maximally_mixed_state",
    "werner_state",
    # State invariants
    "purity",
    "rank",
    "linear_entropy",
    "von_neumann_entropy",
    "negativity",
    "logarithmic_negativity",
    "fidelity",
    "entanglement_entropy",
    "mutual_information",
    # State properties
    "is_valid_state",
    "is_pure",
    "is_mixed",
    "has_positive_partial_transpose",
    "is_product_state",
    "is_entangled",
    # Channel generators
    "identity_channel",
    "depolarizing_channel",
    "bit_flip_channel",
    "phase_flip_channel",
    "bit_phase_flip_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
    # Channel invariants
    "choi_rank",
    "kraus_rank",
    "input_dimension",
    "output_dimension",
    "choi_eigenvalues",
    # Channel properties
    "is_completely_positive",
    "is_trace_preserving",
    "is_unital",
    "is_quantum_channel",
    "is_unitary_channel",
    # Measurements
    "computational_basis_measurement",
    "pauli_x_measurement",
    "pauli_y_measurement",
    "pauli_z_measurement",
    "bell_basis_measurement",
    "is_povm",
    "is_projective_measurement",
    "is_rank_one_measurement",
    # Local channel actions
    "apply_channel_to_subsystem",
    "apply_channels_to_subsystems",
    # Dataset generation
    "quantum_dataset_column_definitions",
    "generate_quantum_state_dataset",
    "generate_parameter_grid",
    "small_quantum_snapshot",
    "medium_quantum_snapshot",
    "large_quantum_snapshot",
    # Dataset export
    "quantum_rows_to_dataframe",
    "quantum_package_to_dataframe",
    "save_quantum_package_csv",
    "save_quantum_column_definitions_json",
    "save_quantum_metadata_json",
]
