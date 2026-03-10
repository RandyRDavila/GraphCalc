import numpy as np

from graphcalc.quantum.channel_generators import (
    amplitude_damping_channel,
    bit_flip_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
    phase_flip_channel,
)
from graphcalc.quantum.channel_properties import (
    is_completely_positive,
    is_quantum_channel,
    is_trace_preserving,
    is_unital,
    is_unitary_channel,
)
from graphcalc.quantum.channels import QuantumChannel


def test_identity_channel_properties():
    chan = identity_channel(2)

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert is_unital(chan)
    assert is_quantum_channel(chan)
    assert is_unitary_channel(chan)


def test_bit_flip_channel_is_unitary_at_extreme_p_one():
    chan = bit_flip_channel(1.0)

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert is_unital(chan)
    assert is_quantum_channel(chan)
    assert is_unitary_channel(chan)


def test_phase_flip_channel_is_unitary_at_extreme_p_one():
    chan = phase_flip_channel(1.0)

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert is_unital(chan)
    assert is_quantum_channel(chan)
    assert is_unitary_channel(chan)


def test_depolarizing_channel_is_not_unitary_for_nontrivial_parameter():
    chan = depolarizing_channel(0.4, dim=2)

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert is_unital(chan)
    assert is_quantum_channel(chan)
    assert not is_unitary_channel(chan)


def test_phase_damping_channel_is_not_unitary_for_nontrivial_parameter():
    chan = phase_damping_channel(0.4)

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert is_unital(chan)
    assert is_quantum_channel(chan)
    assert not is_unitary_channel(chan)


def test_amplitude_damping_channel_is_not_unital():
    chan = amplitude_damping_channel(0.4)

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert not is_unital(chan)
    assert is_quantum_channel(chan)
    assert not is_unitary_channel(chan)


def test_rectangular_trace_preserving_channel_is_not_unital_or_unitary():
    k0 = np.array([[1, 0]], dtype=complex)
    k1 = np.array([[0, 1]], dtype=complex)
    chan = QuantumChannel.from_kraus([k0, k1])

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert not is_unital(chan)
    assert is_quantum_channel(chan)
    assert not is_unitary_channel(chan)


def test_cp_but_not_trace_preserving_operator_is_not_quantum_channel():
    choi = np.eye(4, dtype=complex)
    chan = QuantumChannel.from_choi(choi, input_dim=2, output_dim=2, validate=False)

    assert is_completely_positive(chan)
    assert not is_trace_preserving(chan)
    assert not is_unital(chan)
    assert not is_quantum_channel(chan)
    assert not is_unitary_channel(chan)


def test_non_cp_operator_fails_quantum_channel_test():
    choi = np.diag([1.0, -0.25, 0.25, 1.0]).astype(complex)
    chan = QuantumChannel.from_choi(choi, input_dim=2, output_dim=2, validate=False)

    assert not is_completely_positive(chan)
    assert not is_quantum_channel(chan)
    assert not is_unitary_channel(chan)


def test_identity_channel_in_dimension_three_is_unitary():
    chan = identity_channel(3)

    assert is_completely_positive(chan)
    assert is_trace_preserving(chan)
    assert is_unital(chan)
    assert is_quantum_channel(chan)
    assert is_unitary_channel(chan)
