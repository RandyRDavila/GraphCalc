import numpy as np
import pytest

from graphcalc.quantum.channel_generators import (
    amplitude_damping_channel,
    bit_flip_channel,
    bit_phase_flip_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
    phase_flip_channel,
)
from graphcalc.quantum.generators import basis_state, maximally_mixed_state, plus_state
from graphcalc.quantum.states import QuantumState


def test_identity_channel_generator_matches_core_behavior():
    chan = identity_channel(2)
    state = plus_state()
    out = chan.apply(state)

    assert chan.input_dim == 2
    assert chan.output_dim == 2
    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert chan.is_unital()
    assert np.allclose(out.rho, state.rho)


def test_identity_channel_invalid_dimension_raises():
    with pytest.raises(ValueError, match="positive"):
        identity_channel(0)


def test_depolarizing_channel_endpoints_qubit():
    state = plus_state()

    chan0 = depolarizing_channel(0.0, dim=2)
    chan1 = depolarizing_channel(1.0, dim=2)

    out0 = chan0.apply(state)
    out1 = chan1.apply(state)

    expected1 = np.eye(2, dtype=complex) / 2

    assert chan0.is_completely_positive()
    assert chan0.is_trace_preserving()
    assert chan0.is_unital()
    assert np.allclose(out0.rho, state.rho)
    assert np.allclose(out1.rho, expected1)


def test_depolarizing_channel_qutrit_endpoint():
    chan = depolarizing_channel(1.0, dim=3)
    state = basis_state(2, dim=3)
    out = chan.apply(state)

    expected = np.eye(3, dtype=complex) / 3
    assert chan.input_dim == 3
    assert chan.output_dim == 3
    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert chan.is_unital()
    assert np.allclose(out.rho, expected)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_depolarizing_channel_invalid_p_raises(p):
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        depolarizing_channel(p)


def test_depolarizing_channel_invalid_dim_raises():
    with pytest.raises(ValueError, match="positive"):
        depolarizing_channel(0.5, dim=0)


def test_bit_flip_channel_endpoints():
    zero = basis_state(0, dim=2)
    one = basis_state(1, dim=2)

    chan0 = bit_flip_channel(0.0)
    chan1 = bit_flip_channel(1.0)

    out0 = chan0.apply(zero)
    out1 = chan1.apply(zero)

    assert chan0.is_completely_positive()
    assert chan0.is_trace_preserving()
    assert chan0.is_unital()
    assert np.allclose(out0.rho, zero.rho)
    assert np.allclose(out1.rho, one.rho)


@pytest.mark.parametrize("p", [-0.2, 1.2])
def test_bit_flip_channel_invalid_p_raises(p):
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        bit_flip_channel(p)


def test_phase_flip_channel_on_plus_state():
    chan = phase_flip_channel(1.0)
    plus = plus_state()
    out = chan.apply(plus)

    minus_rho = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)

    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert chan.is_unital()
    assert np.allclose(out.rho, minus_rho)


@pytest.mark.parametrize("p", [-0.3, 1.3])
def test_phase_flip_channel_invalid_p_raises(p):
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        phase_flip_channel(p)


def test_bit_phase_flip_channel_on_zero_state():
    chan = bit_phase_flip_channel(1.0)
    zero = basis_state(0, dim=2)
    out = chan.apply(zero)

    one_rho = np.array([[0, 0], [0, 1]], dtype=complex)

    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert chan.is_unital()
    assert np.allclose(out.rho, one_rho)


@pytest.mark.parametrize("p", [-0.4, 1.4])
def test_bit_phase_flip_channel_invalid_p_raises(p):
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        bit_phase_flip_channel(p)


def test_phase_damping_channel_preserves_populations_and_damps_coherences():
    chan = phase_damping_channel(0.75)
    plus = plus_state()
    out = chan.apply(plus)

    expected = np.array(
        [
            [0.5, 0.25],
            [0.25, 0.5],
        ],
        dtype=complex,
    )

    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert chan.is_unital()
    assert np.allclose(out.rho, expected)


def test_phase_damping_channel_endpoints():
    chan0 = phase_damping_channel(0.0)
    chan1 = phase_damping_channel(1.0)
    plus = plus_state()

    out0 = chan0.apply(plus)
    out1 = chan1.apply(plus)

    expected1 = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
        ],
        dtype=complex,
    )

    assert np.allclose(out0.rho, plus.rho)
    assert np.allclose(out1.rho, expected1)


@pytest.mark.parametrize("gamma", [-0.1, 1.1])
def test_phase_damping_channel_invalid_gamma_raises(gamma):
    with pytest.raises(ValueError, match="0 <= gamma <= 1"):
        phase_damping_channel(gamma)


def test_amplitude_damping_channel_fixes_ground_state():
    chan = amplitude_damping_channel(0.6)
    zero = basis_state(0, dim=2)
    out = chan.apply(zero)

    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert not chan.is_unital()
    assert np.allclose(out.rho, zero.rho)


def test_amplitude_damping_channel_excited_state_endpoint():
    chan = amplitude_damping_channel(1.0)
    one = basis_state(1, dim=2)
    zero = basis_state(0, dim=2)

    out = chan.apply(one)

    assert np.allclose(out.rho, zero.rho)


def test_amplitude_damping_channel_partial_damping_of_excited_state():
    gamma = 0.3
    chan = amplitude_damping_channel(gamma)
    one = basis_state(1, dim=2)
    out = chan.apply(one)

    expected = np.array(
        [
            [gamma, 0.0],
            [0.0, 1.0 - gamma],
        ],
        dtype=complex,
    )

    assert np.allclose(out.rho, expected)


def test_amplitude_damping_channel_on_plus_state():
    gamma = 0.36
    chan = amplitude_damping_channel(gamma)
    plus = plus_state()
    out = chan.apply(plus)

    expected = np.array(
        [
            [0.5 + 0.5 * gamma, 0.5 * np.sqrt(1.0 - gamma)],
            [0.5 * np.sqrt(1.0 - gamma), 0.5 * (1.0 - gamma)],
        ],
        dtype=complex,
    )

    assert np.allclose(out.rho, expected)


@pytest.mark.parametrize("gamma", [-0.2, 1.2])
def test_amplitude_damping_channel_invalid_gamma_raises(gamma):
    with pytest.raises(ValueError, match="0 <= gamma <= 1"):
        amplitude_damping_channel(gamma)


def test_all_named_qubit_channels_send_states_to_states():
    channels = [
        bit_flip_channel(0.2),
        phase_flip_channel(0.2),
        bit_phase_flip_channel(0.2),
        phase_damping_channel(0.2),
        amplitude_damping_channel(0.2),
        depolarizing_channel(0.2, dim=2),
    ]

    state = plus_state()
    for chan in channels:
        out = chan.apply(state)
        assert out.dims == (2,)
        assert np.isclose(out.trace.real, 1.0)
        assert np.all(out.eigenvalues() >= -out.tol)


def test_depolarizing_channel_sends_maximally_mixed_state_to_itself():
    chan = depolarizing_channel(0.7, dim=2)
    state = maximally_mixed_state(2)
    out = chan.apply(state)

    assert np.allclose(out.rho, state.rho)
