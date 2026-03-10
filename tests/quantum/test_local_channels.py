import numpy as np
import pytest

from graphcalc.quantum.channel_generators import (
    amplitude_damping_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
)
from graphcalc.quantum.generators import bell_state, ghz_state
from graphcalc.quantum.local_channels import (
    apply_channel_to_subsystem,
    apply_channels_to_subsystems,
)
from graphcalc.quantum.states import QuantumState


def test_identity_channel_on_one_subsystem_preserves_bell_state():
    state = bell_state(0)
    channel = identity_channel(2)

    out0 = apply_channel_to_subsystem(state, channel, 0)
    out1 = apply_channel_to_subsystem(state, channel, 1)

    assert out0.dims == (2, 2)
    assert out1.dims == (2, 2)
    assert np.allclose(out0.rho, state.rho)
    assert np.allclose(out1.rho, state.rho)


def test_phase_damping_on_one_half_of_bell_state_preserves_dimensions():
    state = bell_state(0)
    channel = phase_damping_channel(0.5)

    out = apply_channel_to_subsystem(state, channel, 0)

    assert out.dims == (2, 2)
    assert np.isclose(out.trace.real, 1.0)
    assert np.all(out.eigenvalues() >= -out.tol)


def test_amplitude_damping_on_bell_state_changes_state():
    state = bell_state(0)
    channel = amplitude_damping_channel(0.4)

    out = apply_channel_to_subsystem(state, channel, 0)

    assert out.dims == (2, 2)
    assert not np.allclose(out.rho, state.rho)
    assert np.isclose(out.trace.real, 1.0)


def test_apply_channel_to_subsystem_invalid_index_raises():
    state = bell_state(0)
    channel = identity_channel(2)

    with pytest.raises(ValueError, match="out of range"):
        apply_channel_to_subsystem(state, channel, 2)


def test_apply_channel_to_subsystem_dimension_mismatch_raises():
    state = bell_state(0)
    channel = identity_channel(3)

    with pytest.raises(ValueError, match="does not match"):
        apply_channel_to_subsystem(state, channel, 0)


def test_apply_channel_to_subsystem_rejects_nonsquare_channel():
    state = bell_state(0)
    k0 = np.array([[1, 0]], dtype=complex)
    k1 = np.array([[0, 1]], dtype=complex)

    from graphcalc.quantum.channels import QuantumChannel

    channel = QuantumChannel.from_kraus([k0, k1])

    with pytest.raises(ValueError, match="square channel"):
        apply_channel_to_subsystem(state, channel, 0)


def test_apply_channels_to_subsystems_identity_on_ghz_state():
    state = ghz_state(3)
    channels = [identity_channel(2), identity_channel(2)]
    subsystems = [0, 2]

    out = apply_channels_to_subsystems(state, channels, subsystems)

    assert out.dims == (2, 2, 2)
    assert np.allclose(out.rho, state.rho)


def test_apply_channels_to_subsystems_distinct_noise():
    state = ghz_state(3)
    channels = [phase_damping_channel(0.3), amplitude_damping_channel(0.2)]
    subsystems = [0, 2]

    out = apply_channels_to_subsystems(state, channels, subsystems)

    assert out.dims == (2, 2, 2)
    assert np.isclose(out.trace.real, 1.0)
    assert np.all(out.eigenvalues() >= -out.tol)


def test_apply_channels_to_subsystems_length_mismatch_raises():
    state = ghz_state(3)
    channels = [identity_channel(2)]
    subsystems = [0, 1]

    with pytest.raises(ValueError, match="same length"):
        apply_channels_to_subsystems(state, channels, subsystems)


def test_apply_channels_to_subsystems_rejects_repeated_subsystems():
    state = ghz_state(3)
    channels = [identity_channel(2), depolarizing_channel(0.2, dim=2)]
    subsystems = [1, 1]

    with pytest.raises(ValueError, match="distinct"):
        apply_channels_to_subsystems(state, channels, subsystems)


def test_local_action_matches_global_channel_on_single_subsystem_state():
    state = QuantumState.basis_state(1, dim=2)
    channel = amplitude_damping_channel(0.25)

    local_out = apply_channel_to_subsystem(state, channel, 0)
    direct_out = channel.apply(state)

    assert np.allclose(local_out.rho, direct_out.rho)
