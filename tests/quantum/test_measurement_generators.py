import numpy as np

from graphcalc.quantum.generators import bell_state, plus_state
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
from graphcalc.quantum.states import QuantumState


def test_computational_basis_measurement_generator():
    meas = computational_basis_measurement(dim=2)

    assert meas.dim == 2
    assert meas.num_outcomes == 2
    assert is_povm(meas)
    assert is_projective_measurement(meas)
    assert is_rank_one_measurement(meas)


def test_pauli_z_measurement_on_basis_states():
    meas = pauli_z_measurement()
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)

    assert np.allclose(meas.outcome_probabilities(zero), np.array([1.0, 0.0]))
    assert np.allclose(meas.outcome_probabilities(one), np.array([0.0, 1.0]))


def test_pauli_x_measurement_on_plus_state():
    meas = pauli_x_measurement()
    plus = plus_state()

    probs = meas.outcome_probabilities(plus)
    assert np.allclose(probs, np.array([1.0, 0.0]))


def test_pauli_y_measurement_is_projective_rank_one_povm():
    meas = pauli_y_measurement()

    assert meas.dim == 2
    assert meas.num_outcomes == 2
    assert is_povm(meas)
    assert is_projective_measurement(meas)
    assert is_rank_one_measurement(meas)


def test_pauli_y_measurement_on_zero_state_is_uniform():
    meas = pauli_y_measurement()
    zero = QuantumState.basis_state(0, dim=2)

    probs = meas.outcome_probabilities(zero)
    assert np.allclose(probs, np.array([0.5, 0.5]))


def test_bell_basis_measurement_properties():
    meas = bell_basis_measurement()

    assert meas.dim == 4
    assert meas.num_outcomes == 4
    assert is_povm(meas)
    assert is_projective_measurement(meas)
    assert is_rank_one_measurement(meas)


def test_bell_basis_measurement_on_bell_state_zero():
    meas = bell_basis_measurement()
    state = bell_state(0)

    probs = meas.outcome_probabilities(state)
    expected = np.array([1.0, 0.0, 0.0, 0.0])

    assert np.allclose(probs, expected)


def test_bell_basis_measurement_post_measurement_state():
    meas = bell_basis_measurement()
    state = bell_state(2)

    post = meas.post_measurement_state(state, 2)
    assert np.allclose(post.rho, state.rho)
