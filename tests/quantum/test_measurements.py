import numpy as np
import pytest

from graphcalc.quantum.generators import plus_state
from graphcalc.quantum.measurements import QuantumMeasurement
from graphcalc.quantum.states import QuantumState


def test_computational_basis_measurement_qubit():
    meas = QuantumMeasurement.computational_basis(dim=2)

    assert meas.dim == 2
    assert meas.num_outcomes == 2


def test_computational_basis_measurement_on_basis_states():
    meas = QuantumMeasurement.computational_basis(dim=2)
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)

    p_zero = meas.outcome_probabilities(zero)
    p_one = meas.outcome_probabilities(one)

    assert np.allclose(p_zero, np.array([1.0, 0.0]))
    assert np.allclose(p_one, np.array([0.0, 1.0]))


def test_computational_basis_measurement_on_plus_state():
    meas = QuantumMeasurement.computational_basis(dim=2)
    plus = plus_state()

    probs = meas.outcome_probabilities(plus)
    assert np.allclose(probs, np.array([0.5, 0.5]))


def test_post_measurement_state_for_plus_state():
    meas = QuantumMeasurement.computational_basis(dim=2)
    plus = plus_state()

    post0 = meas.post_measurement_state(plus, 0)
    post1 = meas.post_measurement_state(plus, 1)

    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)

    assert np.allclose(post0.rho, zero.rho)
    assert np.allclose(post1.rho, one.rho)


def test_zero_probability_post_measurement_state_raises():
    meas = QuantumMeasurement.computational_basis(dim=2)
    zero = QuantumState.basis_state(0, dim=2)

    with pytest.raises(ValueError, match="zero-probability"):
        meas.post_measurement_state(zero, 1)


def test_invalid_measurement_operator_shape_raises():
    with pytest.raises(ValueError, match="square matrix"):
        QuantumMeasurement([np.array([1, 0], dtype=complex)])


def test_incomplete_measurement_raises():
    p0 = np.array([[1, 0], [0, 0]], dtype=complex)

    with pytest.raises(ValueError, match="sum to the identity"):
        QuantumMeasurement.from_projectors([p0])


def test_state_dimension_mismatch_raises():
    meas = QuantumMeasurement.computational_basis(dim=2)
    qutrit = QuantumState.basis_state(0, dim=3)

    with pytest.raises(ValueError, match="does not match"):
        meas.outcome_probabilities(qutrit)


def test_outcome_index_out_of_range_raises():
    meas = QuantumMeasurement.computational_basis(dim=2)
    zero = QuantumState.basis_state(0, dim=2)

    with pytest.raises(ValueError, match="out of range"):
        meas.outcome_probability(zero, 2)


def test_effects_of_computational_basis_measurement():
    meas = QuantumMeasurement.computational_basis(dim=2)
    effects = meas.effects()

    expected0 = np.array([[1, 0], [0, 0]], dtype=complex)
    expected1 = np.array([[0, 0], [0, 1]], dtype=complex)

    assert len(effects) == 2
    assert np.allclose(effects[0], expected0)
    assert np.allclose(effects[1], expected1)
