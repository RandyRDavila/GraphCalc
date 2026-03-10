import numpy as np

from graphcalc.quantum.measurement_properties import (
    is_povm,
    is_projective_measurement,
    is_rank_one_measurement,
)
from graphcalc.quantum.measurements import QuantumMeasurement


def test_computational_basis_measurement_properties():
    meas = QuantumMeasurement.computational_basis(dim=2)

    assert is_povm(meas)
    assert is_projective_measurement(meas)
    assert is_rank_one_measurement(meas)


def test_qutrit_computational_basis_measurement_properties():
    meas = QuantumMeasurement.computational_basis(dim=3)

    assert is_povm(meas)
    assert is_projective_measurement(meas)
    assert is_rank_one_measurement(meas)


def test_two_outcome_trivial_projective_measurement_is_not_rank_one():
    I = np.eye(2, dtype=complex)
    zero = np.zeros((2, 2), dtype=complex)
    meas = QuantumMeasurement([I, zero])

    assert is_povm(meas)
    assert is_projective_measurement(meas)
    assert not is_rank_one_measurement(meas)


def test_nonprojective_but_valid_povm():
    e0 = np.array([[0.75, 0.0], [0.0, 0.25]], dtype=complex)
    e1 = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=complex)

    m0 = np.diag(np.sqrt(np.diag(e0))).astype(complex)
    m1 = np.diag(np.sqrt(np.diag(e1))).astype(complex)

    meas = QuantumMeasurement([m0, m1])

    assert is_povm(meas)
    assert not is_projective_measurement(meas)
    assert not is_rank_one_measurement(meas)


def test_rank_one_nonprojective_povm():
    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi1 = np.array([-0.5, np.sqrt(3.0) / 2.0], dtype=complex)
    psi2 = np.array([-0.5, -np.sqrt(3.0) / 2.0], dtype=complex)

    p0 = np.outer(psi0, psi0.conj())
    p1 = np.outer(psi1, psi1.conj())
    p2 = np.outer(psi2, psi2.conj())

    # Trine POVM effects: E_k = (2/3) |psi_k><psi_k|
    m0 = np.sqrt(2.0 / 3.0) * p0
    m1 = np.sqrt(2.0 / 3.0) * p1
    m2 = np.sqrt(2.0 / 3.0) * p2

    meas = QuantumMeasurement([m0, m1, m2])

    assert is_povm(meas)
    assert not is_projective_measurement(meas)
    assert is_rank_one_measurement(meas)
