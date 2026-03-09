import numpy as np
import pytest

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


def test_basis_state_matches_core_constructor():
    state = basis_state(1, dim=2)
    expected = np.array([[0, 0], [0, 1]], dtype=complex)

    assert state.dims == (2,)
    assert state.is_pure
    assert np.allclose(state.rho, expected)


def test_computational_basis_state_for_bits_011():
    state = computational_basis_state((0, 1, 1))

    expected_diag = np.zeros(8, dtype=complex)
    expected_diag[3] = 1.0
    expected = np.diag(expected_diag)

    assert state.dims == (2, 2, 2)
    assert state.is_pure
    assert np.allclose(state.rho, expected)


def test_computational_basis_state_rejects_empty_bits():
    with pytest.raises(ValueError, match="nonempty"):
        computational_basis_state(())


def test_computational_basis_state_rejects_nonbits():
    with pytest.raises(ValueError, match="0 or 1"):
        computational_basis_state((0, 2, 1))


def test_plus_state_density_matrix():
    state = plus_state()
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

    assert state.dims == (2,)
    assert state.is_pure
    assert np.allclose(state.rho, expected)


def test_minus_state_density_matrix():
    state = minus_state()
    expected = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)

    assert state.dims == (2,)
    assert state.is_pure
    assert np.allclose(state.rho, expected)


@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_all_bell_states_are_pure_two_qubit_states(which):
    state = bell_state(which)

    assert state.dims == (2, 2)
    assert state.dimension == 4
    assert state.is_pure
    assert state.rank == 1
    assert np.isclose(state.purity(), 1.0)


def test_bell_state_zero_matches_expected_density():
    state = bell_state(0)
    ket = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    expected = np.outer(ket, ket.conj())

    assert np.allclose(state.rho, expected)


def test_bell_state_invalid_index_raises():
    with pytest.raises(ValueError, match="0, 1, 2, 3"):
        bell_state(4)


def test_ghz_state_three_qubits():
    state = ghz_state(3)

    ket = np.zeros(8, dtype=complex)
    ket[0] = 1 / np.sqrt(2)
    ket[-1] = 1 / np.sqrt(2)
    expected = np.outer(ket, ket.conj())

    assert state.dims == (2, 2, 2)
    assert state.is_pure
    assert np.allclose(state.rho, expected)


def test_ghz_state_one_qubit_is_plus_state():
    ghz1 = ghz_state(1)
    plus = plus_state()

    assert np.allclose(ghz1.rho, plus.rho)


def test_ghz_state_invalid_n_raises():
    with pytest.raises(ValueError, match="positive"):
        ghz_state(0)


def test_w_state_three_qubits():
    state = w_state(3)

    ket = np.zeros(8, dtype=complex)
    ket[1] = 1 / np.sqrt(3)
    ket[2] = 1 / np.sqrt(3)
    ket[4] = 1 / np.sqrt(3)
    expected = np.outer(ket, ket.conj())

    assert state.dims == (2, 2, 2)
    assert state.is_pure
    assert np.allclose(state.rho, expected)


def test_w_state_one_qubit_is_one_state():
    state = w_state(1)
    expected = np.array([[0, 0], [0, 1]], dtype=complex)

    assert state.dims == (2,)
    assert state.is_pure
    assert np.allclose(state.rho, expected)


def test_w_state_invalid_n_raises():
    with pytest.raises(ValueError, match="positive"):
        w_state(0)


def test_maximally_mixed_state_qubit():
    state = maximally_mixed_state(2)
    expected = np.eye(2, dtype=complex) / 2

    assert state.dims == (2,)
    assert state.is_mixed
    assert state.rank == 2
    assert np.isclose(state.purity(), 0.5)
    assert np.allclose(state.rho, expected)


def test_maximally_mixed_state_invalid_dim_raises():
    with pytest.raises(ValueError, match="positive"):
        maximally_mixed_state(0)


def test_werner_state_endpoints():
    rho0 = werner_state(0.0)
    rho1 = werner_state(1.0)

    expected0 = np.eye(4, dtype=complex) / 4.0
    expected1 = bell_state(3).rho

    assert rho0.dims == (2, 2)
    assert rho1.dims == (2, 2)
    assert np.allclose(rho0.rho, expected0)
    assert np.allclose(rho1.rho, expected1)


def test_werner_state_is_valid_mixed_state_for_interior_p():
    state = werner_state(0.5)

    assert state.dims == (2, 2)
    assert state.is_mixed
    assert np.isclose(state.trace.real, 1.0)
    assert np.all(state.eigenvalues() >= -state.tol)


def test_werner_state_invalid_p_raises():
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        werner_state(-0.1)

    with pytest.raises(ValueError, match="0 <= p <= 1"):
        werner_state(1.1)
