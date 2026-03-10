# tests/test_state_generators.py

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
    state = basis_state(1, dim=3)
    expected = np.zeros((3, 3), dtype=complex)
    expected[1, 1] = 1.0

    assert state.dims == (3,)
    assert np.allclose(state.rho, expected)
    assert state.is_pure is True
    assert state.rank == 1


def test_computational_basis_state_011():
    state = computational_basis_state((0, 1, 1))

    expected_ket = np.zeros(8, dtype=complex)
    expected_ket[3] = 1.0  # binary 011 = 3
    expected_rho = np.outer(expected_ket, expected_ket.conj())

    assert state.dims == (2, 2, 2)
    assert state.dimension == 8
    assert np.allclose(state.rho, expected_rho)
    assert state.is_pure is True


def test_computational_basis_state_rejects_empty_bits():
    with pytest.raises(ValueError, match="nonempty sequence"):
        computational_basis_state(())


def test_computational_basis_state_rejects_non_bits():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        computational_basis_state((0, 2, 1))


def test_plus_state_exact_density_matrix():
    state = plus_state()
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

    assert state.dims == (2,)
    assert np.allclose(state.rho, expected)
    assert state.is_pure is True
    assert np.isclose(state.purity(), 1.0)


def test_minus_state_exact_density_matrix():
    state = minus_state()
    expected = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)

    assert state.dims == (2,)
    assert np.allclose(state.rho, expected)
    assert state.is_pure is True
    assert np.isclose(state.purity(), 1.0)


@pytest.mark.parametrize(
    ("which", "ket"),
    [
        (0, np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)),
        (1, np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)),
        (2, np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)),
        (3, np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)),
    ],
)
def test_bell_state_exact_density_matrices(which, ket):
    state = bell_state(which)
    expected = np.outer(ket, ket.conj())

    assert state.dims == (2, 2)
    assert state.dimension == 4
    assert np.allclose(state.rho, expected)
    assert state.is_pure is True
    assert state.rank == 1


def test_bell_state_rejects_invalid_label():
    with pytest.raises(ValueError, match="one of 0, 1, 2, 3"):
        bell_state(4)


def test_ghz_state_n1_equals_plus_state():
    ghz1 = ghz_state(1)
    plus = plus_state()

    assert ghz1.dims == (2,)
    assert np.allclose(ghz1.rho, plus.rho)


def test_ghz_state_n3_exact_density_matrix():
    state = ghz_state(3)

    ket = np.zeros(8, dtype=complex)
    ket[0] = 1 / np.sqrt(2)
    ket[-1] = 1 / np.sqrt(2)
    expected = np.outer(ket, ket.conj())

    assert state.dims == (2, 2, 2)
    assert state.dimension == 8
    assert np.allclose(state.rho, expected)
    assert state.is_pure is True


def test_ghz_state_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="n must be positive"):
        ghz_state(0)


def test_w_state_n3_exact_support():
    state = w_state(3)

    ket = np.zeros(8, dtype=complex)
    ket[4] = 1.0
    ket[2] = 1.0
    ket[1] = 1.0
    ket /= np.linalg.norm(ket)
    expected = np.outer(ket, ket.conj())

    assert state.dims == (2, 2, 2)
    assert state.dimension == 8
    assert np.allclose(state.rho, expected)
    assert state.is_pure is True
    assert np.isclose(state.purity(), 1.0)


def test_w_state_n1_is_one_state():
    state = w_state(1)

    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert state.dims == (2,)
    assert np.allclose(state.rho, expected)


def test_w_state_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="n must be positive"):
        w_state(-1)


def test_maximally_mixed_state_exact_matrix():
    state = maximally_mixed_state(3)
    expected = np.eye(3, dtype=complex) / 3.0

    assert state.dims == (3,)
    assert np.allclose(state.rho, expected)
    assert state.is_mixed is True
    assert state.rank == 3
    assert np.isclose(state.purity(), 1 / 3)


def test_maximally_mixed_state_rejects_nonpositive_dim():
    with pytest.raises(ValueError, match="dim must be positive"):
        maximally_mixed_state(0)


def test_werner_state_p0_is_maximally_mixed_two_qubit():
    state = werner_state(0.0)
    expected = np.eye(4, dtype=complex) / 4.0

    assert state.dims == (2, 2)
    assert np.allclose(state.rho, expected)
    assert state.rank == 4
    assert state.is_mixed is True


def test_werner_state_p1_is_singlet_state():
    state = werner_state(1.0)
    singlet = bell_state(3)

    assert state.dims == (2, 2)
    assert np.allclose(state.rho, singlet.rho)
    assert state.rank == 1
    assert state.is_pure is True


def test_werner_state_interpolates_correctly():
    p = 0.25
    state = werner_state(p)

    singlet = bell_state(3).rho
    mixed = np.eye(4, dtype=complex) / 4.0
    expected = p * singlet + (1 - p) * mixed

    assert np.allclose(state.rho, expected)
    assert state.dims == (2, 2)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_werner_state_rejects_invalid_p(p):
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        werner_state(p)
