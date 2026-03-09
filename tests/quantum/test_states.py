import numpy as np
import pytest

from graphcalc.quantum.states import QuantumState


def test_basis_state_basic_properties():
    rho = QuantumState.basis_state(0, dim=2)

    assert rho.dims == (2,)
    assert rho.dimension == 2
    assert rho.num_subsystems == 1
    assert rho.is_pure
    assert not rho.is_mixed
    assert rho.rank == 1
    assert np.allclose(rho.rho, np.array([[1, 0], [0, 0]], dtype=complex))


def test_from_ket_normalizes_by_default():
    psi = np.array([2.0, 0.0], dtype=complex)
    state = QuantumState.from_ket(psi, dims=(2,))

    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    assert np.allclose(state.rho, expected)


def test_invalid_ket_length_raises():
    with pytest.raises(ValueError, match="Ket has length"):
        QuantumState.from_ket([1, 0, 0], dims=(2,))


def test_zero_ket_raises():
    with pytest.raises(ValueError, match="nonzero"):
        QuantumState.from_ket([0, 0], dims=(2,))


def test_invalid_dims_raises():
    rho = np.eye(2, dtype=complex) / 2
    with pytest.raises(ValueError, match="nonempty"):
        QuantumState(rho, dims=())

    with pytest.raises(ValueError, match="positive"):
        QuantumState(rho, dims=(2, 0))


def test_nonhermitian_density_raises():
    rho = np.array([[1, 1], [0, 0]], dtype=complex)
    with pytest.raises(ValueError, match="Hermitian"):
        QuantumState(rho, dims=(2,))


def test_wrong_trace_raises():
    rho = np.eye(2, dtype=complex)
    with pytest.raises(ValueError, match="trace 1"):
        QuantumState(rho, dims=(2,))


def test_not_positive_semidefinite_raises():
    rho = np.array([[1.2, 0], [0, -0.2]], dtype=complex)
    with pytest.raises(ValueError, match="positive semidefinite"):
        QuantumState(rho, dims=(2,))


def test_maximally_mixed_qubit_is_mixed():
    rho = np.eye(2, dtype=complex) / 2
    state = QuantumState.from_density(rho, dims=(2,))

    assert not state.is_pure
    assert state.is_mixed
    assert state.rank == 2
    assert np.isclose(state.purity(), 0.5)


def test_tensor_product_of_basis_states():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)

    tensor = zero.tensor(one)

    expected_diag = np.zeros(4, dtype=complex)
    expected_diag[1] = 1.0
    expected = np.diag(expected_diag)

    assert tensor.dims == (2, 2)
    assert tensor.dimension == 4
    assert tensor.is_pure
    assert np.allclose(tensor.rho, expected)


def test_partial_trace_of_product_state():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    left = state.reduced_state([0])
    right = state.reduced_state([1])

    assert left.dims == (2,)
    assert right.dims == (2,)
    assert np.allclose(left.rho, zero.rho)
    assert np.allclose(right.rho, one.rho)


def test_partial_trace_of_bell_state_is_maximally_mixed():
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    bell = QuantumState.from_ket(psi, dims=(2, 2))

    left = bell.reduced_state([0])
    right = bell.reduced_state([1])
    expected = np.eye(2, dtype=complex) / 2

    assert np.allclose(left.rho, expected)
    assert np.allclose(right.rho, expected)
    assert left.is_mixed
    assert right.is_mixed


def test_partial_trace_all_subsystems_returns_scalar_state():
    rho = QuantumState.basis_state(0, dim=2)
    reduced = rho.partial_trace([0])

    assert reduced.dims == (1,)
    assert reduced.dimension == 1
    assert np.allclose(reduced.rho, np.array([[1.0 + 0.0j]]))


def test_copy_is_independent():
    state = QuantumState.basis_state(0, dim=2)
    copied = state.copy()

    assert copied is not state
    assert copied.dims == state.dims
    assert np.allclose(copied.rho, state.rho)
