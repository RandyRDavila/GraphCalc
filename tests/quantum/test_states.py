# tests/test_states.py

import numpy as np
import pytest

from graphcalc.quantum.states import QuantumState


def test_from_density_valid_qubit_basis_state():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    state = QuantumState.from_density(rho, dims=(2,))

    assert state.dims == (2,)
    assert state.num_subsystems == 1
    assert state.dimension == 2
    assert np.isclose(state.trace, 1.0)
    assert state.rank == 1
    assert state.is_pure is True
    assert state.is_mixed is False


def test_rho_property_returns_copy():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    state = QuantumState(rho, dims=(2,))

    returned = state.rho
    returned[0, 0] = 0.0

    assert np.allclose(state.rho, rho)


def test_from_ket_normalizes_by_default():
    ket = np.array([2.0, 0.0], dtype=complex)
    state = QuantumState.from_ket(ket, dims=(2,))

    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    assert np.allclose(state.rho, expected)
    assert state.is_pure is True
    assert np.isclose(state.purity(), 1.0)


def test_from_ket_without_normalization_can_fail_validation():
    ket = np.array([2.0, 0.0], dtype=complex)

    with pytest.raises(ValueError, match="trace 1"):
        QuantumState.from_ket(ket, dims=(2,), normalize=False)


def test_from_ket_rejects_wrong_length():
    ket = np.array([1.0, 0.0, 0.0], dtype=complex)

    with pytest.raises(ValueError, match="Ket has length"):
        QuantumState.from_ket(ket, dims=(2, 2))


def test_from_ket_rejects_zero_vector():
    ket = np.array([0.0, 0.0], dtype=complex)

    with pytest.raises(ValueError, match="must be nonzero"):
        QuantumState.from_ket(ket, dims=(2,))


def test_basis_state_construction():
    state = QuantumState.basis_state(1, dim=2)

    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(state.rho, expected)
    assert state.dims == (2,)
    assert state.is_pure is True


def test_basis_state_rejects_invalid_dim():
    with pytest.raises(ValueError, match="dim must be positive"):
        QuantumState.basis_state(0, dim=0)


def test_basis_state_rejects_invalid_index():
    with pytest.raises(ValueError, match="index must satisfy"):
        QuantumState.basis_state(2, dim=2)


def test_validate_rejects_empty_dims():
    rho = np.array([[1]], dtype=complex)

    with pytest.raises(ValueError, match="dims must be a nonempty tuple"):
        QuantumState(rho, dims=())


def test_validate_rejects_nonpositive_dims():
    rho = np.array([[1]], dtype=complex)

    with pytest.raises(ValueError, match="Every subsystem dimension must be positive"):
        QuantumState(rho, dims=(0,))


def test_validate_rejects_negative_tol():
    rho = np.array([[1]], dtype=complex)

    with pytest.raises(ValueError, match="tol must be nonnegative"):
        QuantumState(rho, dims=(1,), tol=-1e-9)


def test_validate_rejects_wrong_shape():
    rho = np.eye(2, dtype=complex)

    with pytest.raises(ValueError, match="rho must have shape"):
        QuantumState(rho, dims=(2, 2))


def test_validate_rejects_nonhermitian_matrix():
    rho = np.array([[0.5, 1.0], [0.0, 0.5]], dtype=complex)

    with pytest.raises(ValueError, match="Hermitian"):
        QuantumState(rho, dims=(2,))


def test_validate_rejects_wrong_trace():
    rho = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=complex)

    with pytest.raises(ValueError, match="trace 1"):
        QuantumState(rho, dims=(2,))


def test_validate_rejects_negative_eigenvalue():
    rho = np.array([[1.2, 0.0], [0.0, -0.2]], dtype=complex)

    with pytest.raises(ValueError, match="positive semidefinite"):
        QuantumState(rho, dims=(2,))


def test_eigenvalues_zero_out_tiny_values():
    rho = np.array([[1.0, 0.0], [0.0, 1e-12]], dtype=complex)
    rho = rho / np.trace(rho)
    state = QuantumState(rho, dims=(2,), tol=1e-9)

    evals = state.eigenvalues()
    assert np.isclose(evals[0], 0.0, atol=1e-9)
    assert np.isclose(evals[1], 1.0, atol=1e-9)


def test_rank_uses_tolerance():
    rho = np.diag([1.0, 1e-12]).astype(complex)
    rho /= np.trace(rho)
    state = QuantumState(rho, dims=(2,), tol=1e-9)

    assert state.rank == 1


def test_copy_returns_independent_state():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    state = QuantumState(rho, dims=(2,))
    state2 = state.copy()

    assert state2 is not state
    assert state2.dims == state.dims
    assert np.allclose(state2.rho, state.rho)


def test_tensor_product_of_basis_states():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)

    tensor_state = zero.tensor(one)

    expected_ket = np.array([0, 1, 0, 0], dtype=complex)  # |01>
    expected_rho = np.outer(expected_ket, expected_ket.conj())

    assert tensor_state.dims == (2, 2)
    assert tensor_state.dimension == 4
    assert np.allclose(tensor_state.rho, expected_rho)
    assert tensor_state.is_pure is True


def test_partial_trace_of_product_state():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    reduced0 = state.partial_trace([1])
    reduced1 = state.partial_trace([0])

    assert reduced0.dims == (2,)
    assert reduced1.dims == (2,)
    assert np.allclose(reduced0.rho, zero.rho)
    assert np.allclose(reduced1.rho, one.rho)


def test_partial_trace_over_all_subsystems_returns_scalar_state():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    reduced = state.partial_trace([0, 1])

    assert reduced.dims == (1,)
    assert reduced.dimension == 1
    assert np.allclose(reduced.rho, np.array([[1.0]], dtype=complex))


def test_partial_trace_rejects_out_of_range_index():
    state = QuantumState.basis_state(0, dim=2)

    with pytest.raises(ValueError, match="Subsystem index out of range"):
        state.partial_trace([1])


def test_reduced_state_matches_partial_trace_complement():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    reduced = state.reduced_state([0])
    expected = state.partial_trace([1])

    assert np.allclose(reduced.rho, expected.rho)
    assert reduced.dims == expected.dims


def test_reduced_state_rejects_out_of_range_index():
    state = QuantumState.basis_state(0, dim=2)

    with pytest.raises(ValueError, match="Subsystem index out of range"):
        state.reduced_state([3])


def test_partial_transpose_on_product_state_matches_full_transpose_of_factor():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    pt = state.partial_transpose([1])

    # |1><1| is unchanged by transpose, so the whole state is unchanged
    assert pt.dims == state.dims
    assert np.allclose(pt.rho, state.rho)


def test_partial_transpose_rejects_out_of_range_index():
    state = QuantumState.basis_state(0, dim=2)

    with pytest.raises(ValueError, match="Subsystem index out of range"):
        state.partial_transpose([2])


def test_subsystem_dimensions_returns_requested_dims():
    state = QuantumState.from_density(np.eye(6) / 6, dims=(2, 3))

    assert state.subsystem_dimensions([0]) == (2,)
    assert state.subsystem_dimensions([1]) == (3,)
    assert state.subsystem_dimensions([0, 1]) == (2, 3)


def test_subsystem_dimensions_rejects_out_of_range_index():
    state = QuantumState.from_density(np.eye(6) / 6, dims=(2, 3))

    with pytest.raises(ValueError, match="Subsystem index out of range"):
        state.subsystem_dimensions([2])


def test_repr_contains_useful_summary_fields():
    state = QuantumState.basis_state(0, dim=2)
    text = repr(state)

    assert "QuantumState" in text
    assert "num_subsystems=1" in text
    assert "dimension=2" in text
    assert "is_pure=True" in text
    assert "rank=1" in text


def test_partial_trace_of_empty_set_returns_same_state():
    state = QuantumState.from_density(np.eye(2) / 2, dims=(2,))
    reduced = state.partial_trace([])

    assert reduced.dims == state.dims
    assert np.allclose(reduced.rho, state.rho)
