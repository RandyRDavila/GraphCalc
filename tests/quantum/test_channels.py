import numpy as np
import pytest

from graphcalc.quantum.channels import QuantumChannel
from graphcalc.quantum.generators import bell_state, maximally_mixed_state, plus_state
from graphcalc.quantum.states import QuantumState


def test_identity_channel_basic_properties():
    chan = QuantumChannel.identity(2)

    assert chan.input_dim == 2
    assert chan.output_dim == 2
    assert chan.dimension == 4
    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert chan.is_unital()
    assert chan.choi_rank == 1


def test_identity_channel_applies_correctly_to_pure_state():
    chan = QuantumChannel.identity(2)
    state = plus_state()
    out = chan.apply(state)

    assert out.dims == (2,)
    assert out.is_pure
    assert np.allclose(out.rho, state.rho)


def test_identity_channel_applies_correctly_to_mixed_state():
    chan = QuantumChannel.identity(2)
    state = maximally_mixed_state(2)
    out = chan.apply(state)

    assert out.dims == (2,)
    assert out.is_mixed
    assert np.allclose(out.rho, state.rho)


def test_from_kraus_builds_identity_channel():
    I = np.eye(2, dtype=complex)
    chan = QuantumChannel.from_kraus([I])

    assert chan.input_dim == 2
    assert chan.output_dim == 2
    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert chan.is_unital()
    assert chan.choi_rank == 1


def test_identity_channel_choi_has_expected_shape():
    chan = QuantumChannel.identity(3)

    assert chan.choi.shape == (9, 9)


def test_invalid_empty_kraus_list_raises():
    with pytest.raises(ValueError, match="nonempty"):
        QuantumChannel.from_kraus([])


def test_invalid_kraus_shape_raises():
    k1 = np.eye(2, dtype=complex)
    k2 = np.zeros((3, 2), dtype=complex)

    with pytest.raises(ValueError, match="All Kraus operators must have shape"):
        QuantumChannel.from_kraus([k1, k2], input_dim=2, output_dim=2)


def test_invalid_nonmatrix_kraus_operator_raises():
    with pytest.raises(ValueError, match="must be a matrix"):
        QuantumChannel.from_kraus([np.array([1, 0], dtype=complex)])


def test_invalid_input_dimension_raises():
    choi = np.eye(4, dtype=complex)

    with pytest.raises(ValueError, match="input_dim must be positive"):
        QuantumChannel.from_choi(choi, input_dim=0, output_dim=2, validate=False)


def test_invalid_output_dimension_raises():
    choi = np.eye(4, dtype=complex)

    with pytest.raises(ValueError, match="output_dim must be positive"):
        QuantumChannel.from_choi(choi, input_dim=2, output_dim=0, validate=False)


def test_invalid_choi_shape_raises():
    choi = np.eye(3, dtype=complex)

    with pytest.raises(ValueError, match="choi must have shape"):
        QuantumChannel.from_choi(choi, input_dim=2, output_dim=2, validate=False)


def test_nonhermitian_choi_raises():
    choi = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )

    with pytest.raises(ValueError, match="Hermitian"):
        QuantumChannel.from_choi(choi, input_dim=2, output_dim=2)


def test_non_positive_choi_raises():
    choi = np.diag([1.0, -0.1, 0.1, 1.0]).astype(complex)

    with pytest.raises(ValueError, match="positive semidefinite"):
        QuantumChannel.from_choi(choi, input_dim=2, output_dim=2)


def test_bad_partial_trace_raises():
    choi = np.eye(4, dtype=complex)

    with pytest.raises(ValueError, match="Partial trace over the output subsystem"):
        QuantumChannel.from_choi(choi, input_dim=2, output_dim=2)


def test_apply_dimension_mismatch_raises():
    chan = QuantumChannel.identity(2)
    state = bell_state(0)

    with pytest.raises(ValueError, match="does not match"):
        chan.apply(state)


def test_kraus_reconstruction_of_identity_preserves_channel_action():
    chan = QuantumChannel.identity(2)
    kraus = chan.kraus_operators()

    assert len(kraus) == 1

    # Kraus operators are not unique: global phase does not matter.
    # For a one-Kraus identity channel, the recovered operator should be unitary.
    assert np.allclose(kraus[0].conj().T @ kraus[0], np.eye(2))
    assert np.allclose(kraus[0] @ kraus[0].conj().T, np.eye(2))

    reconstructed = QuantumChannel.from_kraus(kraus)
    state = plus_state()
    out = reconstructed.apply(state)

    assert np.allclose(out.rho, state.rho)


def test_kraus_reconstruction_matches_original_channel_on_mixed_state():
    chan = QuantumChannel.identity(2)
    kraus = chan.kraus_operators()
    reconstructed = QuantumChannel.from_kraus(kraus)

    state = maximally_mixed_state(2)
    out_original = chan.apply(state)
    out_reconstructed = reconstructed.apply(state)

    assert np.allclose(out_reconstructed.rho, out_original.rho)


def test_compose_identity_with_identity():
    c1 = QuantumChannel.identity(2)
    c2 = QuantumChannel.identity(2)
    comp = c1.compose(c2)

    assert comp.input_dim == 2
    assert comp.output_dim == 2
    assert comp.is_completely_positive()
    assert comp.is_trace_preserving()
    assert comp.is_unital()

    state = maximally_mixed_state(2)
    out = comp.apply(state)
    assert np.allclose(out.rho, state.rho)


def test_compose_dimension_mismatch_raises():
    c1 = QuantumChannel.identity(2)
    c2 = QuantumChannel.identity(3)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        c1.compose(c2)


def test_tensor_of_identity_channels():
    c2 = QuantumChannel.identity(2)
    c3 = QuantumChannel.identity(3)
    tensor = c2.tensor(c3)

    assert tensor.input_dim == 6
    assert tensor.output_dim == 6
    assert tensor.dimension == 36
    assert tensor.is_completely_positive()
    assert tensor.is_trace_preserving()
    assert tensor.is_unital()


def test_tensor_applies_as_expected_on_product_basis_state():
    chan = QuantumChannel.identity(2).tensor(QuantumChannel.identity(2))

    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    out = chan.apply(state)
    assert out.dims == (4,)
    assert np.allclose(out.rho, state.rho)


def test_from_kraus_rectangular_channel_trace_preserving():
    k0 = np.array([[1, 0]], dtype=complex)
    k1 = np.array([[0, 1]], dtype=complex)

    chan = QuantumChannel.from_kraus([k0, k1])

    assert chan.input_dim == 2
    assert chan.output_dim == 1
    assert chan.is_completely_positive()
    assert chan.is_trace_preserving()
    assert not chan.is_unital()

    rho = np.array([[0.3, 0.1], [0.1, 0.7]], dtype=complex)
    state = QuantumState.from_density(rho, dims=(2,))
    out = chan.apply(state)

    assert out.dims == (1,)
    assert np.allclose(out.rho, np.array([[1.0 + 0.0j]]))


def test_rectangular_channel_maps_every_input_to_one_dimensional_output():
    k0 = np.array([[1, 0]], dtype=complex)
    k1 = np.array([[0, 1]], dtype=complex)
    chan = QuantumChannel.from_kraus([k0, k1])

    state0 = QuantumState.basis_state(0, dim=2)
    state1 = QuantumState.basis_state(1, dim=2)

    out0 = chan.apply(state0)
    out1 = chan.apply(state1)

    expected = np.array([[1.0 + 0.0j]])
    assert np.allclose(out0.rho, expected)
    assert np.allclose(out1.rho, expected)


def test_copy_is_independent():
    chan = QuantumChannel.identity(2)
    copied = chan.copy()

    assert copied is not chan
    assert copied.input_dim == chan.input_dim
    assert copied.output_dim == chan.output_dim
    assert np.allclose(copied.choi, chan.choi)


def test_from_choi_with_validate_false_allows_nonchannel_operator():
    choi = np.eye(4, dtype=complex)
    chan = QuantumChannel.from_choi(choi, input_dim=2, output_dim=2, validate=False)

    assert chan.input_dim == 2
    assert chan.output_dim == 2
    assert chan.choi.shape == (4, 4)
    assert chan.is_completely_positive()
    assert not chan.is_trace_preserving()


def test_identity_channel_on_three_dimensional_system():
    chan = QuantumChannel.identity(3)
    state = QuantumState.basis_state(2, dim=3)

    out = chan.apply(state)

    assert out.dims == (3,)
    assert np.allclose(out.rho, state.rho)
