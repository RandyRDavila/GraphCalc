import numpy as np
import pytest

from graphcalc.quantum.generators import (
    bell_state,
    ghz_state,
    maximally_mixed_state,
    plus_state,
    werner_state,
)
from graphcalc.quantum.invariants import (
    entanglement_entropy,
    fidelity,
    linear_entropy,
    logarithmic_negativity,
    mutual_information,
    negativity,
    purity,
    rank,
    von_neumann_entropy,
)
from graphcalc.quantum.states import QuantumState


def test_purity_and_rank_for_pure_state():
    state = plus_state()

    assert np.isclose(purity(state), 1.0)
    assert rank(state) == 1
    assert np.isclose(linear_entropy(state), 0.0)
    assert np.isclose(von_neumann_entropy(state), 0.0)


def test_purity_and_entropy_for_maximally_mixed_qubit():
    state = maximally_mixed_state(2)

    assert np.isclose(purity(state), 0.5)
    assert rank(state) == 2
    assert np.isclose(linear_entropy(state), 0.5)
    assert np.isclose(von_neumann_entropy(state), 1.0)


def test_von_neumann_entropy_base_e():
    state = maximally_mixed_state(2)
    expected = np.log(2.0)

    assert np.isclose(von_neumann_entropy(state, base=np.e), expected)


def test_von_neumann_entropy_invalid_base_raises():
    state = plus_state()

    with pytest.raises(ValueError, match="positive and not equal to 1"):
        von_neumann_entropy(state, base=0.0)

    with pytest.raises(ValueError, match="positive and not equal to 1"):
        von_neumann_entropy(state, base=1.0)


def test_negativity_of_product_state_is_zero():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    assert np.isclose(negativity(state, subsystems=[0]), 0.0)
    assert np.isclose(logarithmic_negativity(state, subsystems=[0]), 0.0)


def test_negativity_of_bell_state():
    state = bell_state(0)

    assert np.isclose(negativity(state, subsystems=[0]), 0.5)
    assert np.isclose(negativity(state, subsystems=[1]), 0.5)
    assert np.isclose(logarithmic_negativity(state, subsystems=[0]), 1.0)


def test_negativity_of_werner_state_endpoints():
    mixed = werner_state(0.0)
    singlet = werner_state(1.0)

    assert np.isclose(negativity(mixed, subsystems=[0]), 0.0)
    assert np.isclose(negativity(singlet, subsystems=[0]), 0.5)


def test_entropy_of_reduced_bell_state_is_one():
    bell = bell_state(0)
    reduced = bell.reduced_state([0])

    assert np.isclose(von_neumann_entropy(reduced), 1.0)
    assert np.isclose(linear_entropy(reduced), 0.5)


def test_partial_transpose_preserves_shape_and_dims():
    bell = bell_state(0)
    pt = bell.partial_transpose([0])

    assert pt.dims == (2, 2)
    assert pt.rho.shape == (4, 4)


def test_partial_transpose_invalid_subsystem_raises():
    bell = bell_state(0)

    with pytest.raises(ValueError, match="out of range"):
        bell.partial_transpose([2])


def test_negativity_of_three_qubit_product_state_is_zero():
    a = QuantumState.basis_state(0, dim=2)
    b = QuantumState.basis_state(0, dim=2)
    c = QuantumState.basis_state(1, dim=2)
    state = a.tensor(b).tensor(c)

    assert np.isclose(negativity(state, subsystems=[0]), 0.0)
    assert np.isclose(negativity(state, subsystems=[1, 2]), 0.0)


def test_entropy_of_classical_mixture():
    rho = np.array([[0.75, 0.0], [0.0, 0.25]], dtype=complex)
    state = QuantumState.from_density(rho, dims=(2,))

    expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert np.isclose(von_neumann_entropy(state), expected)


def test_fidelity_of_identical_states_is_one():
    state = plus_state()
    assert np.isclose(fidelity(state, state), 1.0)


def test_fidelity_of_orthogonal_basis_states_is_zero():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)

    assert np.isclose(fidelity(zero, one), 0.0)


def test_fidelity_of_plus_and_maximally_mixed_qubit():
    plus = plus_state()
    mixed = maximally_mixed_state(2)

    assert np.isclose(fidelity(plus, mixed), 0.5)


def test_fidelity_dimension_mismatch_raises():
    qubit = plus_state()
    bell = bell_state(0)

    with pytest.raises(ValueError, match="same total dimension"):
        fidelity(qubit, bell)


def test_entanglement_entropy_of_product_state_is_zero():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    assert np.isclose(entanglement_entropy(state, subsystems=[0]), 0.0)


def test_entanglement_entropy_of_bell_state_is_one():
    bell = bell_state(0)

    assert np.isclose(entanglement_entropy(bell, subsystems=[0]), 1.0)
    assert np.isclose(entanglement_entropy(bell, subsystems=[1]), 1.0)


def test_entanglement_entropy_of_ghz_state_is_one():
    ghz = ghz_state(3)

    assert np.isclose(entanglement_entropy(ghz, subsystems=[0]), 1.0)
    assert np.isclose(entanglement_entropy(ghz, subsystems=[1, 2]), 1.0)


def test_entanglement_entropy_for_mixed_state_raises():
    mixed = maximally_mixed_state(2)

    with pytest.raises(ValueError, match="only defined here for pure states"):
        entanglement_entropy(mixed, subsystems=[0])


def test_mutual_information_of_product_state_is_zero():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    assert np.isclose(mutual_information(state, subsystems_a=[0], subsystems_b=[1]), 0.0)


def test_mutual_information_of_bell_state_is_two_bits():
    bell = bell_state(0)

    assert np.isclose(mutual_information(bell, subsystems_a=[0], subsystems_b=[1]), 2.0)


def test_mutual_information_requires_disjoint_nonempty_subsystems():
    bell = bell_state(0)

    with pytest.raises(ValueError, match="nonempty"):
        mutual_information(bell, subsystems_a=[], subsystems_b=[1])

    with pytest.raises(ValueError, match="nonempty"):
        mutual_information(bell, subsystems_a=[0], subsystems_b=[])

    with pytest.raises(ValueError, match="disjoint"):
        mutual_information(bell, subsystems_a=[0], subsystems_b=[0])

    with pytest.raises(ValueError, match="out of range"):
        mutual_information(bell, subsystems_a=[0], subsystems_b=[2])
