import numpy as np
import pytest

from graphcalc.quantum.generators import (
    bell_state,
    ghz_state,
    maximally_mixed_state,
    plus_state,
    werner_state,
)
from graphcalc.quantum.properties import (
    has_positive_partial_transpose,
    is_entangled,
    is_mixed,
    is_product_state,
    is_pure,
    is_valid_state,
)
from graphcalc.quantum.states import QuantumState


def test_is_valid_state_true_for_standard_states():
    assert is_valid_state(plus_state())
    assert is_valid_state(bell_state(0))
    assert is_valid_state(maximally_mixed_state(2))


def test_is_valid_state_false_for_partial_transpose_of_bell_state():
    bell = bell_state(0)
    pt = bell.partial_transpose([0])

    assert not is_valid_state(pt)


def test_is_pure_and_is_mixed_match_basic_examples():
    pure = plus_state()
    mixed = maximally_mixed_state(2)

    assert is_pure(pure)
    assert not is_mixed(pure)

    assert not is_pure(mixed)
    assert is_mixed(mixed)


def test_has_positive_partial_transpose_for_product_state():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    product = zero.tensor(one)

    assert has_positive_partial_transpose(product, subsystems=[0])
    assert has_positive_partial_transpose(product, subsystems=[1])


def test_bell_state_fails_ppt():
    bell = bell_state(0)

    assert not has_positive_partial_transpose(bell, subsystems=[0])
    assert not has_positive_partial_transpose(bell, subsystems=[1])


def test_maximally_mixed_state_is_product_for_single_system():
    state = maximally_mixed_state(2)
    assert is_product_state(state)
    assert not is_entangled(state)


def test_two_qubit_product_state_is_product():
    zero = QuantumState.basis_state(0, dim=2)
    one = QuantumState.basis_state(1, dim=2)
    state = zero.tensor(one)

    assert is_product_state(state)
    assert not is_entangled(state)


def test_bell_state_is_entangled():
    bell = bell_state(0)

    assert not is_product_state(bell)
    assert is_entangled(bell)


def test_ghz_state_is_entangled_across_every_nontrivial_bipartition():
    ghz = ghz_state(3)

    assert not is_product_state(ghz)
    assert is_entangled(ghz)


def test_explicit_partition_for_three_qubit_product_state():
    left = bell_state(0)
    right = QuantumState.basis_state(0, dim=2)
    state = left.tensor(right)

    assert is_product_state(state, partition=[0, 1])
    assert not is_entangled(state, partition=[0, 1])

    assert not is_product_state(state, partition=[0])
    assert is_entangled(state, partition=[0])


def test_partition_must_be_nontrivial():
    bell = bell_state(0)

    with pytest.raises(ValueError, match="nonempty proper subset"):
        is_product_state(bell, partition=[])

    with pytest.raises(ValueError, match="nonempty proper subset"):
        is_product_state(bell, partition=[0, 1])


def test_partition_out_of_range_raises():
    bell = bell_state(0)

    with pytest.raises(ValueError, match="out of range"):
        is_product_state(bell, partition=[2])

    with pytest.raises(ValueError, match="out of range"):
        has_positive_partial_transpose(bell, subsystems=[2])


def test_werner_state_ppt_behavior_at_simple_examples():
    mixed = werner_state(0.0)
    singlet = werner_state(1.0)

    assert has_positive_partial_transpose(mixed, subsystems=[0])
    assert not has_positive_partial_transpose(singlet, subsystems=[0])


def test_is_valid_state_false_for_bad_trace():
    rho = np.array([[0.6, 0.0], [0.0, 0.6]], dtype=complex)
    state = QuantumState.from_density(rho, dims=(2,), validate=False)

    assert not is_valid_state(state)
