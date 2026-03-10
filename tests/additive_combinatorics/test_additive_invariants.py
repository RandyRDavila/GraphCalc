import math

import pytest

from graphcalc.metadata import build_module_registry, get_graphcalc_metadata
from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
from graphcalc.additive_combinatorics.generators import arithmetic_progression, subgroup_from_generators
from graphcalc.additive_combinatorics.invariants import (
    additive_energy,
    cardinality,
    diffset_size,
    difference_constant,
    doubling_constant,
    max_difference_representation_count,
    max_sum_representation_count,
    stabilizer_size,
    stabilizer_size_of_sumset,
    sumset_size,
    tripling_constant,
)
from graphcalc.additive_combinatorics.sets import AdditiveSet


def test_cardinality():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert cardinality(A) == 3


def test_sumset_size_for_small_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert sumset_size(A) == 6


def test_diffset_size_for_small_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert diffset_size(A) == 7


def test_doubling_constant_for_small_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert math.isclose(doubling_constant(A), 2.0)


def test_difference_constant_for_small_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert math.isclose(difference_constant(A), 7 / 3)


def test_tripling_constant_for_small_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    # 3A = whole group in Z/7Z
    assert math.isclose(tripling_constant(A), 7 / 3)


def test_additive_energy_for_two_point_set():
    G = FiniteAbelianGroup((10,))
    A = AdditiveSet([(0,), (1,)], group=G)

    # representation counts of A+A: 1,2,1
    assert additive_energy(A) == 1**2 + 2**2 + 1**2


def test_additive_energy_of_empty_set_is_zero():
    G = FiniteAbelianGroup((10,))
    A = AdditiveSet([], group=G)

    assert additive_energy(A) == 0


def test_max_sum_representation_count():
    G = FiniteAbelianGroup((10,))
    A = AdditiveSet([(0,), (1,)], group=G)

    assert max_sum_representation_count(A) == 2


def test_max_difference_representation_count():
    G = FiniteAbelianGroup((10,))
    A = AdditiveSet([(0,), (1,)], group=G)

    # differences: 0 occurs twice, 1 and -1 each once
    assert max_difference_representation_count(A) == 2


def test_stabilizer_size_of_singleton():
    G = FiniteAbelianGroup((9,))
    A = AdditiveSet([(3,)], group=G)

    assert stabilizer_size(A) == 1


def test_stabilizer_size_of_subgroup():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    assert stabilizer_size(H) == 3


def test_stabilizer_size_of_sumset_for_aperiodic_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert stabilizer_size_of_sumset(A) == 1


def test_stabilizer_size_of_sumset_for_subgroup():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    assert stabilizer_size_of_sumset(H) == 3


def test_doubling_constant_rejects_empty_set():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([], group=G)

    with pytest.raises(ValueError, match="undefined for the empty set"):
        doubling_constant(A)


def test_difference_constant_rejects_empty_set():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([], group=G)

    with pytest.raises(ValueError, match="undefined for the empty set"):
        difference_constant(A)


def test_tripling_constant_rejects_empty_set():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([], group=G)

    with pytest.raises(ValueError, match="undefined for the empty set"):
        tripling_constant(A)


def test_interval_progression_has_expected_doubling():
    G = FiniteAbelianGroup((11,))
    A = arithmetic_progression(G, start=(0,), step=(1,), length=4)

    assert cardinality(A) == 4
    assert sumset_size(A) == 7
    assert math.isclose(doubling_constant(A), 7 / 4)


def test_metadata_attached_to_each_public_invariant():
    functions = [
        cardinality,
        sumset_size,
        diffset_size,
        doubling_constant,
        difference_constant,
        tripling_constant,
        additive_energy,
        max_sum_representation_count,
        max_difference_representation_count,
        stabilizer_size,
        stabilizer_size_of_sumset,
    ]

    for func in functions:
        meta = get_graphcalc_metadata(func)
        assert meta is not None
        assert meta["display_name"]
        assert meta["category"] == "additive combinatorics invariants"
        assert "definition" in meta


def test_module_registry_contains_all_public_invariants():
    import graphcalc.additive_combinatorics.invariants as invariants_module

    registry = build_module_registry(invariants_module)

    expected = {
        "cardinality",
        "sumset_size",
        "diffset_size",
        "doubling_constant",
        "difference_constant",
        "tripling_constant",
        "additive_energy",
        "max_sum_representation_count",
        "max_difference_representation_count",
        "stabilizer_size",
        "stabilizer_size_of_sumset",
    }

    assert expected.issubset(set(registry))
