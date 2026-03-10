import pytest

from graphcalc.metadata import build_module_registry, get_graphcalc_metadata
from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
from graphcalc.additive_combinatorics.generators import coset, subgroup_from_generators
from graphcalc.additive_combinatorics.properties import (
    contains_zero,
    has_small_doubling,
    is_aperiodic,
    is_coset,
    is_periodic,
    is_sidon,
    is_subgroup,
    is_sum_free,
    is_symmetric,
)
from graphcalc.additive_combinatorics.sets import AdditiveSet


def test_contains_zero():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (2,)], group=G)
    B = AdditiveSet([(1,), (2,)], group=G)

    assert contains_zero(A) is True
    assert contains_zero(B) is False


def test_is_symmetric_true():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (6,)], group=G)

    assert is_symmetric(A) is True


def test_is_symmetric_false():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (2,)], group=G)

    assert is_symmetric(A) is False


def test_is_subgroup_true_for_trivial_subgroup():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,)], group=G)

    assert is_subgroup(A) is True


def test_is_subgroup_true_for_nontrivial_subgroup():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    assert is_subgroup(H) is True


def test_is_subgroup_false_when_missing_zero():
    G = FiniteAbelianGroup((6,))
    A = AdditiveSet([(2,), (4,)], group=G)

    assert is_subgroup(A) is False


def test_is_subgroup_false_when_not_closed_under_subtraction():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,)], group=G)

    assert is_subgroup(A) is False


def test_is_coset_true_for_subgroup():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    assert is_coset(H) is True


def test_is_coset_true_for_proper_translate():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])
    C = coset(H, (1,))

    assert is_coset(C) is True


def test_is_coset_false_for_empty_set():
    G = FiniteAbelianGroup((6,))
    A = AdditiveSet([], group=G)

    assert is_coset(A) is False


def test_is_coset_false_for_noncoset_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert is_coset(A) is False


def test_is_sum_free_true():
    G = FiniteAbelianGroup((10,))
    A = AdditiveSet([(3,), (4,)], group=G)

    # 3+3=6, 3+4=7, 4+4=8; none lie in A
    assert is_sum_free(A) is True


def test_is_sum_free_false():
    G = FiniteAbelianGroup((10,))
    A = AdditiveSet([(1,), (2,), (3,)], group=G)

    # 1+2 = 3
    assert is_sum_free(A) is False


def test_is_sidon_true():
    G = FiniteAbelianGroup((11,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert is_sidon(A) is True


def test_is_sidon_false():
    G = FiniteAbelianGroup((8,))
    A = AdditiveSet([(0,), (1,), (2,), (3,)], group=G)

    # 0+3 = 1+2
    assert is_sidon(A) is False


def test_has_small_doubling_true():
    G = FiniteAbelianGroup((11,))
    A = AdditiveSet([(0,), (1,), (2,), (3,)], group=G)

    assert has_small_doubling(A, 2.0) is True


def test_has_small_doubling_false():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert has_small_doubling(A, 1.5) is False


def test_has_small_doubling_on_empty_set():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([], group=G)

    assert has_small_doubling(A, 0.0) is True
    assert has_small_doubling(A, 2.0) is True


def test_has_small_doubling_rejects_negative_K():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,)], group=G)

    with pytest.raises(ValueError, match="nonnegative"):
        has_small_doubling(A, -1)


def test_is_periodic_true():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    assert is_periodic(H) is True
    assert is_aperiodic(H) is False


def test_is_aperiodic_true():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    assert is_periodic(A) is False
    assert is_aperiodic(A) is True


def test_metadata_attached_to_each_public_predicate():
    functions = [
        contains_zero,
        is_symmetric,
        is_subgroup,
        is_coset,
        is_sum_free,
        is_sidon,
        has_small_doubling,
        is_periodic,
        is_aperiodic,
    ]

    for func in functions:
        meta = get_graphcalc_metadata(func)
        assert meta is not None
        assert meta["display_name"]
        assert meta["category"] == "additive combinatorics predicates"
        assert "definition" in meta


def test_module_registry_contains_all_public_predicates():
    import graphcalc.additive_combinatorics.properties as properties_module

    registry = build_module_registry(properties_module)

    expected = {
        "contains_zero",
        "is_symmetric",
        "is_subgroup",
        "is_coset",
        "is_sum_free",
        "is_sidon",
        "has_small_doubling",
        "is_periodic",
        "is_aperiodic",
    }

    assert expected.issubset(set(registry))
