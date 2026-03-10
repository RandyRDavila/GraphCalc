import pytest

from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
from graphcalc.additive_combinatorics.sets import AdditiveSet


def test_init_canonicalizes_and_deduplicates_elements():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,), (6,), (-4,)], group=G)

    assert A.elements == ((0,), (1,))
    assert A.size == 2
    assert len(A) == 2


def test_empty_set_is_allowed():
    G = FiniteAbelianGroup((4,))
    A = AdditiveSet([], group=G)

    assert A.elements == ()
    assert A.size == 0
    assert A.is_empty is True


def test_rejects_invalid_group_type():
    with pytest.raises(TypeError, match="FiniteAbelianGroup"):
        AdditiveSet([(0,)], group="not a group")  # type: ignore[arg-type]


def test_repr_contains_size_and_group():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)
    text = repr(A)

    assert "AdditiveSet" in text
    assert "size=2" in text
    assert "FiniteAbelianGroup" in text


def test_copy_returns_equal_but_distinct_object():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)
    B = A.copy()

    assert B is not A
    assert B.equal(A)


def test_contains_uses_canonical_group_reduction():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(1,), (2,)], group=G)

    assert (6,) in A
    assert (2,) in A
    assert (3,) not in A


def test_contains_rejects_wrong_length():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(1,), (2,)], group=G)

    with pytest.raises(ValueError, match="length 1"):
        (1, 2) in A


def test_iteration_matches_elements():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(2,), (0,), (1,)], group=G)

    assert tuple(iter(A)) == A.elements


def test_equal_requires_same_elements_and_same_moduli():
    G1 = FiniteAbelianGroup((5,))
    G2 = FiniteAbelianGroup((6,))

    A = AdditiveSet([(0,), (1,)], group=G1)
    B = AdditiveSet([(1,), (0,)], group=G1)
    C = AdditiveSet([(0,), (2,)], group=G1)
    D = AdditiveSet([(0,), (1,)], group=G2)

    assert A.equal(B) is True
    assert A.equal(C) is False
    assert A.equal(D) is False


def test_contains_zero_detects_identity():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (2,)], group=G)
    B = AdditiveSet([(1,), (2,)], group=G)

    assert A.contains_zero() is True
    assert B.contains_zero() is False


def test_negate_reflects_set():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(1,), (2,)], group=G)

    negA = A.negate()

    assert negA.elements == ((3,), (4,))


def test_translate_shifts_all_elements():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (2,)], group=G)

    B = A.translate((3,))

    assert B.elements == ((0,), (3,))


def test_translate_rejects_wrong_length():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (2,)], group=G)

    with pytest.raises(ValueError, match="length 1"):
        A.translate((1, 2))


def test_dilate_multiplies_all_elements():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(1,), (3,)], group=G)

    B = A.dilate(2)

    assert B.elements == ((2,), (6,))


def test_self_sumset_in_cyclic_group():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    AA = A.sumset()

    assert AA.elements == ((0,), (1,), (2,), (3,), (4,))


def test_binary_sumset():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,)], group=G)
    B = AdditiveSet([(0,), (2,)], group=G)

    AB = A.sumset(B)

    assert AB.elements == ((0,), (1,), (2,), (3,))


def test_sumset_rejects_mismatched_groups():
    A = AdditiveSet([(0,), (1,)], group=FiniteAbelianGroup((5,)))
    B = AdditiveSet([(0,), (1,)], group=FiniteAbelianGroup((6,)))

    with pytest.raises(ValueError, match="same ambient group"):
        A.sumset(B)


def test_self_diffset():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    diff = A.diffset()

    assert diff.elements == ((0,), (1,), (2,), (3,), (4,))


def test_binary_diffset():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (3,)], group=G)
    B = AdditiveSet([(0,), (2,)], group=G)

    diff = A.diffset(B)

    assert diff.elements == ((0,), (1,), (3,), (5,))


def test_diffset_rejects_mismatched_groups():
    A = AdditiveSet([(0,), (1,)], group=FiniteAbelianGroup((5,)))
    B = AdditiveSet([(0,), (1,)], group=FiniteAbelianGroup((6,)))

    with pytest.raises(ValueError, match="same ambient group"):
        A.diffset(B)


def test_k_fold_sum_zero_is_zero_set():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(1,), (3,)], group=G)

    zeroA = A.k_fold_sum(0)

    assert zeroA.elements == ((0,),)


def test_k_fold_sum_one_is_self():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(1,), (3,)], group=G)

    assert A.k_fold_sum(1).equal(A)


def test_k_fold_sum_two_matches_sumset():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(1,), (3,)], group=G)

    assert A.k_fold_sum(2).equal(A.sumset())


def test_k_fold_sum_rejects_negative_k():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(1,), (3,)], group=G)

    with pytest.raises(ValueError, match="nonnegative"):
        A.k_fold_sum(-1)


def test_representation_function_for_self_sumset():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)

    reps = A.representation_function()

    assert reps == {
        (0,): 1,  # 0+0
        (1,): 2,  # 0+1, 1+0
        (2,): 1,  # 1+1
    }


def test_representation_function_for_distinct_sets():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,)], group=G)
    B = AdditiveSet([(0,), (2,)], group=G)

    reps = A.representation_function(B)

    assert reps == {
        (0,): 1,
        (1,): 1,
        (2,): 1,
        (3,): 1,
    }


def test_representation_function_rejects_mismatched_groups():
    A = AdditiveSet([(0,), (1,)], group=FiniteAbelianGroup((5,)))
    B = AdditiveSet([(0,), (1,)], group=FiniteAbelianGroup((6,)))

    with pytest.raises(ValueError, match="same ambient group"):
        A.representation_function(B)


def test_max_sum_representation_count():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)

    assert A.max_sum_representation_count() == 2


def test_max_sum_representation_count_of_empty_sumset_is_zero():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([], group=G)

    assert A.max_sum_representation_count() == 0


def test_is_subset_of():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)
    B = AdditiveSet([(0,), (1,), (2,)], group=G)
    C = AdditiveSet([(0,), (3,)], group=G)

    assert A.is_subset_of(B) is True
    assert A.is_subset_of(C) is False


def test_is_subset_of_false_for_different_groups():
    A = AdditiveSet([(0,), (1,)], group=FiniteAbelianGroup((5,)))
    B = AdditiveSet([(0,), (1,), (2,)], group=FiniteAbelianGroup((6,)))

    assert A.is_subset_of(B) is False


def test_stabilizer_of_singleton_is_trivial():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,)], group=G)

    stab = A.stabilizer()

    assert stab.elements == ((0,),)


def test_stabilizer_of_whole_group_is_whole_group():
    G = FiniteAbelianGroup((4,))
    A = AdditiveSet(G.elements(), group=G)

    stab = A.stabilizer()

    assert stab.equal(A)


def test_stabilizer_of_even_subgroup_in_z4():
    G = FiniteAbelianGroup((4,))
    A = AdditiveSet([(0,), (2,)], group=G)

    stab = A.stabilizer()

    assert stab.elements == ((0,), (2,))
