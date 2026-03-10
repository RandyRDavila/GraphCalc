import numpy as np
import pytest

from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
from graphcalc.additive_combinatorics.generators import (
    arithmetic_progression,
    coset,
    empty_set,
    interval,
    random_subset,
    singleton,
    subgroup_from_generators,
)
from graphcalc.additive_combinatorics.sets import AdditiveSet


def test_empty_set_returns_empty_additive_set():
    G = FiniteAbelianGroup((5,))
    A = empty_set(G)

    assert isinstance(A, AdditiveSet)
    assert A.group is G
    assert A.elements == ()
    assert A.size == 0


def test_singleton_returns_canonical_singleton():
    G = FiniteAbelianGroup((5,))
    A = singleton(G, (6,))

    assert A.group is G
    assert A.elements == ((1,),)
    assert A.size == 1


def test_singleton_rejects_wrong_length():
    G = FiniteAbelianGroup((5,))
    with pytest.raises(ValueError, match="length 1"):
        singleton(G, (1, 2))


def test_interval_basic_cyclic_interval():
    A = interval(7, 4, start=2)

    assert A.group.moduli == (7,)
    assert A.elements == ((2,), (3,), (4,), (5,))
    assert A.size == 4


def test_interval_with_wraparound():
    A = interval(5, 4, start=3)

    assert A.elements == ((0,), (1,), (3,), (4,))
    assert A.size == 4


def test_interval_length_zero():
    A = interval(5, 0)

    assert A.elements == ()
    assert A.size == 0


def test_interval_length_larger_than_modulus_gives_whole_group():
    A = interval(5, 8, start=0)

    assert A.elements == ((0,), (1,), (2,), (3,), (4,))
    assert A.size == 5


def test_interval_rejects_nonpositive_modulus():
    with pytest.raises(ValueError, match="modulus must be positive"):
        interval(0, 3)


def test_interval_rejects_negative_length():
    with pytest.raises(ValueError, match="length must be nonnegative"):
        interval(5, -1)


def test_arithmetic_progression_in_cyclic_group():
    G = FiniteAbelianGroup((7,))
    A = arithmetic_progression(G, start=(1,), step=(2,), length=4)

    assert A.elements == ((0,), (1,), (3,), (5,))
    assert A.size == 4


def test_arithmetic_progression_in_product_group():
    G = FiniteAbelianGroup((4, 5))
    A = arithmetic_progression(G, start=(1, 1), step=(1, 2), length=3)

    assert A.elements == ((1, 1), (2, 3), (3, 0))
    assert A.size == 3


def test_arithmetic_progression_length_zero():
    G = FiniteAbelianGroup((7,))
    A = arithmetic_progression(G, start=(1,), step=(2,), length=0)

    assert A.elements == ()
    assert A.size == 0


def test_arithmetic_progression_can_collapse_in_torsion():
    G = FiniteAbelianGroup((4,))
    A = arithmetic_progression(G, start=(0,), step=(2,), length=4)

    assert A.elements == ((0,), (2,))
    assert A.size == 2


def test_arithmetic_progression_rejects_negative_length():
    G = FiniteAbelianGroup((7,))
    with pytest.raises(ValueError, match="length must be nonnegative"):
        arithmetic_progression(G, start=(0,), step=(1,), length=-1)


def test_arithmetic_progression_rejects_wrong_start_shape():
    G = FiniteAbelianGroup((7,))
    with pytest.raises(ValueError, match="length 1"):
        arithmetic_progression(G, start=(0, 1), step=(1,), length=3)


def test_arithmetic_progression_rejects_wrong_step_shape():
    G = FiniteAbelianGroup((7,))
    with pytest.raises(ValueError, match="length 1"):
        arithmetic_progression(G, start=(0,), step=(1, 2), length=3)


def test_random_subset_by_size_has_requested_cardinality():
    G = FiniteAbelianGroup((10,))
    rng = np.random.default_rng(123)

    A = random_subset(G, size=4, rng=rng)

    assert A.size == 4
    assert A.is_subset_of(AdditiveSet(G.elements(), group=G))


def test_random_subset_by_size_zero():
    G = FiniteAbelianGroup((10,))
    rng = np.random.default_rng(123)

    A = random_subset(G, size=0, rng=rng)

    assert A.size == 0
    assert A.elements == ()


def test_random_subset_by_density_zero_is_empty():
    G = FiniteAbelianGroup((10,))
    rng = np.random.default_rng(123)

    A = random_subset(G, density=0.0, rng=rng)

    assert A.elements == ()
    assert A.size == 0


def test_random_subset_by_density_one_is_whole_group():
    G = FiniteAbelianGroup((6,))
    rng = np.random.default_rng(123)

    A = random_subset(G, density=1.0, rng=rng)

    assert A.elements == tuple(G.elements())
    assert A.size == G.order


def test_random_subset_by_size_is_reproducible_with_seeded_rng():
    G = FiniteAbelianGroup((8,))
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    A1 = random_subset(G, size=3, rng=rng1)
    A2 = random_subset(G, size=3, rng=rng2)

    assert A1.equal(A2)


def test_random_subset_by_density_is_reproducible_with_seeded_rng():
    G = FiniteAbelianGroup((8,))
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    A1 = random_subset(G, density=0.4, rng=rng1)
    A2 = random_subset(G, density=0.4, rng=rng2)

    assert A1.equal(A2)


def test_random_subset_requires_exactly_one_of_size_and_density():
    G = FiniteAbelianGroup((5,))

    with pytest.raises(ValueError, match="Exactly one"):
        random_subset(G)

    with pytest.raises(ValueError, match="Exactly one"):
        random_subset(G, size=2, density=0.5)


def test_random_subset_rejects_invalid_size():
    G = FiniteAbelianGroup((5,))

    with pytest.raises(ValueError, match=r"0 <= size <= \|G\|"):
        random_subset(G, size=-1)

    with pytest.raises(ValueError, match=r"0 <= size <= \|G\|"):
        random_subset(G, size=6)


def test_random_subset_rejects_invalid_density():
    G = FiniteAbelianGroup((5,))

    with pytest.raises(ValueError, match="0 <= density <= 1"):
        random_subset(G, density=-0.1)

    with pytest.raises(ValueError, match="0 <= density <= 1"):
        random_subset(G, density=1.1)


def test_subgroup_from_empty_generators_is_trivial_subgroup():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [])

    assert H.elements == ((0,),)
    assert H.size == 1


def test_subgroup_from_generator_in_cyclic_group():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    assert H.elements == ((0,), (2,), (4,))
    assert H.size == 3


def test_subgroup_from_generators_in_product_group():
    G = FiniteAbelianGroup((2, 4))
    H = subgroup_from_generators(G, [(1, 0), (0, 2)])

    assert H.elements == ((0, 0), (0, 2), (1, 0), (1, 2))
    assert H.size == 4


def test_subgroup_from_generators_normalizes_inputs():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(8,)])

    assert H.elements == ((0,), (2,), (4,))


def test_subgroup_from_generators_rejects_wrong_length():
    G = FiniteAbelianGroup((6,))
    with pytest.raises(ValueError, match="length 1"):
        subgroup_from_generators(G, [(1, 2)])


def test_coset_translates_subgroup():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])
    C = coset(H, (1,))

    assert C.elements == ((1,), (3,), (5,))
    assert C.size == 3


def test_coset_rejects_wrong_length():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    with pytest.raises(ValueError, match="length 1"):
        coset(H, (1, 2))
