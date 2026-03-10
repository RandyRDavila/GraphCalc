import numpy as np
import pytest

from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup


def test_init_stores_moduli_and_basic_properties():
    G = FiniteAbelianGroup((2, 3, 5))

    assert G.moduli == (2, 3, 5)
    assert G.rank == 3
    assert G.order == 30


def test_repr_contains_useful_summary():
    G = FiniteAbelianGroup((2, 3))
    text = repr(G)

    assert "FiniteAbelianGroup" in text
    assert "rank=2" in text
    assert "order=6" in text
    assert "moduli=(2, 3)" in text


def test_rejects_empty_moduli():
    with pytest.raises(ValueError, match="nonempty sequence"):
        FiniteAbelianGroup(())


@pytest.mark.parametrize("moduli", [(0,), (-1,), (2, 0), (3, -4)])
def test_rejects_nonpositive_moduli(moduli):
    with pytest.raises(ValueError, match="must be positive"):
        FiniteAbelianGroup(moduli)


def test_zero_element():
    G = FiniteAbelianGroup((4, 5))
    assert G.zero() == (0, 0)


def test_normalize_reduces_coordinates_modulo_moduli():
    G = FiniteAbelianGroup((4, 5))
    assert G.normalize((5, -1)) == (1, 4)


def test_normalize_rejects_wrong_length():
    G = FiniteAbelianGroup((2, 3))

    with pytest.raises(ValueError, match="length 2"):
        G.normalize((1,))


def test_add_computes_coordinatewise_modular_sum():
    G = FiniteAbelianGroup((4, 5))
    assert G.add((3, 4), (2, 3)) == (1, 2)


def test_add_rejects_wrong_length():
    G = FiniteAbelianGroup((4, 5))

    with pytest.raises(ValueError, match="length 2"):
        G.add((1, 2), (3,))


def test_neg_computes_inverse():
    G = FiniteAbelianGroup((4, 5))
    assert G.neg((1, 2)) == (3, 3)


def test_sub_computes_coordinatewise_modular_difference():
    G = FiniteAbelianGroup((4, 5))
    assert G.sub((1, 1), (3, 4)) == (2, 2)


def test_scalar_mul():
    G = FiniteAbelianGroup((6, 10))
    assert G.scalar_mul(3, (2, 7)) == (0, 1)


def test_equal_uses_canonical_reduction():
    G = FiniteAbelianGroup((4, 5))

    assert G.equal((5, -1), (1, 4)) is True
    assert G.equal((1, 4), (1, 3)) is False


def test_contains_checks_length_only():
    G = FiniteAbelianGroup((4, 5))

    assert G.contains((100, -7)) is True
    assert G.contains((1,)) is False
    assert G.contains((1, 2, 3)) is False


def test_elements_lists_all_elements_in_lexicographic_order():
    G = FiniteAbelianGroup((2, 3))
    elems = list(G.elements())

    assert elems == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]


def test_list_elements_matches_elements_iterator():
    G = FiniteAbelianGroup((2, 2))
    assert G.list_elements() == list(G.elements())


def test_number_of_elements_matches_order():
    G = FiniteAbelianGroup((2, 3, 2))
    elems = list(G.elements())

    assert len(elems) == G.order
    assert len(set(elems)) == G.order


def test_group_law_identity_and_inverse():
    G = FiniteAbelianGroup((7, 8))
    x = (3, 5)
    zero = G.zero()

    assert G.add(x, zero) == G.normalize(x)
    assert G.add(zero, x) == G.normalize(x)
    assert G.add(x, G.neg(x)) == zero


def test_group_law_associativity_on_sample():
    G = FiniteAbelianGroup((5, 6))
    x = (1, 2)
    y = (3, 4)
    z = (2, 5)

    left = G.add(G.add(x, y), z)
    right = G.add(x, G.add(y, z))

    assert left == right


def test_group_law_commutativity_on_sample():
    G = FiniteAbelianGroup((5, 6))
    x = (1, 2)
    y = (3, 4)

    assert G.add(x, y) == G.add(y, x)


def test_random_element_has_correct_shape_and_range():
    G = FiniteAbelianGroup((3, 4, 5))
    rng = np.random.default_rng(123)

    x = G.random_element(rng=rng)

    assert len(x) == 3
    assert 0 <= x[0] < 3
    assert 0 <= x[1] < 4
    assert 0 <= x[2] < 5


def test_random_element_is_reproducible_with_seeded_rng():
    G = FiniteAbelianGroup((10, 10))
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    assert G.random_element(rng=rng1) == G.random_element(rng=rng2)
