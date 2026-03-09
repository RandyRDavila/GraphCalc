import pytest

from graphcalc.hypergraphs.core.basics import Hypergraph
from graphcalc.hypergraphs.invariants.codegree import (
    average_codegree,
    codegree,
    lower_shadow,
    lower_shadow_size,
    maximum_codegree,
    minimum_codegree,
    upper_shadow,
    upper_shadow_size,
)


def test_codegree_basic():
    H = Hypergraph(E=[{1, 2, 3}, {2, 3, 4}])
    assert codegree(H, [2, 3]) == 2
    assert codegree(H, [1, 4]) == 0
    assert codegree(H, []) == 2


def test_maximum_minimum_average_codegree_t2():
    H = Hypergraph(E=[{1, 2, 3}, {2, 3, 4}])
    assert maximum_codegree(H, 2) == 2
    assert minimum_codegree(H, 2) == 0
    assert average_codegree(H, 2) == pytest.approx(1.0)


def test_codegree_t_zero():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    assert maximum_codegree(H, 0) == 2
    assert minimum_codegree(H, 0) == 2
    assert average_codegree(H, 0) == pytest.approx(2.0)


def test_codegree_t_too_large():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    assert maximum_codegree(H, 5) == 0
    assert minimum_codegree(H, 5) == 0
    assert average_codegree(H, 5) == 0.0


def test_codegree_negative_t_raises():
    H = Hypergraph(E=[{1, 2}])
    with pytest.raises(ValueError, match="t must be >= 0"):
        maximum_codegree(H, -1)
    with pytest.raises(ValueError, match="t must be >= 0"):
        minimum_codegree(H, -1)
    with pytest.raises(ValueError, match="t must be >= 0"):
        average_codegree(H, -1)


def test_lower_shadow():
    H = Hypergraph(E=[{1, 2, 3}, {2, 3, 4}])
    assert lower_shadow(H) == {
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
        frozenset({2, 4}),
        frozenset({3, 4}),
    }
    assert lower_shadow_size(H) == 5


def test_upper_shadow_default_ground_set():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    assert upper_shadow(H) == {
        frozenset({1, 2, 3}),
    }
    assert upper_shadow_size(H) == 1


def test_upper_shadow_custom_ground_set():
    H = Hypergraph(E=[{1, 2}])
    assert upper_shadow(H, ground_set=[1, 2, 3, 4]) == {
        frozenset({1, 2, 3}),
        frozenset({1, 2, 4}),
    }
    assert upper_shadow_size(H, ground_set=[1, 2, 3, 4]) == 2
