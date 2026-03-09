from graphcalc.hypergraphs.core.basics import Hypergraph
from graphcalc.hypergraphs.invariants.structure import (
    is_intersecting,
    is_linear,
    is_pair_covering,
    is_simple,
    is_sperner,
)


def test_is_simple():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    assert is_simple(H) is True


def test_is_linear_true():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {3, 4}])
    assert is_linear(H) is True


def test_is_linear_false():
    H = Hypergraph(E=[{1, 2, 3}, {2, 3, 4}])
    assert is_linear(H) is False


def test_is_intersecting_true():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {2, 4}])
    assert is_intersecting(H) is True


def test_is_intersecting_false():
    H = Hypergraph(E=[{1, 2}, {3, 4}])
    assert is_intersecting(H) is False


def test_is_pair_covering_true():
    H = Hypergraph(E=[{1, 2, 3}])
    assert is_pair_covering(H) is True


def test_is_pair_covering_false():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    assert is_pair_covering(H) is False


def test_is_pair_covering_vacuous():
    assert is_pair_covering(Hypergraph()) is True
    assert is_pair_covering(Hypergraph(V=[1])) is True


def test_is_sperner_true():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {1, 3}])
    assert is_sperner(H) is True


def test_is_sperner_false():
    H = Hypergraph(allow_singletons=True, E=[{1}, {1, 2}, {2, 3}])
    assert is_sperner(H) is False
