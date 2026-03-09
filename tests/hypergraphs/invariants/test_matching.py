import pytest

from graphcalc.hypergraphs.core.basics import Hypergraph
from graphcalc.hypergraphs.invariants.matching import (
    edge_cover_number,
    fractional_matching_number,
    matching_number,
    maximum_matching,
    minimum_edge_cover,
)


def test_maximum_matching_empty_hypergraph():
    H = Hypergraph()
    assert maximum_matching(H) == set()
    assert matching_number(H) == 0
    assert fractional_matching_number(H) == 0.0


def test_maximum_matching_simple_example():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {4, 5}])
    M = maximum_matching(H)
    assert len(M) == 2
    assert frozenset({4, 5}) in M
    assert matching_number(H) == 2


def test_fractional_matching_number_simple_example():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    assert fractional_matching_number(H) == pytest.approx(1.0)


def test_minimum_edge_cover_empty_vertex_set():
    H = Hypergraph()
    assert minimum_edge_cover(H) == set()
    assert edge_cover_number(H) == 0


def test_minimum_edge_cover_simple_example():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    C = minimum_edge_cover(H)
    assert len(C) == 2
    assert edge_cover_number(H) == 2


def test_minimum_edge_cover_single_edge_covers_all():
    H = Hypergraph(E=[{1, 2, 3}])
    C = minimum_edge_cover(H)
    assert C == {frozenset({1, 2, 3})}
    assert edge_cover_number(H) == 1


def test_minimum_edge_cover_raises_on_isolated_vertex():
    H = Hypergraph(V=[1, 2, 3], E=[{1, 2}], auto_add_vertices=False)
    with pytest.raises(ValueError, match="isolated vertices"):
        minimum_edge_cover(H)
