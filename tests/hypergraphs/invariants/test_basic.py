import pytest

from graphcalc.hypergraphs.core.basics import Hypergraph
from graphcalc.hypergraphs.invariants.basic import (
    average_degree,
    co_rank,
    degree_sequence,
    edge_size_sequence,
    is_d_regular,
    is_empty,
    is_k_uniform,
    is_regular,
    is_trivial,
    maximum_degree,
    minimum_degree,
    number_of_edges,
    number_of_vertices,
    rank,
)


def test_number_of_vertices_and_edges():
    H = Hypergraph(E=[{1, 2}, {2, 3, 4}])
    assert number_of_vertices(H) == 4
    assert number_of_edges(H) == 2


def test_is_empty():
    assert is_empty(Hypergraph()) is True
    assert is_empty(Hypergraph(E=[{1, 2}])) is False


def test_is_trivial():
    assert is_trivial(Hypergraph()) is True
    assert is_trivial(Hypergraph(V=[1])) is True
    assert is_trivial(Hypergraph(V=[1, 2])) is False


def test_rank_and_co_rank():
    H = Hypergraph(E=[{1, 2}, {2, 3, 4}, {5, 6}])
    assert rank(H) == 3
    assert co_rank(H) == 2


def test_rank_and_co_rank_empty():
    H = Hypergraph()
    assert rank(H) == 0
    assert co_rank(H) == 0


def test_is_k_uniform():
    H1 = Hypergraph(E=[{1, 2}, {2, 3}, {4, 5}])
    H2 = Hypergraph(E=[{1, 2}, {2, 3, 4}])
    assert is_k_uniform(H1, 2) is True
    assert is_k_uniform(H1, 3) is False
    assert is_k_uniform(H2, 2) is False


def test_degree_extremes_and_average():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {2, 3, 4}])
    assert maximum_degree(H) == 3
    assert minimum_degree(H) == 1
    assert average_degree(H) == pytest.approx(1.75)


def test_degree_extremes_and_average_empty():
    H = Hypergraph()
    assert maximum_degree(H) == 0
    assert minimum_degree(H) == 0
    assert average_degree(H) == 0.0


def test_degree_sequence():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {2, 3, 4}])
    assert degree_sequence(H) == [3, 2, 1, 1]
    assert degree_sequence(H, nonincreasing=False) == [1, 1, 2, 3]


def test_edge_size_sequence():
    H = Hypergraph(E=[{1, 2}, {2, 3, 4}, {5, 6}])
    assert edge_size_sequence(H) == [3, 2, 2]
    assert edge_size_sequence(H, nonincreasing=False) == [2, 2, 3]


def test_is_regular_and_is_d_regular():
    H1 = Hypergraph(E=[{1, 2}, {3, 4}])
    H2 = Hypergraph(E=[{1, 2}, {2, 3}])

    assert is_regular(H1) is True
    assert is_d_regular(H1, 1) is True
    assert is_d_regular(H1, 2) is False

    assert is_regular(H2) is False
    assert is_d_regular(H2, 1) is False
