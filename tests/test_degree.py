import pytest
from graphcalc.generators.simple_graphs import (
    complete_graph,
    cycle_graph,
    path_graph,
    star_graph,
)
from graphcalc.invariants.degree import (
    degree,
    degree_sequence,
    average_degree,
    maximum_degree,
    minimum_degree,
    sub_k_domination_number,
    slater,
    sub_total_domination_number,
    annihilation_number,
    residue,
    harmonic_index,
)

@pytest.mark.parametrize("G, node, expected", [
    (complete_graph(4), 0, 3),  # Complete graph: degree is n-1
    (path_graph(4), 1, 2),  # Path graph: middle node degree is 2
    (cycle_graph(4), 2, 2),  # Cycle graph: all nodes degree is 2
    (star_graph(4), 0, 4),  # Star graph: center node degree is n
    (star_graph(4), 1, 1),  # Star graph: leaf node degree is 1
])
def test_degree(G, node, expected):
    assert degree(G, node) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), [3, 3, 3, 3]),  # All nodes same degree
    (path_graph(4), [2, 2, 1, 1]),  # Endpoints degree 1, middle 2
    (cycle_graph(4), [2, 2, 2, 2]),  # All nodes degree 2
    (star_graph(4), [4, 1, 1, 1, 1]),  # Center 4, leaves 1
])
def test_degree_sequence(G, expected):
    assert degree_sequence(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 3),  # Degree is consistent across all nodes
    (path_graph(4), 1.5),  # Average of [1, 2, 2, 1]
    (cycle_graph(4), 2),  # All nodes degree 2
    (star_graph(4), 1.6),  # Average of [4, 1, 1, 1, 1]
])
def test_average_degree(G, expected):
    assert average_degree(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 3),  # Max degree is consistent
    (path_graph(4), 2),  # Max degree of path graph
    (cycle_graph(4), 2),  # All nodes degree 2
    (star_graph(4), 4),  # Center node degree
])
def test_maximum_degree(G, expected):
    assert maximum_degree(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 3),  # Min degree is consistent
    (path_graph(4), 1),  # Endpoints degree
    (cycle_graph(4), 2),  # All nodes degree 2
    (star_graph(4), 1),  # Leaf node degree
])
def test_minimum_degree(G, expected):
    assert minimum_degree(G) == expected

# def test_singleton_graph():
#     G = path_graph(1)
#     assert degree(G, 0) == 0  # Single node, no edges
#     assert degree_sequence(G) == [0]  # Degree of single node
#     assert average_degree(G) == 0  # No edges
#     assert maximum_degree(G) == 0  # Single node, max degree 0
#     assert minimum_degree(G) == 0  # Single node, min degree 0


@pytest.mark.parametrize("G, k, expected", [
    (cycle_graph(4), 1, 2),  # Cycle graph
    (path_graph(4), 1, 2),  # Path graph
    (star_graph(4), 1, 1),  # Star graph, k = 2
    (complete_graph(4), 1, 1),  # Complete graph, k = 2
])
def test_sub_k_domination_number(G, k, expected):
    assert sub_k_domination_number(G, k) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 1),  # Star graph
    (complete_graph(4), 1),  # Complete graph
])
def test_slater(G, expected):
    assert slater(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 2),  # Star graph
    (complete_graph(4), 2),  # Complete graph
])
def test_sub_total_domination_number(G, expected):
    assert sub_total_domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 4),  # Star graph
    (complete_graph(4), 2),  # Complete graph
])
def test_annihilation_number(G, expected):
    assert annihilation_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 4),  # Star graph
    (complete_graph(4), 1),  # Complete graph
])
def test_residue(G, expected):
    assert residue(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (complete_graph(4), 2),  # Complete graph
])
def test_harmonic_index(G, expected):
    assert harmonic_index(G) == pytest.approx(expected, rel=1e-2)
