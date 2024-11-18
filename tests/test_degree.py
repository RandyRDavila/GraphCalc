import pytest
import networkx as nx
from graphcalc.degree import (
    degree,
    degree_sequence,
    average_degree,
    maximum_degree,
    minimum_degree,
)

@pytest.mark.parametrize("G, node, expected", [
    (nx.complete_graph(4), 0, 3),  # Complete graph: degree is n-1
    (nx.path_graph(4), 1, 2),  # Path graph: middle node degree is 2
    (nx.cycle_graph(4), 2, 2),  # Cycle graph: all nodes degree is 2
    (nx.star_graph(4), 0, 4),  # Star graph: center node degree is n
    (nx.star_graph(4), 1, 1),  # Star graph: leaf node degree is 1
])
def test_degree(G, node, expected):
    assert degree(G, node) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), [3, 3, 3, 3]),  # All nodes same degree
    (nx.path_graph(4), [2, 2, 1, 1]),  # Endpoints degree 1, middle 2
    (nx.cycle_graph(4), [2, 2, 2, 2]),  # All nodes degree 2
    (nx.star_graph(4), [4, 1, 1, 1, 1]),  # Center 4, leaves 1
])
def test_degree_sequence(G, expected):
    assert degree_sequence(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 3),  # Degree is consistent across all nodes
    (nx.path_graph(4), 1.5),  # Average of [1, 2, 2, 1]
    (nx.cycle_graph(4), 2),  # All nodes degree 2
    (nx.star_graph(4), 1.6),  # Average of [4, 1, 1, 1, 1]
])
def test_average_degree(G, expected):
    assert average_degree(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 3),  # Max degree is consistent
    (nx.path_graph(4), 2),  # Max degree of path graph
    (nx.cycle_graph(4), 2),  # All nodes degree 2
    (nx.star_graph(4), 4),  # Center node degree
])
def test_maximum_degree(G, expected):
    assert maximum_degree(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 3),  # Min degree is consistent
    (nx.path_graph(4), 1),  # Endpoints degree
    (nx.cycle_graph(4), 2),  # All nodes degree 2
    (nx.star_graph(4), 1),  # Leaf node degree
])
def test_minimum_degree(G, expected):
    assert minimum_degree(G) == expected

def test_singleton_graph():
    G = nx.path_graph(1)
    assert degree(G, 0) == 0  # Single node, no edges
    assert degree_sequence(G) == [0]  # Degree of single node
    assert average_degree(G) == 0  # No edges
    assert maximum_degree(G) == 0  # Single node, max degree 0
    assert minimum_degree(G) == 0  # Single node, min degree 0
