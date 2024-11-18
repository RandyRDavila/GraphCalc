import pytest
import networkx as nx
from graphcalc.neighborhoods import (
    neighborhood,
    closed_neighborhood,
    set_neighbors,
    set_closed_neighbors,
)

@pytest.mark.parametrize("G, node, expected", [
    (nx.complete_graph(4), 0, {1, 2, 3}),
    (nx.star_graph(3), 0, {1, 2, 3}),
    (nx.path_graph(4), 1, {0, 2}),
    (nx.path_graph(4), 3, {2}),
])
def test_neighborhood(G, node, expected):
    assert neighborhood(G, node) == expected

@pytest.mark.parametrize("G, node, expected", [
    (nx.complete_graph(4), 0, {0, 1, 2, 3}),
    (nx.star_graph(3), 0, {0, 1, 2, 3}),
    (nx.path_graph(4), 1, {0, 1, 2}),
    (nx.path_graph(4), 3, {2, 3}),
])
def test_closed_neighborhood(G, node, expected):
    assert closed_neighborhood(G, node) == expected

@pytest.mark.parametrize("G, node_set, expected", [
    (nx.star_graph(3), {1}, {0}),
    (nx.path_graph(4), {1}, {0, 2}),
    (nx.path_graph(4), {1, 2}, {0, 1, 2, 3}),
])
def test_set_neighbors(G, node_set, expected):
    assert set_neighbors(G, node_set) == expected

@pytest.mark.parametrize("G, node_set, expected", [
    (nx.star_graph(3), {1}, {0, 1}),
    (nx.path_graph(4), {1}, {0, 1, 2}),
    (nx.path_graph(4), {1, 2}, {0, 1, 2, 3}),
])
def test_set_closed_neighbors(G, node_set, expected):
    assert set_closed_neighbors(G, node_set) == expected

def test_invalid_node():
    G = nx.path_graph(4)
    with pytest.raises(KeyError):
        neighborhood(G, 10)  # Node 10 doesn't exist
    with pytest.raises(KeyError):
        closed_neighborhood(G, 10)  # Node 10 doesn't exist
