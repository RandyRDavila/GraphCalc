import pytest
import networkx as nx
from graphcalc.basics import (
    order,
    size,
    connected,
    diameter,
    radius,
    connected_and_bipartite,
    connected_and_cubic,
    connected_and_subcubic,
    connected_and_regular,
    connected_and_eulerian,
    tree,
)

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 4),  # Complete graph with 4 nodes
    (nx.path_graph(3), 3),      # Path graph with 3 nodes
    (nx.Graph(), 0),            # Empty graph
    (nx.path_graph(1), 1),      # Single-node graph
])
def test_order(G, expected):
    assert order(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 6),  # Complete graph with 4 nodes
    (nx.path_graph(3), 2),      # Path graph with 3 nodes
    (nx.Graph(), 0),            # Empty graph
    (nx.path_graph(1), 0),      # Single-node graph
])
def test_size(G, expected):
    assert size(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 1),  # Diameter of a complete graph
    (nx.path_graph(4), 3),      # Diameter of a path graph
    (nx.cycle_graph(4), 2),     # Diameter of a cycle graph
])
def test_diameter(G, expected):
    assert diameter(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 1),  # Radius of a complete graph
    (nx.path_graph(4), 2),      # Radius of a path graph
    (nx.cycle_graph(4), 2),     # Radius of a cycle graph
])
def test_radius(G, expected):
    assert radius(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), True),   # Complete graph is connected
    (nx.path_graph(4), True),       # Path graph is connected
    (nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)), False),  # Disconnected graph
])
def test_connected(G, expected):
    assert connected(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), False),  # Complete graph is not bipartite
    (nx.path_graph(4), True),       # Path graph is bipartite
    (nx.cycle_graph(4), True),      # Even cycle is bipartite
    (nx.cycle_graph(5), False),     # Odd cycle is not bipartite
])
def test_connected_and_bipartite(G, expected):
    assert connected_and_bipartite(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), True),  # Complete graph K_4 is cubic
    (nx.cycle_graph(6), False),      # Cycle graph with degree 2 is not cubic
    (nx.star_graph(3), False),      # Star graph is not cubic
    (nx.petersen_graph(), True),    # Petersen graph is cubic
])
def test_connected_and_cubic(G, expected):
    assert connected_and_cubic(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(3), True),       # Star graph is subcubic
    (nx.cycle_graph(6), True),      # Cycle graph is subcubic
    (nx.complete_graph(4), True),  # Complete graph is not subcubic
    (nx.complete_graph(5), False),  # Complete graph is not subcubic
])
def test_connected_and_subcubic(G, expected):
    assert connected_and_subcubic(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), True),  # Complete graph is regular
    (nx.cycle_graph(4), True),     # Cycle graph is regular
    (nx.path_graph(4), False),     # Path graph is not regular
])
def test_connected_and_regular(G, expected):
    assert connected_and_regular(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), False),  # Complete graph is not Eulerian
    (nx.cycle_graph(4), True),      # Cycle graph is Eulerian
    (nx.path_graph(4), False),      # Path graph is not Eulerian
])
def test_connected_and_eulerian(G, expected):
    assert connected_and_eulerian(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), False),  # Complete graph is not a tree
    (nx.path_graph(4), True),       # Path graph is a tree
    (nx.cycle_graph(4), False),     # Cycle graph is not a tree
])
def test_tree(G, expected):
    assert tree(G) == expected
