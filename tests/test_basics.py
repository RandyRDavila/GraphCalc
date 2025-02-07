import pytest
from graphcalc.core.basics import (
    SimpleGraph,
    order,
    size,
    connected,
    diameter,
    radius,
    connected_and_bipartite,
    connected_and_cubic,
)

@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 4),  # Complete graph K4
    ([(0, 1), (1, 2)], 3),                                  # Path graph with 3 nodes                                   # Single-node graph
])
def test_order(edges, expected):
    G = SimpleGraph(edges=edges)
    assert order(G) == expected

@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 6),  # Complete graph K4
    ([(0, 1), (1, 2)], 2),                                  # Path graph with 3 nodes                                    # Single-node graph
])
def test_size(edges, expected):
    G = SimpleGraph(edges=edges)
    assert size(G) == expected

@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 1),  # Diameter of a complete graph
    ([(0, 1), (1, 2), (2, 3)], 3),                          # Diameter of a path graph
    ([(0, 1), (1, 2), (2, 3), (3, 0)], 2),                  # Diameter of a cycle graph
])
def test_diameter(edges, expected):
    G = SimpleGraph(edges=edges)
    assert diameter(G) == expected

@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 1),  # Radius of a complete graph
    ([(0, 1), (1, 2), (2, 3)], 2),                          # Radius of a path graph
    ([(0, 1), (1, 2), (2, 3), (3, 0)], 2),                  # Radius of a cycle graph
])
def test_radius(edges, expected):
    G = SimpleGraph(edges=edges)
    assert radius(G) == expected

@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], True),  # Complete graph is connected
    ([(0, 1), (1, 2), (2, 3)], True),                          # Path graph is connected
    ([(0, 1), (2, 3)], False),                                 # Disconnected graph
])
def test_connected(edges, expected):
    G = SimpleGraph(edges=edges)
    assert connected(G) == expected

@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], False),  # Complete graph is not bipartite
    ([(0, 1), (1, 2), (2, 3)], True),                           # Path graph is bipartite
    ([(0, 1), (1, 2), (2, 3), (3, 0)], True),                   # Even cycle is bipartite
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], False),          # Odd cycle is not bipartite
])
def test_connected_and_bipartite(edges, expected):
    G = SimpleGraph(edges=edges)
    assert connected_and_bipartite(G) == expected

@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (2, 0), (3, 0), (3, 1), (3, 2)], True),   # Petersen graph is cubic
    ([(0, 1), (1, 2), (2, 0)], False),                         # Cycle graph is not cubic
    ([(0, 1), (1, 2), (2, 3), (3, 0)], False),                 # Star graph is not cubic
])
def test_connected_and_cubic(edges, expected):
    G = SimpleGraph(edges=edges)
    assert connected_and_cubic(G) == expected
