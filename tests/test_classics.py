import pytest
import networkx as nx
from graphcalc.generators.simple_graphs import (
    complete_graph,
    cycle_graph,
    path_graph,
    star_graph,
)
from graphcalc.invariants.classics import (
    maximum_clique,
    clique_number,
    maximum_independent_set,
    independence_number,
    chromatic_number,
    maximum_matching,
    matching_number,
    minimum_vertex_cover,
    vertex_cover_number,
    edge_cover_number,
)

@pytest.mark.parametrize("G, expected_cliques", [
    # Complete graph with 4 nodes: single maximum clique
    (complete_graph(4), [
        {0, 1, 2, 3},
    ]),
    # Cycle graph with 5 nodes: any two adjacent nodes form a maximum clique
    (cycle_graph(5), [
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0},
    ]),
    # Path graph with 4 nodes: each edge is a maximum clique
    (path_graph(4), [
        {0, 1}, {1, 2}, {2, 3},
    ]),
    # Star graph with 5 nodes: any edge forms a maximum clique
    (star_graph(4), [
        {0, 1}, {0, 2}, {0, 3}, {0, 4},
    ]),
])
def test_maximum_clique(G, expected_cliques):
    result = maximum_clique(G)
    # Ensure result matches one of the expected maximum cliques
    assert any(result == clique for clique in expected_cliques)

def test_edge_cases():
    # Empty graph
    G = nx.Graph()
    result = maximum_clique(G)
    assert result == set()

    # # Single-node graph
    # G = path_graph(1)
    # result = maximum_clique(G)
    # assert result == {0}

    # Disconnected graph
    G = nx.disjoint_union(complete_graph(3), path_graph(2))
    result = maximum_clique(G)
    assert result in [{0, 1, 2}, {3, 4}]  # Largest clique is in either component

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 4),
    (cycle_graph(5), 2),  # Maximum clique size
    (path_graph(4), 2),
])
def test_clique_number(G, expected):
    assert clique_number(G) == expected


@pytest.mark.parametrize("G, expected_independent_sets", [
    # Star graph with 5 nodes: any combination of leaves forms a maximum independent set
    (star_graph(4), [
        {1, 2, 3, 4},  # All leaves are independent
    ]),
    # Cycle graph with 6 nodes: alternating vertices form independent sets
    (cycle_graph(6), [
        {0, 2, 4}, {1, 3, 5},
    ]),
    # Path graph with 4 nodes: alternating vertices form independent sets
    (path_graph(4), [
        {0, 2}, {1, 3}, {0, 3},
    ]),
    # Complete graph with 4 nodes: each single node is independent
    (complete_graph(4), [
        {0}, {1}, {2}, {3},
    ]),
])
def test_maximum_independent_set(G, expected_independent_sets):
    result = maximum_independent_set(G)
    # Ensure result matches one of the expected independent sets
    assert any(result == independent_set for independent_set in expected_independent_sets)

def test_edge_cases():
    # Empty graph
    G = nx.Graph()
    result = maximum_independent_set(G)
    assert result == set()

    # # Single-node graph
    # G = path_graph(1)
    # result = maximum_independent_set(G)
    # assert result == {0}

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), 4),  # Number of leaves
    (complete_graph(3), 1),  # Single independent vertex in a clique
    (path_graph(4), 2),  # Alternating vertices
])
def test_independence_number(G, expected):
    assert independence_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 4),  # Complete graph requires 4 colors
    (cycle_graph(5), 3),  # Odd cycle requires 3 colors
    (path_graph(4), 2),  # Bipartite graph, 2 colors
])
def test_chromatic_number(G, expected):
    assert chromatic_number(G) == expected

@pytest.mark.parametrize("G, expected_matchings", [
    # Path graph with 4 nodes has two maximum matchings
    (path_graph(4), [
        {(0, 1), (2, 3)},
    ]),
    # Complete graph with 4 nodes has multiple matchings of size 2
    (complete_graph(4), [
        {(0, 1), (2, 3)},
        {(0, 2), (1, 3)},
        {(0, 3), (1, 2)},
    ]),
    # Star graph with 5 nodes: any 4 edges incident to the center form a valid maximum matching
    (star_graph(4), [
        {(0, 1)},
        {(0, 2)},
        {(0, 3)},
        {(0, 4)},
    ]),
])
def test_maximum_matching(G, expected_matchings):
    result = maximum_matching(G)
    # Ensure result matches any one of the expected matchings
    assert any(result == matching for matching in expected_matchings)

@pytest.mark.parametrize("G, expected", [
    (path_graph(4), 2),  # Path graph matching number
    (complete_graph(4), 2),  # Complete graph matching number
    (star_graph(4), 1),  # Star graph matching number
])
def test_matching_number(G, expected):
    assert matching_number(G) == expected

@pytest.mark.parametrize("G, expected_vertex_covers", [
    # Star graph with 5 nodes: only the center node is needed
    (star_graph(4), [
        {0},  # The center node covers all edges
    ]),
    # Complete graph with 4 nodes: any 3 nodes form a vertex cover
    (complete_graph(4), [
        {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
    ]),
    # Cycle graph with 4 nodes: alternating vertices cover edges
    (cycle_graph(4), [
        {0, 2}, {1, 3},  # Both pairs are valid vertex covers
    ]),
])
def test_minimum_vertex_cover(G, expected_vertex_covers):
    result = minimum_vertex_cover(G)
    # Ensure result matches one of the expected minimum vertex covers
    assert any(result == vertex_cover for vertex_cover in expected_vertex_covers)

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), 1),  # Center node
    (complete_graph(4), 3),  # All but one node required
    (cycle_graph(4), 2),  # Alternating vertices
])
def test_vertex_cover_number(G, expected):
    assert vertex_cover_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), 4),  # Star graph: 1 edge per leaf
    (complete_graph(4), 2),  # Complete graph: n - 1 edges
    (cycle_graph(4), 2),  # Cycle graph: alternating edges
])
def test_edge_cover_number(G, expected):
    assert edge_cover_number(G) == expected

def petersen_tests():
    G = nx.petersen_graph()
    assert independence_number(G) == 4
    assert chromatic_number(G) == 3
    assert matching_number(G) == 5
    assert vertex_cover_number(G) == 6
    assert edge_cover_number(G) == 3
    assert clique_number(G) == 2