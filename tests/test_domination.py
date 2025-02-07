import pytest
import networkx as nx
from graphcalc.generators.simple_graphs import (
    complete_graph,
    cycle_graph,
    path_graph,
    star_graph,
)
from graphcalc.invariants.domination import (
    is_dominating_set,
    minimum_dominating_set,
    domination_number,
    minimum_total_domination_set,
    total_domination_number,
    minimum_independent_dominating_set,
    independent_domination_number,
    minimum_outer_connected_dominating_set,
    outer_connected_domination_number,
    # double_roman_domination_number,
    # rainbow_domination_number,
    # restrained_domination_number,
)

@pytest.mark.parametrize("G, dom_set, expected", [
    (complete_graph(4), {0}, True),
    (star_graph(4), {0}, True),
    (path_graph(4), {1, 3}, True),  # Non-minimal, still valid
    (cycle_graph(5), {0, 2}, True),
    (cycle_graph(5), {0, 1}, False),  # Incomplete dominating set
])
def test_is_dominating_set(G, dom_set, expected):
    assert is_dominating_set(G, dom_set) == expected

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), {0}),  # Center of the star
    (complete_graph(4), {0}),  # Any single node
    (path_graph(4), {0, 3}),  # Minimal set for endpoints
    (cycle_graph(5), {0, 2}),  # Alternating vertices
])
def test_minimum_dominating_set(G, expected):
    result = minimum_dominating_set(G)
    assert len(result) == len(expected)

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), 1),
    (complete_graph(4), 1),
    (path_graph(4), 2),
    (cycle_graph(5), 2),
])
def test_domination_number(G, expected):
    assert domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), 2),  # Center requires all leaves
    (path_graph(4), 2),  # Disjoint pairs dominate
    (cycle_graph(5), 3),  # Total domination forces additional vertex
])
def test_minimum_total_domination_set(G, expected):
    result = minimum_total_domination_set(G)
    assert len(result) == expected
    for node in result:
        assert all(nx.has_path(G, node, neighbor) for neighbor in result if neighbor != node)

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), 2),
    (path_graph(4), 2),
    (cycle_graph(5), 3),
])
def test_total_domination_number(G, expected):
    assert total_domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), {0}),  # Center is independent and dominates
    (path_graph(4), {0, 3}),  # Endpoints independent and dominating
    (cycle_graph(5), {0, 2}),  # Alternating vertices
])
def test_minimum_independent_dominating_set(G, expected):
    result = minimum_independent_dominating_set(G)
    assert len(result) == len(expected)

@pytest.mark.parametrize("G, expected", [
    (star_graph(4), 1),
    (path_graph(4), 2),
    (cycle_graph(5), 2),
])
def test_independent_domination_number(G, expected):
    assert independent_domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(5), {0}),  # Ensure graph object is valid
    (path_graph(4), {0, 3}),
    (cycle_graph(5), {0, 1, 2}),
])
def test_minimum_outer_connected_dominating_set(G, expected):
    result = minimum_outer_connected_dominating_set(G)
    assert len(result) == len(expected)

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 1),
    (path_graph(4), 2),
    (cycle_graph(5), 3),
])
def test_outer_connected_domination_number(G, expected):
    assert outer_connected_domination_number(G) == expected

def petersen_tests():
    G = nx.petersen_graph()
    assert domination_number(G) == 3
    assert total_domination_number(G) == 4
    assert independent_domination_number(G) == 3
    assert outer_connected_domination_number(G) == 3
