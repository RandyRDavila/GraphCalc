import pytest
import networkx as nx
from graphcalc.domination import (
    is_dominating_set,
    minimum_dominating_set,
    domination_number,
    minimum_total_domination_set,
    total_domination_number,
    minimum_independent_dominating_set,
    independent_domination_number,
    minimum_outer_connected_dominating_set,
    outer_connected_domination_number,
    double_roman_domination_number,
    rainbow_domination_number,
    restrained_domination_number,
)

@pytest.mark.parametrize("G, dom_set, expected", [
    (nx.complete_graph(4), {0}, True),
    (nx.star_graph(4), {0}, True),
    (nx.path_graph(4), {1, 3}, True),  # Non-minimal, still valid
    (nx.cycle_graph(5), {0, 2}, True),
    (nx.cycle_graph(5), {0, 1}, False),  # Incomplete dominating set
])
def test_is_dominating_set(G, dom_set, expected):
    assert is_dominating_set(G, dom_set) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(4), {0}),  # Center of the star
    (nx.complete_graph(4), {0}),  # Any single node
    (nx.path_graph(4), {0, 3}),  # Minimal set for endpoints
    (nx.cycle_graph(5), {0, 2}),  # Alternating vertices
])
def test_minimum_dominating_set(G, expected):
    result = minimum_dominating_set(G)
    assert len(result) == len(expected)

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(4), 1),
    (nx.complete_graph(4), 1),
    (nx.path_graph(4), 2),
    (nx.cycle_graph(5), 2),
])
def test_domination_number(G, expected):
    assert domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(4), 2),  # Center requires all leaves
    (nx.path_graph(4), 2),  # Disjoint pairs dominate
    (nx.cycle_graph(5), 3),  # Total domination forces additional vertex
])
def test_minimum_total_domination_set(G, expected):
    result = minimum_total_domination_set(G)
    assert len(result) == expected
    for node in result:
        assert all(nx.has_path(G, node, neighbor) for neighbor in result if neighbor != node)

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(4), 2),
    (nx.path_graph(4), 2),
    (nx.cycle_graph(5), 3),
])
def test_total_domination_number(G, expected):
    assert total_domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(4), {0}),  # Center is independent and dominates
    (nx.path_graph(4), {0, 3}),  # Endpoints independent and dominating
    (nx.cycle_graph(5), {0, 2}),  # Alternating vertices
])
def test_minimum_independent_dominating_set(G, expected):
    result = minimum_independent_dominating_set(G)
    assert len(result) == len(expected)

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(4), 1),
    (nx.path_graph(4), 2),
    (nx.cycle_graph(5), 2),
])
def test_independent_domination_number(G, expected):
    assert independent_domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(5), {0}),  # Ensure graph object is valid
    (nx.path_graph(4), {0, 3}),
    (nx.cycle_graph(5), {0, 1, 2}),
])
def test_minimum_outer_connected_dominating_set(G, expected):
    result = minimum_outer_connected_dominating_set(G)
    assert len(result) == len(expected)

@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), 1),
    (nx.path_graph(4), 2),
    (nx.cycle_graph(5), 3),
])
def test_outer_connected_domination_number(G, expected):
    assert outer_connected_domination_number(G) == expected
