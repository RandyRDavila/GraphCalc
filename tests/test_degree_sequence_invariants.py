import pytest
import networkx as nx
from graphcalc.degree_sequence_invariants import (
    sub_k_domination_number,
    slater,
    sub_total_domination_number,
    annihilation_number,
    residue,
    harmonic_index,
)

@pytest.mark.parametrize("G, k, expected", [
    (nx.cycle_graph(4), 1, 2),  # Cycle graph
    (nx.path_graph(4), 1, 2),  # Path graph
    (nx.star_graph(4), 1, 1),  # Star graph, k = 2
    (nx.complete_graph(4), 1, 1),  # Complete graph, k = 2
])
def test_sub_k_domination_number(G, k, expected):
    assert sub_k_domination_number(G, k) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.cycle_graph(4), 2),  # Cycle graph
    (nx.path_graph(4), 2),  # Path graph
    (nx.star_graph(4), 1),  # Star graph
    (nx.complete_graph(4), 1),  # Complete graph
])
def test_slater(G, expected):
    assert slater(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.cycle_graph(4), 2),  # Cycle graph
    (nx.path_graph(4), 2),  # Path graph
    (nx.star_graph(4), 2),  # Star graph
    (nx.complete_graph(4), 2),  # Complete graph
])
def test_sub_total_domination_number(G, expected):
    assert sub_total_domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.cycle_graph(4), 2),  # Cycle graph
    (nx.path_graph(4), 2),  # Path graph
    (nx.star_graph(4), 4),  # Star graph
    (nx.complete_graph(4), 2),  # Complete graph
])
def test_annihilation_number(G, expected):
    assert annihilation_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.cycle_graph(4), 2),  # Cycle graph
    (nx.path_graph(4), 2),  # Path graph
    (nx.star_graph(4), 4),  # Star graph
    (nx.complete_graph(4), 1),  # Complete graph
])
def test_residue(G, expected):
    assert residue(G) == expected

@pytest.mark.parametrize("G, expected", [
    (nx.cycle_graph(4), 2),  # Cycle graph
    (nx.complete_graph(4), 2),  # Complete graph
])
def test_harmonic_index(G, expected):
    assert harmonic_index(G) == pytest.approx(expected, rel=1e-2)
