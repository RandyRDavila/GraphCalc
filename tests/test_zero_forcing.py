import networkx as nx
from graphcalc.zero_forcing import (
    is_zero_forcing_set,
    minimum_zero_forcing_set,
    zero_forcing_number,
    minimum_k_forcing_set,
    k_forcing_number,
    is_power_dominating_set,
    minimum_power_dominating_set,
    power_domination_number,
    positive_semidefinite_zero_forcing_number,
)

def test_is_zero_forcing_set():
    test_cases = [
        (nx.star_graph(4), {1, 2, 3}, True),
        (nx.path_graph(4), {1}, False),
        (nx.cycle_graph(4), {0, 1}, True),
        (nx.cycle_graph(4), {0, 2}, False),
    ]
    for G, forcing_set, expected in test_cases:
        assert is_zero_forcing_set(G, forcing_set) == expected

def test_minimum_zero_forcing_set():
    G = nx.path_graph(4)
    result = minimum_zero_forcing_set(G)
    assert result == {0}

def test_zero_forcing_number():
    test_cases = [
        (nx.path_graph(4), 1),
        (nx.star_graph(4), 3),
        (nx.complete_graph(4), 3),
    ]
    for G, expected in test_cases:
        assert zero_forcing_number(G) == expected

def test_minimum_k_forcing_set():
    G = nx.path_graph(4)
    result = minimum_k_forcing_set(G, 1)
    assert result == {0}

def test_k_forcing_number():
    test_cases = [
        (nx.path_graph(4), 1, 1),
        (nx.cycle_graph(4), 1, 2),
        (nx.complete_graph(3), 2, 1),
        (nx.cycle_graph(4), 2, 1),
        (nx.complete_graph(4), 3, 1),
        (nx.cycle_graph(4), 3, 1),
    ]
    for G, k, expected in test_cases:
        assert k_forcing_number(G, k) == expected

def test_is_power_dominating_set():
    test_cases = [
        (nx.star_graph(4), {0}, True),
        (nx.path_graph(4), {2}, True),
        (nx.complete_graph(4), {0}, True),
    ]
    for G, dominating_set, expected in test_cases:
        assert is_power_dominating_set(G, dominating_set) == expected

def test_minimum_power_dominating_set():
    G = nx.star_graph(4)
    result = minimum_power_dominating_set(G)
    assert result == {0}

def test_power_domination_number():
    test_cases = [
        (nx.star_graph(4), 1),
        (nx.path_graph(4), 1),
        (nx.complete_graph(4), 1),
    ]
    for G, expected in test_cases:
        result = power_domination_number(G)
        assert result == expected

def test_positive_semidefinite_zero_forcing_number():
    for n in range(2, 6):  # Test for trees of size 2 to 5
        G = nx.random_labeled_tree(n)
        assert positive_semidefinite_zero_forcing_number(G) == 1
