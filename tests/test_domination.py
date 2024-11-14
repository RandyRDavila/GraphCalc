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
)

def test_is_dominating_set():
    G = nx.complete_graph(4)
    dominating_set = {0}
    result = is_dominating_set(G, dominating_set)
    assert result == True

def test_minimum_dominating_set():
    G = nx.star_graph(3)
    result = minimum_dominating_set(G)
    assert result == {0}

def test_domination_number():
    G = nx.star_graph(3)
    result = domination_number(G)
    assert result == 1

def test_minimum_total_domination_set():
    G = nx.star_graph(3)
    result = minimum_total_domination_set(G)
    assert (00 in result) and len(result) == 2

def test_total_domination_number():
    G = nx.star_graph(3)
    result = total_domination_number(G)
    assert result == 2

def test_minimum_independent_dominating_set():
    G = nx.star_graph(3)
    result = minimum_independent_dominating_set(G)
    assert result == {0}

def test_independent_domination_number():
    G = nx.star_graph(3)
    result = independent_domination_number(G)
    assert result == 1

def test_minimum_outer_connected_dominating_set():
    G = nx.complete_graph(4)
    G.add_edge(0, 4)
    result = minimum_outer_connected_dominating_set(G)
    assert result == {0, 4}

def test_outer_connected_domination_number():
    G = nx.complete_graph(4)
    G.add_edge(0, 4)
    result = outer_connected_domination_number(G)
    assert result == 2
