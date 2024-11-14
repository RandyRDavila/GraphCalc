import networkx as nx
from graphcalc.degree import (
    degree,
    degree_sequence,
    average_degree,
    maximum_degree,
    minimum_degree,
    )

def test_degree():
    G = nx.complete_graph(4)
    result = degree(G, 0)
    assert result == 3

def test_degree_sequence():
    G = nx.complete_graph(4)
    result = degree_sequence(G)
    assert result == [3, 3, 3, 3]

def test_average_degree():
    G = nx.complete_graph(4)
    result = average_degree(G)
    assert result == 3

def test_maximum_degree():
    G = nx.complete_graph(4)
    result = maximum_degree(G)
    assert result == 3

def test_minimum_degree():
    G = nx.complete_graph(4)
    result = minimum_degree(G)
    assert result == 3
