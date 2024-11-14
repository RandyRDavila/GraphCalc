import networkx as nx
from graphcalc.neighborhoods import (
    neighborhood,
    closed_neighborhood,
    set_neighbors,
    set_closed_neighbors,
    )

def test_neighborhood():
    G = nx.complete_graph(4)
    result = neighborhood(G, 0)
    assert result == {1, 2, 3}

def test_closed_neighborhood():
    G = nx.complete_graph(4)
    result = closed_neighborhood(G, 0)
    assert result == {0, 1, 2, 3}

def test_set_neighbors():
    G = nx.star_graph(3)
    result = set_neighbors(G, {1})
    assert result == {0}

def test_set_closed_neighbors():
    G = nx.star_graph(3)
    result = set_closed_neighbors(G, {1})
    assert result == {0, 1}





