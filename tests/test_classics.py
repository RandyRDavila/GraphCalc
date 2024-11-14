import networkx as nx
from graphcalc.classics import (
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

def test_maximum_clique():
    G = nx.complete_graph(4)
    result = maximum_clique(G)
    assert result == {0, 1, 2, 3}

def test_clique_number():
    G = nx.complete_graph(3)
    result = clique_number(G)
    assert result == 3

def test_maximum_independent_set():
    G = nx.star_graph(4)
    result = maximum_independent_set(G)
    assert result == {1, 2, 3, 4}

def test_independence_number():
    G = nx.complete_graph(3)
    result = independence_number(G)
    assert result == 1

def test_chromatic_number():
    G = nx.complete_graph(3)
    result = chromatic_number(G)
    assert result == 3

def test_maximum_matching():
    G = nx.path_graph(4)
    result = maximum_matching(G)
    assert result == {(0, 1), (2, 3)}

def test_matching_number():
    G = nx.complete_graph(4)
    result = matching_number(G)
    assert result == 2

def test_minimum_vertex_cover():
    G = nx.star_graph(4)
    result = minimum_vertex_cover(G)
    assert result == {0}

def test_vertex_cover_number():
    G = nx.star_graph(4)
    result = vertex_cover_number(G)
    assert result == 1

def test_edge_cover_number():
    G = nx.star_graph(4)
    result = edge_cover_number(G)
    assert result == 4
