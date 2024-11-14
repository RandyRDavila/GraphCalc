import networkx as nx
from graphcalc.basics import (
    order,
    size,
    connected,
    diameter,
    radius,
    connected_and_bipartite,
    connected_and_cubic,
    connected_and_subcubic,
    connected_and_regular,
    connected_and_eulerian,
    tree,
    )

def test_order():
    G = nx.complete_graph(4)
    result = order(G)
    assert result == 4

def test_size():
    G = nx.complete_graph(4)
    result = size(G)
    assert result == 6

def test_diameter():
    G = nx.complete_graph(4)
    result = diameter(G)
    assert result == 1

def test_radius():
    G = nx.complete_graph(4)
    result = radius(G)
    assert result == 1

def test_connected():
    G = nx.complete_graph(4)
    result = connected(G)
    assert result == True

def test_connected_and_bipartite():
    G = nx.complete_graph(4)
    result = connected_and_bipartite(G)
    assert result == False

def test_connected_and_cubic():
    G = nx.complete_graph(4)
    result = connected_and_cubic(G)
    assert result == True

def test_connected_and_subcubic():
    G = nx.star_graph(3)
    result = connected_and_subcubic(G)
    assert result == True

def test_connected_and_regular():
    G = nx.complete_graph(4)
    result = connected_and_regular(G)
    assert result == True

def test_connected_and_eulerian():
    G = nx.complete_graph(4)
    result = connected_and_eulerian(G)
    assert result == False

def test_tree():
    G = nx.complete_graph(4)
    result = tree(G)
    assert result == False
