import networkx as nx
from graphcalc.zero_forcing import (
    is_zero_forcing_set,
    minimum_zero_forcing_set,
    zero_forcing_number,
    minimum_k_forcing_set,
    k_forcing_number,
)

def test_is_zero_forcing_set():
    G = nx.star_graph(4)
    zero_forcing_set = {1, 2, 3}
    result = is_zero_forcing_set(G, zero_forcing_set)
    assert result == True

def test_minimum_zero_forcing_set():
    G = nx.path_graph(4)
    result = minimum_zero_forcing_set(G)
    assert result == {0}

def test_zero_forcing_number():
    G = nx.path_graph(4)
    result = zero_forcing_number(G)
    assert result == 1

    G = nx.star_graph(4)
    result = zero_forcing_number(G)
    assert result == 3

    G = nx.complete_graph(4)
    result = zero_forcing_number(G)
    assert result == 3

def test_minimum_k_forcing_set():
    G = nx.path_graph(4)
    result = minimum_k_forcing_set(G, 1)
    assert result == {0}

def test_k_forcing_number():
    G = nx.path_graph(4)
    result = k_forcing_number(G, 1)
    assert result == 1

    G = nx.cycle_graph(4)
    result = k_forcing_number(G, 1)
    assert result == 2

    G = nx.cycle_graph(4)
    result = k_forcing_number(G, 2)
    assert result == 1
