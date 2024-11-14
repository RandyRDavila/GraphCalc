import networkx as nx
from graphcalc.degree_sequence_invariants import (
    sub_k_domination_number,
    slater,
    sub_total_domination_number,
    annihilation_number,
    residue,
    harmonic_index,
    )

def test_sub_k_domination_number():
    G = nx.cycle_graph(4)
    result = sub_k_domination_number(G, 1)
    assert result == 2

def test_slater():
    G = nx.cycle_graph(4)
    result = slater(G)
    assert result == 2

def test_sub_total_domination_number():
    G = nx.cycle_graph(4)
    result = sub_total_domination_number(G)
    assert result == 2

def test_annihilation_number():
    G = nx.cycle_graph(4)
    result = annihilation_number(G)
    assert result == 2

def test_residue():
    G = nx.cycle_graph(4)
    result = residue(G)
    assert result == 2

def test_harmonic_index():
    G = nx.cycle_graph(4)
    result = harmonic_index(G)
    assert result == 2
