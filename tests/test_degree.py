import pytest
import networkx as nx
from graphcalc.generators.simple_graphs import (
    complete_graph,
    cycle_graph,
    path_graph,
    star_graph,
)
from graphcalc.invariants.degree import (
    degree,
    degree_sequence,
    average_degree,
    maximum_degree,
    minimum_degree,
    sub_k_domination_number,
    slater,
    sub_total_domination_number,
    annihilation_number,
    residue,
    harmonic_index,
    elimination_sequence_from_degrees,
    k_residue_from_degrees,
    residue_from_degrees,
    k_residue,
)

@pytest.mark.parametrize("G, node, expected", [
    (complete_graph(4), 0, 3),  # Complete graph: degree is n-1
    (path_graph(4), 1, 2),  # Path graph: middle node degree is 2
    (cycle_graph(4), 2, 2),  # Cycle graph: all nodes degree is 2
    (star_graph(4), 0, 4),  # Star graph: center node degree is n
    (star_graph(4), 1, 1),  # Star graph: leaf node degree is 1
])
def test_degree(G, node, expected):
    assert degree(G, node) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), [3, 3, 3, 3]),  # All nodes same degree
    (path_graph(4), [2, 2, 1, 1]),  # Endpoints degree 1, middle 2
    (cycle_graph(4), [2, 2, 2, 2]),  # All nodes degree 2
    (star_graph(4), [4, 1, 1, 1, 1]),  # Center 4, leaves 1
])
def test_degree_sequence(G, expected):
    assert degree_sequence(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 3),  # Degree is consistent across all nodes
    (path_graph(4), 1.5),  # Average of [1, 2, 2, 1]
    (cycle_graph(4), 2),  # All nodes degree 2
    (star_graph(4), 1.6),  # Average of [4, 1, 1, 1, 1]
])
def test_average_degree(G, expected):
    assert average_degree(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 3),  # Max degree is consistent
    (path_graph(4), 2),  # Max degree of path graph
    (cycle_graph(4), 2),  # All nodes degree 2
    (star_graph(4), 4),  # Center node degree
])
def test_maximum_degree(G, expected):
    assert maximum_degree(G) == expected

@pytest.mark.parametrize("G, expected", [
    (complete_graph(4), 3),  # Min degree is consistent
    (path_graph(4), 1),  # Endpoints degree
    (cycle_graph(4), 2),  # All nodes degree 2
    (star_graph(4), 1),  # Leaf node degree
])
def test_minimum_degree(G, expected):
    assert minimum_degree(G) == expected

# def test_singleton_graph():
#     G = path_graph(1)
#     assert degree(G, 0) == 0  # Single node, no edges
#     assert degree_sequence(G) == [0]  # Degree of single node
#     assert average_degree(G) == 0  # No edges
#     assert maximum_degree(G) == 0  # Single node, max degree 0
#     assert minimum_degree(G) == 0  # Single node, min degree 0


@pytest.mark.parametrize("G, k, expected", [
    (cycle_graph(4), 1, 2),  # Cycle graph
    (path_graph(4), 1, 2),  # Path graph
    (star_graph(4), 1, 1),  # Star graph, k = 2
    (complete_graph(4), 1, 1),  # Complete graph, k = 2
])
def test_sub_k_domination_number(G, k, expected):
    assert sub_k_domination_number(G, k) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 1),  # Star graph
    (complete_graph(4), 1),  # Complete graph
])
def test_slater(G, expected):
    assert slater(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 2),  # Star graph
    (complete_graph(4), 2),  # Complete graph
])
def test_sub_total_domination_number(G, expected):
    assert sub_total_domination_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 4),  # Star graph
    (complete_graph(4), 2),  # Complete graph
])
def test_annihilation_number(G, expected):
    assert annihilation_number(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (path_graph(4), 2),  # Path graph
    (star_graph(4), 4),  # Star graph
    (complete_graph(4), 1),  # Complete graph
])
def test_residue(G, expected):
    assert residue(G) == expected

@pytest.mark.parametrize("G, expected", [
    (cycle_graph(4), 2),  # Cycle graph
    (complete_graph(4), 2),  # Complete graph
])
def test_harmonic_index(G, expected):
    assert harmonic_index(G) == pytest.approx(expected, rel=1e-2)

# --- Helpers for known graphs and their expected elimination sequences ---
def degrees(G):
    # Replace with gc.degree_sequence(G) if you prefer your function
    return sorted([d for _, d in G.degree()], reverse=True)

KNOWN_E = {
    "K4":        [3, 2, 1, 0],
    "P4":        [2, 1, 0, 0],            # path on 4
    "C5":        [2, 2, 1, 0, 0],         # 5-cycle
    "Star5":     [4, 0, 0, 0, 0],         # K_{1,4}
    "Empty5":    [0, 0, 0, 0, 0],
    "C6":        [2, 2, 1, 1, 0, 0],
}

# Precomputed exact k-residues from the formula R_k = sum_{i=0}^{k-1} (k-i) f_i(E)
EXPECTED_RK = {
    "K4":    {1: 1, 2: 3, 3: 6, 4: 10},
    "P4":    {1: 2, 2: 5, 3: 9, 4: 13},
    "C5":    {1: 2, 2: 5, 3: 10, 4: 15},
    "Star5": {1: 4, 2: 8, 3: 12, 4: 16},
    "Empty5":{1: 5, 2: 10, 3: 15, 4: 20},
    "C6":    {1: 2, 2: 6, 3: 12, 4: 18},
}

# --- Graph constructors for the above labels ---
def make_graph(label):
    if label == "K4":
        return nx.complete_graph(4)
    if label == "P4":
        return nx.path_graph(4)
    if label == "C5":
        return nx.cycle_graph(5)
    if label == "Star5":
        return nx.star_graph(4)  # K_{1,4}
    if label == "Empty5":
        G = nx.Graph()
        G.add_nodes_from(range(5))
        return G
    if label == "C6":
        return nx.cycle_graph(6)
    raise ValueError(f"Unknown graph label: {label}")

# -------------------------- Tests --------------------------

@pytest.mark.parametrize("label", list(KNOWN_E.keys()))
def test_elimination_sequence_matches_known(label):
    G = make_graph(label)
    E = elimination_sequence_from_degrees(degrees(G))
    assert E == KNOWN_E[label]

@pytest.mark.parametrize("label", list(KNOWN_E.keys()))
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_k_residue_from_degrees_matches_expected(label, k):
    G = make_graph(label)
    dseq = degrees(G)
    rk = k_residue_from_degrees(dseq, k)
    assert rk == EXPECTED_RK[label][k]

@pytest.mark.parametrize("label", list(KNOWN_E.keys()))
def test_residue_from_degrees_equals_f0(label):
    G = make_graph(label)
    E = elimination_sequence_from_degrees(degrees(G))
    R = residue_from_degrees(degrees(G))
    assert R == E.count(0)

@pytest.mark.parametrize("label", list(KNOWN_E.keys()))
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_graph_wrapper_equals_degree_version(label, k):
    G = make_graph(label)
    dseq = degrees(G)
    assert k_residue(G, k) == k_residue_from_degrees(dseq, k)

@pytest.mark.parametrize("label", list(KNOWN_E.keys()))
def test_k_equals_1_matches_classical_residue(label):
    G = make_graph(label)
    # If you expose a classical residue(G) function, also assert equality here:
    R1 = k_residue(G, 1)
    R0 = residue_from_degrees(degrees(G))
    assert R1 == R0

def test_k_residue_monotone_in_k():
    G = nx.cycle_graph(8)  # any nontrivial graph
    dseq = degrees(G)
    vals = [k_residue_from_degrees(dseq, k) for k in range(1, 6)]
    # Nondecreasing in k
    assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))

def test_isomorphism_invariance():
    G = nx.path_graph(6)
    # relabel nodes
    mapping = {i: (i*7) % 6 for i in range(6)}
    H = nx.relabel_nodes(G, mapping)
    for k in (1, 2, 3):
        assert k_residue(G, k) == k_residue(H, k)

def test_invalid_k_raises():
    G = nx.path_graph(4)
    with pytest.raises(ValueError):
        _ = k_residue(G, 0)
    with pytest.raises(ValueError):
        _ = k_residue_from_degrees(degrees(G), 0)

def test_non_graphical_degree_sequence_raises():
    # This sequence is not graphical (violates Erdős–Gallai)
    bad = [4, 4, 1, 0]
    with pytest.raises(ValueError):
        _ = elimination_sequence_from_degrees(bad)
    with pytest.raises(ValueError):
        _ = k_residue_from_degrees(bad, 2)

def test_empty_graph_cases():
    G = nx.Graph()
    # empty
    assert k_residue(G, 1) == 0
    assert elimination_sequence_from_degrees([]) == []
    assert residue_from_degrees([]) == 0
