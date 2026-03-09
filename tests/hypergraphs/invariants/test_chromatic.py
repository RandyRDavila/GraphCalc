# tests/hypergraphs/invariants/test_chromatic.py

import pytest

from graphcalc.hypergraphs.core.basics import Hypergraph
from graphcalc.hypergraphs.invariants.chromatic import (
    edge_coloring,
    edge_chromatic_number,
    strong_coloring,
    strong_chromatic_number,
    weak_coloring,
    weak_chromatic_number,
)


# =========================
# Helper validators
# =========================

def assert_valid_weak_coloring(H, coloring, k):
    assert set(coloring.keys()) == set(H.V)
    for v, c in coloring.items():
        assert isinstance(c, int)
        assert 0 <= c < k

    for edge in H.E:
        if len(edge) >= 2:
            colors = {coloring[v] for v in edge}
            assert len(colors) >= 2, f"Weak coloring made edge {edge} monochromatic."


def assert_valid_strong_coloring(H, coloring, k):
    assert set(coloring.keys()) == set(H.V)
    for v, c in coloring.items():
        assert isinstance(c, int)
        assert 0 <= c < k

    for edge in H.E:
        colors = [coloring[v] for v in edge]
        assert len(colors) == len(set(colors)), f"Strong coloring is not rainbow on edge {edge}."


def assert_valid_edge_coloring(H, coloring, k):
    assert set(coloring.keys()) == set(H.E)
    for e, c in coloring.items():
        assert isinstance(c, int)
        assert 0 <= c < k

    edges = list(H.E)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edges[i] & edges[j]:
                assert coloring[edges[i]] != coloring[edges[j]], (
                    f"Intersecting edges {edges[i]} and {edges[j]} received same color."
                )


# =========================
# Weak coloring
# =========================

def test_weak_coloring_empty_hypergraph():
    H = Hypergraph()
    assert weak_coloring(H, k=1) == {}
    assert weak_chromatic_number(H) == 0


def test_weak_coloring_no_edges():
    H = Hypergraph(V=[1, 2, 3], allow_singletons=True)
    coloring = weak_coloring(H, k=1)
    assert_valid_weak_coloring(H, coloring, 1)
    assert weak_chromatic_number(H) == 1


def test_weak_coloring_single_edge_of_size_two():
    H = Hypergraph(E=[{1, 2}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        weak_coloring(H, k=1)

    coloring = weak_coloring(H, k=2)
    assert_valid_weak_coloring(H, coloring, 2)
    assert weak_chromatic_number(H) == 2


def test_weak_coloring_single_edge_of_size_three():
    H = Hypergraph(E=[{1, 2, 3}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        weak_coloring(H, k=1)

    coloring = weak_coloring(H, k=2)
    assert_valid_weak_coloring(H, coloring, 2)
    assert weak_chromatic_number(H) == 2


def test_weak_coloring_graph_triangle():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {1, 3}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        weak_coloring(H, k=2)

    coloring = weak_coloring(H, k=3)
    assert_valid_weak_coloring(H, coloring, 3)
    assert weak_chromatic_number(H) == 3


def test_weak_coloring_graph_path():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {3, 4}])
    coloring = weak_coloring(H, k=2)
    assert_valid_weak_coloring(H, coloring, 2)
    assert weak_chromatic_number(H) == 2


def test_weak_coloring_ignores_singleton_edges():
    H = Hypergraph(allow_singletons=True, E=[{1}, {2}, {1, 2}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        weak_coloring(H, k=1)

    coloring = weak_coloring(H, k=2)
    assert_valid_weak_coloring(H, coloring, 2)
    assert weak_chromatic_number(H) == 2


def test_weak_coloring_ignores_empty_and_singleton_edges_when_no_real_constraint():
    H = Hypergraph(allow_empty=True, allow_singletons=True, E=[[], [1], [2]])
    coloring = weak_coloring(H, k=1)
    assert_valid_weak_coloring(H, coloring, 1)
    assert weak_chromatic_number(H) == 1


def test_weak_coloring_invalid_k_raises():
    H = Hypergraph(E=[{1, 2}])
    with pytest.raises(ValueError, match="positive integer"):
        weak_coloring(H, k=0)
    with pytest.raises(ValueError, match="positive integer"):
        weak_coloring(H, k=-3)


def test_weak_coloring_returns_all_vertices_even_if_isolated():
    H = Hypergraph(V=[1, 2, 3], E=[{1, 2}], auto_add_vertices=False)
    coloring = weak_coloring(H, k=2)
    assert set(coloring.keys()) == {1, 2, 3}
    assert_valid_weak_coloring(H, coloring, 2)


# =========================
# Strong coloring
# =========================

def test_strong_coloring_empty_hypergraph():
    H = Hypergraph()
    assert strong_coloring(H, k=1) == {}
    assert strong_chromatic_number(H) == 0


def test_strong_coloring_no_edges():
    H = Hypergraph(V=[1, 2, 3], allow_singletons=True)
    coloring = strong_coloring(H, k=1)
    assert_valid_strong_coloring(H, coloring, 1)
    assert strong_chromatic_number(H) == 1


def test_strong_coloring_single_edge_of_size_three():
    H = Hypergraph(E=[{1, 2, 3}])

    with pytest.raises(ValueError, match="size > 2|No optimal solution found"):
        strong_coloring(H, k=2)

    coloring = strong_coloring(H, k=3)
    assert_valid_strong_coloring(H, coloring, 3)
    assert strong_chromatic_number(H) == 3


def test_strong_coloring_graph_edge_equals_ordinary_vertex_coloring_on_graphs():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {3, 4}])
    coloring = strong_coloring(H, k=2)
    assert_valid_strong_coloring(H, coloring, 2)
    assert strong_chromatic_number(H) == 2


def test_strong_coloring_triangle():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {1, 3}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        strong_coloring(H, k=2)

    coloring = strong_coloring(H, k=3)
    assert_valid_strong_coloring(H, coloring, 3)
    assert strong_chromatic_number(H) == 3


def test_strong_coloring_detects_large_edge_lower_bound():
    H = Hypergraph(E=[{1, 2, 3, 4}])

    with pytest.raises(ValueError, match="size > 3"):
        strong_coloring(H, k=3)

    coloring = strong_coloring(H, k=4)
    assert_valid_strong_coloring(H, coloring, 4)
    assert strong_chromatic_number(H) == 4


def test_strong_coloring_with_singleton_edges():
    H = Hypergraph(allow_singletons=True, E=[{1}, {2}, {1, 2, 3}])

    with pytest.raises(ValueError, match="size > 2|No optimal solution found"):
        strong_coloring(H, k=2)

    coloring = strong_coloring(H, k=3)
    assert_valid_strong_coloring(H, coloring, 3)
    assert strong_chromatic_number(H) == 3


def test_strong_coloring_with_empty_edges_imposes_no_extra_constraint():
    H = Hypergraph(allow_empty=True, E=[[], [1, 2]])
    with pytest.raises(ValueError, match="size > 1"):
        strong_coloring(H, k=1)

    coloring = strong_coloring(H, k=2)
    assert_valid_strong_coloring(H, coloring, 2)
    assert strong_chromatic_number(H) == 2


def test_strong_coloring_invalid_k_raises():
    H = Hypergraph(E=[{1, 2}])
    with pytest.raises(ValueError, match="positive integer"):
        strong_coloring(H, k=0)
    with pytest.raises(ValueError, match="positive integer"):
        strong_coloring(H, k=-1)


def test_strong_coloring_returns_all_vertices_even_if_isolated():
    H = Hypergraph(V=[1, 2, 3], E=[{1, 2}], auto_add_vertices=False)
    coloring = strong_coloring(H, k=2)
    assert set(coloring.keys()) == {1, 2, 3}
    assert_valid_strong_coloring(H, coloring, 2)


# =========================
# Weak vs strong distinction
# =========================

def test_weak_and_strong_differ_on_single_3_edge():
    H = Hypergraph(E=[{1, 2, 3}])
    assert weak_chromatic_number(H) == 2
    assert strong_chromatic_number(H) == 3


def test_weak_and_strong_agree_on_graph_like_path():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {3, 4}])
    assert weak_chromatic_number(H) == 2
    assert strong_chromatic_number(H) == 2


# =========================
# Edge coloring
# =========================

def test_edge_coloring_empty_hypergraph():
    H = Hypergraph()
    assert edge_coloring(H, k=1) == {}
    assert edge_chromatic_number(H) == 0


def test_edge_coloring_single_edge():
    H = Hypergraph(E=[{1, 2, 3}])
    coloring = edge_coloring(H, k=1)
    assert_valid_edge_coloring(H, coloring, 1)
    assert edge_chromatic_number(H) == 1


def test_edge_coloring_two_disjoint_edges():
    H = Hypergraph(E=[{1, 2}, {3, 4}])
    coloring = edge_coloring(H, k=1)
    assert_valid_edge_coloring(H, coloring, 1)
    assert edge_chromatic_number(H) == 1


def test_edge_coloring_two_intersecting_edges():
    H = Hypergraph(E=[{1, 2}, {2, 3}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        edge_coloring(H, k=1)

    coloring = edge_coloring(H, k=2)
    assert_valid_edge_coloring(H, coloring, 2)
    assert edge_chromatic_number(H) == 2


def test_edge_coloring_three_pairwise_intersecting_edges():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {1, 3}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        edge_coloring(H, k=2)

    coloring = edge_coloring(H, k=3)
    assert_valid_edge_coloring(H, coloring, 3)
    assert edge_chromatic_number(H) == 3


def test_edge_coloring_star_family():
    H = Hypergraph(E=[{0, 1}, {0, 2}, {0, 3}, {0, 4}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        edge_coloring(H, k=3)

    coloring = edge_coloring(H, k=4)
    assert_valid_edge_coloring(H, coloring, 4)
    assert edge_chromatic_number(H) == 4


def test_edge_coloring_mixed_family():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {4, 5}])
    coloring = edge_coloring(H, k=2)
    assert_valid_edge_coloring(H, coloring, 2)
    assert edge_chromatic_number(H) == 2


def test_edge_coloring_with_empty_edge():
    H = Hypergraph(allow_empty=True, E=[[], [1, 2]])
    coloring = edge_coloring(H, k=1)
    assert_valid_edge_coloring(H, coloring, 1)
    assert edge_chromatic_number(H) == 1


def test_edge_coloring_with_singleton_edges():
    H = Hypergraph(allow_singletons=True, E=[{1}, {2}, {1, 2}])

    with pytest.raises(ValueError, match="No optimal solution found"):
        edge_coloring(H, k=1)

    coloring = edge_coloring(H, k=2)
    assert_valid_edge_coloring(H, coloring, 2)
    assert edge_chromatic_number(H) == 2


def test_edge_coloring_invalid_k_raises():
    H = Hypergraph(E=[{1, 2}])
    with pytest.raises(ValueError, match="positive integer"):
        edge_coloring(H, k=0)
    with pytest.raises(ValueError, match="positive integer"):
        edge_coloring(H, k=-2)


# =========================
# Wrapper consistency
# =========================

def test_weak_chromatic_number_matches_smallest_feasible_k():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    chi = weak_chromatic_number(H)
    assert chi == 2

    with pytest.raises(ValueError):
        weak_coloring(H, k=chi - 1)

    coloring = weak_coloring(H, k=chi)
    assert_valid_weak_coloring(H, coloring, chi)


def test_strong_chromatic_number_matches_smallest_feasible_k():
    H = Hypergraph(E=[{1, 2, 3}])
    chi = strong_chromatic_number(H)
    assert chi == 3

    with pytest.raises(ValueError):
        strong_coloring(H, k=chi - 1)

    coloring = strong_coloring(H, k=chi)
    assert_valid_strong_coloring(H, coloring, chi)


def test_edge_chromatic_number_matches_smallest_feasible_k():
    H = Hypergraph(E=[{1, 2}, {2, 3}, {3, 4}])
    chi_e = edge_chromatic_number(H)
    assert chi_e == 2

    with pytest.raises(ValueError):
        edge_coloring(H, k=chi_e - 1)

    coloring = edge_coloring(H, k=chi_e)
    assert_valid_edge_coloring(H, coloring, chi_e)


# =========================
# More edge-case constructions
# =========================

def test_single_vertex_no_edges():
    H = Hypergraph(V=[1], allow_singletons=True)
    assert weak_chromatic_number(H) == 1
    assert strong_chromatic_number(H) == 1
    assert weak_coloring(H, k=1) == {1: 0}
    assert strong_coloring(H, k=1) == {1: 0}


def test_singleton_edge_only():
    H = Hypergraph(allow_singletons=True, E=[{1}])
    assert weak_chromatic_number(H) == 1
    assert strong_chromatic_number(H) == 1
    assert edge_chromatic_number(H) == 1

    assert_valid_weak_coloring(H, weak_coloring(H, k=1), 1)
    assert_valid_strong_coloring(H, strong_coloring(H, k=1), 1)
    assert_valid_edge_coloring(H, edge_coloring(H, k=1), 1)


def test_empty_edge_only():
    H = Hypergraph(allow_empty=True, E=[[]])
    assert weak_chromatic_number(H) == 0
    assert strong_chromatic_number(H) == 0
    assert edge_chromatic_number(H) == 1

    assert weak_coloring(H, k=1) == {}
    assert strong_coloring(H, k=1) == {}
    assert_valid_edge_coloring(H, edge_coloring(H, k=1), 1)


def test_multiple_empty_edges_collapse_in_simple_representation():
    H = Hypergraph(allow_empty=True, E=[[], []])
    assert len(H.E) == 1
    assert edge_chromatic_number(H) == 1


def test_large_edge_plus_isolated_vertex():
    H = Hypergraph(V=[1, 2, 3, 4], E=[{1, 2, 3}], auto_add_vertices=False)
    assert weak_chromatic_number(H) == 2
    assert strong_chromatic_number(H) == 3

    assert_valid_weak_coloring(H, weak_coloring(H, k=2), 2)
    assert_valid_strong_coloring(H, strong_coloring(H, k=3), 3)
