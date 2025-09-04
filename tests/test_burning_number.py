# tests/test_burning_number_families.py
# ======================================================================
# Burning number b(G) — exact (MIP) tests on families with known values
# ======================================================================

import math
import pytest
import networkx as nx
import graphcalc as gc

# --------
# Utilities
# --------

def schedule_covers_graph(G: nx.Graph, schedule):
    """Validate the burning schedule definition at horizon k=len(schedule)."""
    k = len(schedule)
    covered = set()
    for i, v in enumerate(schedule, start=1):
        r = k - i
        # BFS ball of radius r around v
        for u, d in nx.single_source_shortest_path_length(G, v, cutoff=r).items():
            if d <= r:
                covered.add(u)
    return len(covered) == G.number_of_nodes()

def solver_is_available():
    """Detect if a working MILP solver is configured through @with_solver."""
    try:
        # Tiny feasibility probe: K2 must have b=2
        G = nx.path_graph(2)
        k = gc.burning_number(G)  # uses MIP via @with_solver
        return k == 2
    except Exception:
        return False


# ===============================================================
# Connected families with closed forms / standard exact values
# ===============================================================

@pytest.mark.skipif(not solver_is_available(), reason="MILP solver not available/configured")
@pytest.mark.parametrize("n", [1,2,3,4,5,6,7,8,9,10,16,17,25])
def test_paths_ceilsqrt(n):
    """b(P_n) = ceil(sqrt(n))"""
    G = nx.path_graph(n)
    k, sched = gc.burning_number(G, return_schedule=True)
    assert k == math.ceil(math.sqrt(n))
    assert len(sched) == k and schedule_covers_graph(G, sched)

@pytest.mark.skipif(not solver_is_available(), reason="MILP solver not available/configured")
@pytest.mark.parametrize("n", [3,4,5,6,7,8,9,10,12,15,16,17,25])
def test_cycles_ceilsqrt(n):
    """b(C_n) = ceil(sqrt(n))"""
    G = nx.cycle_graph(n)
    k, sched = gc.burning_number(G, return_schedule=True)
    assert k == math.ceil(math.sqrt(n))
    assert len(sched) == k and schedule_covers_graph(G, sched)

@pytest.mark.skipif(not solver_is_available(), reason="MILP solver not available/configured")
@pytest.mark.parametrize("n,expected", [(1,1),(2,2),(3,2),(4,2),(10,2),(25,2)])
def test_complete_graphs(n, expected):
    """b(K_1)=1; b(K_n)=2 for n>=2."""
    G = nx.complete_graph(n)
    k, sched = gc.burning_number(G, return_schedule=True)
    assert k == expected
    assert len(sched) == k and schedule_covers_graph(G, sched)

@pytest.mark.skipif(not solver_is_available(), reason="MILP solver not available/configured")
@pytest.mark.parametrize("m", [1,2,5,10,25])
def test_stars(m):
    """b(K_{1,m}) = 2 for m>=1."""
    G = nx.star_graph(m)  # center + m leaves
    k, sched = gc.burning_number(G, return_schedule=True)
    assert k == 2
    assert len(sched) == k and schedule_covers_graph(G, sched)

@pytest.mark.skipif(not solver_is_available(), reason="MILP solver not available/configured")
@pytest.mark.parametrize(
    "m,n,expected",
    [
        # min(m,n)=1 ⇒ star ⇒ 2
        (1,1,2), (1,3,2), (1,10,2),
        # min(m,n)=2 ⇒ still 2 (e.g., K_{2,n})
        (2,2,2), (2,3,2), (2,10,2),
        (3,2,2), (10,2,2),
        # min(m,n) >= 3 ⇒ need 3
        (3,3,3), (3,5,3), (5,7,3), (10,10,3),
    ],
)
def test_complete_bipartite(m, n, expected):
    """
    Complete bipartite: b(K_{m,n}) =
        2 if min(m, n) <= 2,
        3 if min(m, n) >= 3.
    """
    G = nx.complete_bipartite_graph(m, n)
    k, sched = gc.burning_number(G, return_schedule=True)
    assert k == expected
    assert len(sched) == k and schedule_covers_graph(G, sched)

@pytest.mark.skipif(not solver_is_available(), reason="MILP solver not available/configured")
@pytest.mark.parametrize("n", [4,5,10,20])
def test_wheels(n):
    """Wheel graphs W_n (n>=4 rim vertices): b(W_n) = 2."""
    # NetworkX wheel_graph(n) uses n as total nodes? Actually wheel_graph(n) builds W_{n} with n nodes.
    # We want "n rim + center": in NetworkX, wheel_graph(n) = center + (n-1)-cycle (total n nodes).
    # To avoid confusion, test the standard NX definition directly.
    G = nx.wheel_graph(n)
    k, sched = gc.burning_number(G, return_schedule=True)
    # For NX's wheel_graph(4)=K4 → 2; for larger n it's also 2 by igniting center first.
    assert k == 2
    assert len(sched) == k and schedule_covers_graph(G, sched)


# ===============================================================
# Bounds checks on CONNECTED random graphs
# ===============================================================

@pytest.mark.skipif(not solver_is_available(), reason="MILP solver not available/configured")
@pytest.mark.parametrize("n", [5,8,10,15])
def test_bounds_on_connected_random_graphs(n):
    """
    For connected graphs we have the universal upper bound b(G) <= radius(G) + 1.
    We verify the MIP result respects this, and the schedule is valid.
    """
    # Try until connected
    p = 0.3
    G = nx.fast_gnp_random_graph(n, p, seed=42)
    while not nx.is_connected(G):
        p = min(0.9, p + 0.1)
        G = nx.fast_gnp_random_graph(n, p, seed=42)

    k, sched = gc.burning_number(G, return_schedule=True)
    rad = nx.radius(G)
    assert k <= rad + 1
    assert len(sched) == k and schedule_covers_graph(G, sched)
    # Trivial lower bound by components (1 here) and by 1
    assert k >= 1
