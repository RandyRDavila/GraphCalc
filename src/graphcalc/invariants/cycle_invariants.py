from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Set, Tuple

import math
import itertools
import networkx as nx

__all__ = [
    "triangle_count",
    "cycle_rank",
    "girth",
    "circumference",
    "odd_girth",
    "even_girth",
    "feedback_vertex_set",
    "feedback_vertex_number",
    "maximum_number_of_vertex_disjoint_cycles",
    "decycling_number",
    "maximum_induced_forest_number",
]

def triangle_count(G):
    r"""
    Count (unordered) triangles in an undirected simple graph :math:`G`.

    A **triangle** is a 3-cycle (an induced copy of :math:`C_3`, equivalently a copy of
    :math:`K_3`). This function returns the number of distinct unordered triples
    :math:`\{u,v,w\}` that form a triangle in :math:`G`.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite simple undirected graphs).

    Notes
    -----
    :func:`networkx.triangles` returns a dictionary mapping each vertex :math:`v` to the
    number of triangles incident to :math:`v`. Summing these values counts each triangle
    exactly three times (once at each of its vertices), so we divide by 3.

    This interpretation is for undirected simple graphs. For directed graphs or
    multigraphs, the notion of a “triangle” and the behavior of
    :func:`networkx.triangles` may differ.

    Returns
    -------
    int
        The number of (unordered) triangles in :math:`G`.

    Examples
    --------
    A complete graph :math:`K_3` has exactly one triangle:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.complete_graph(3)
    >>> gc.triangle_count(G)
    1

    A complete graph :math:`K_4` has :math:`\\binom{4}{3} = 4` triangles:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.complete_graph(4)
    >>> gc.triangle_count(G)
    4

    A 4-cycle has no triangles:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(4)
    >>> gc.triangle_count(G)
    0

    Disjoint union: counts triangles across all components:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.disjoint_union(nx.complete_graph(3), nx.path_graph(4))
    >>> gc.triangle_count(G)
    1
    """
    return sum(nx.triangles(G).values()) // 3



def cycle_rank(G):
    r"""
    Compute the cyclomatic number (cycle rank) of an undirected graph :math:`G`.

    The **cyclomatic number** is
    :math:`r(G) = |E(G)| - |V(G)| + c(G)`,
    where :math:`c(G)` is the number of connected components of :math:`G`.

    For undirected graphs, :math:`r(G)` equals the dimension of the cycle space over
    :math:`\mathrm{GF}(2)` (the number of independent cycles). In particular, a forest
    has cycle rank 0.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite undirected graphs).

    Notes
    -----
    - If :math:`|V(G)| = 0`, we take :math:`c(G)=0`, so :math:`r(G)=0`.
    - This formula is for undirected graphs. Directed graphs use different notions of
      cycle rank / cycle space.

    Returns
    -------
    int
        The cyclomatic number :math:`r(G)`.

    Examples
    --------
    Trees and forests have cycle rank 0:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> T = nx.path_graph(5)
    >>> gc.cycle_rank(T)
    0

    A single cycle :math:`C_n` has cycle rank 1:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(6)
    >>> gc.cycle_rank(G)
    1

    A connected graph with two independent cycles has cycle rank 2:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 2)  # adds a chord, creating a second independent cycle
    >>> gc.cycle_rank(G)
    2

    Disconnected graphs add ranks across components:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(4))
    >>> gc.cycle_rank(G)
    2
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    c = nx.number_connected_components(G) if n > 0 else 0
    return m - n + c


def girth(G):
    r"""
    Compute the girth of an undirected graph :math:`G` (length of a shortest cycle).

    The **girth** :math:`g(G)` is the minimum length among all cycles in :math:`G`.

    Conventions
    -----------
    - If :math:`G` is acyclic (a forest), return ``math.inf``.
    - If :math:`|V(G)| < 3`, return ``math.inf``.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite undirected graphs).

    Notes
    -----
    Exact algorithm: run a BFS from each source vertex :math:`s`. Whenever an edge
    :math:`(u,v)` is encountered with :math:`v` already discovered and
    :math:`v \neq \mathrm{parent}[u]`, a cycle is detected with length

    .. math::
        \mathrm{dist}[u] + \mathrm{dist}[v] + 1.

    Taking the minimum over all sources yields the girth.

    This is correct for undirected **simple** graphs. For multigraphs, parallel edges
    create 2-cycles and self-loops create 1-cycles, which this routine does not
    explicitly handle.

    Complexity
    ----------
    :math:`O(n(n+m))` time where :math:`n=|V(G)|` and :math:`m=|E(G)|`.

    Returns
    -------
    int | float
        The length of a shortest cycle, or ``math.inf`` if :math:`G` has no cycles.

    Examples
    --------
    A tree has no cycles:

    >>> import math
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> T = nx.path_graph(6)
    >>> gc.girth(T) == math.inf
    True

    A triangle has girth 3:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(3)
    >>> gc.girth(G)
    3

    A 5-cycle has girth 5:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(5)
    >>> gc.girth(G)
    5

    Adding a chord creates a triangle, reducing the girth:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(5)
    >>> G.add_edge(0, 2)
    >>> gc.girth(G)
    3
    """
    n = G.number_of_nodes()
    if n < 3:
        return math.inf

    best = math.inf
    for s in G.nodes():
        dist = {s: 0}
        parent = {s: None}
        q = [s]
        for u in q:
            for v in G.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v:
                    best = min(best, dist[u] + dist[v] + 1)
        if best == 3:
            return 3
    return best


def odd_girth(G):
    r"""
    Compute the odd girth of an undirected graph :math:`G` (length of a shortest odd cycle).

    The **odd girth** is the minimum length among all **odd** cycles in :math:`G`.

    Conventions
    -----------
    - If :math:`G` has no odd cycle (equivalently, :math:`G` is bipartite), return
      ``math.inf``.
    - If :math:`|V(G)| < 3`, return ``math.inf``.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite undirected graphs).

    Notes
    -----
    This uses the same BFS-based cycle detection as :func:`girth`. For each source
    vertex :math:`s`, run BFS and whenever an edge :math:`(u,v)` is encountered with
    :math:`v` already discovered and :math:`v \neq \mathrm{parent}[u]`, a cycle is
    detected with length

    .. math::
        L = \mathrm{dist}[u] + \mathrm{dist}[v] + 1.

    We take the minimum such :math:`L` over all sources subject to :math:`L` being odd.

    As with :func:`girth`, this is intended for undirected **simple** graphs. Self-loops
    (odd cycle of length 1) and parallel edges (even cycle of length 2) in multigraphs
    are not handled explicitly.

    Complexity
    ----------
    :math:`O(n(n+m))` time where :math:`n=|V(G)|` and :math:`m=|E(G)|`.

    Returns
    -------
    int | float
        The length of a shortest odd cycle in :math:`G`, or ``math.inf`` if none exists.

    Examples
    --------
    Bipartite graphs have no odd cycles:

    >>> import math
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(4)
    >>> gc.odd_girth(G) == math.inf
    True

    A triangle has odd girth 3:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(3)
    >>> gc.odd_girth(G)
    3

    An odd cycle :math:`C_5` has odd girth 5:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(5)
    >>> gc.odd_girth(G)
    5

    If a graph has both even and odd cycles, only odd ones matter:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(6)
    >>> G.add_edge(0, 2)  # creates a triangle
    >>> gc.odd_girth(G)
    3
    """
    n = G.number_of_nodes()
    if n < 3:
        return math.inf

    best = math.inf
    for s in G.nodes():
        dist = {s: 0}
        parent = {s: None}
        q = [s]
        for u in q:
            for v in G.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v:
                    L = dist[u] + dist[v] + 1
                    if L % 2 == 1:
                        best = min(best, L)
        if best == 3:
            return 3
    return best


def even_girth(G):
    r"""
    Compute the even girth of an undirected graph :math:`G` (length of a shortest even cycle).

    The **even girth** is the minimum length among all **even** cycles in :math:`G`.

    Conventions
    -----------
    - If :math:`G` has no even cycle, return ``math.inf``.
    - If :math:`|V(G)| < 4`, return ``math.inf`` (in a simple graph the shortest even
      cycle has length 4).

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite undirected graphs).

    Notes
    -----
    This uses the same BFS-based cycle detection as :func:`girth`. For each source
    vertex :math:`s`, run BFS and whenever an edge :math:`(u,v)` is encountered with
    :math:`v` already discovered and :math:`v \neq \mathrm{parent}[u]`, a cycle is
    detected with length

    .. math::
        L = \mathrm{dist}[u] + \mathrm{dist}[v] + 1.

    We take the minimum such :math:`L` over all sources subject to :math:`L` being even.

    Intended for undirected **simple** graphs. In multigraphs, parallel edges create an
    even 2-cycle, so the even girth could be 2; this routine does not handle that case
    explicitly.

    Complexity
    ----------
    :math:`O(n(n+m))` time where :math:`n=|V(G)|` and :math:`m=|E(G)|`.

    Returns
    -------
    int | float
        The length of a shortest even cycle in :math:`G`, or ``math.inf`` if none exists.

    Examples
    --------
    A triangle has no even cycle:

    >>> import math
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(3)
    >>> gc.even_girth(G) == math.inf
    True

    A 4-cycle has even girth 4:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(4)
    >>> gc.even_girth(G)
    4

    An even cycle :math:`C_6` has even girth 6:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(6)
    >>> gc.even_girth(G)
    6

    If a graph has a 4-cycle anywhere, the even girth is 4:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.grid_2d_graph(2, 3)  # contains a 4-cycle
    >>> gc.even_girth(G)
    4
    """
    n = G.number_of_nodes()
    if n < 4:
        return math.inf

    best = math.inf
    for s in G.nodes():
        dist = {s: 0}
        parent = {s: None}
        q = [s]
        for u in q:
            for v in G.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v:
                    L = dist[u] + dist[v] + 1
                    if L % 2 == 0:
                        best = min(best, L)
        if best == 4:
            return 4
    return best


def circumference(G, max_n=16):
    r"""
    Compute the circumference of an undirected graph :math:`G` (length of a longest simple cycle).

    The **circumference** :math:`c(G)` is the maximum length of a simple cycle in :math:`G`.

    Conventions
    -----------
    - If :math:`G` has no cycles, return ``0``.
    - If :math:`|V(G)| < 3`, return ``0``.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (finite, undirected).
    max_n : int, default=16
        Safety cutoff on :math:`|V(G)|`. This routine is exponential-time.

    Returns
    -------
    int
        The circumference :math:`c(G)`, i.e., the length of a longest simple cycle, or ``0`` if
        :math:`G` is acyclic.

    Raises
    ------
    ValueError
        If :math:`|V(G)| >` ``max_n``.

    Notes
    -----
    This is an exact brute-force method intended only for very small graphs.

    The algorithm searches vertex subsets from large to small. For each :math:`k` and each
    subset :math:`S \subseteq V(G)` with :math:`|S|=k`, it considers the induced subgraph
    :math:`H = G[S]` and checks whether :math:`H` contains a simple cycle of length :math:`k`
    (i.e., a Hamiltonian cycle in :math:`H`).

    A quick certificate catches the case :math:`H \cong C_k`: if :math:`H` is connected, has
    exactly :math:`k` edges, and every vertex in :math:`H` has degree 2, then :math:`H` is a
    cycle and we immediately return :math:`k`.

    Otherwise, the implementation performs an exhaustive Hamiltonian-cycle check on :math:`H`.
    This can still be expensive on dense graphs, but is controlled by ``max_n``.

    Complexity
    ----------
    Exponential in :math:`n=|V(G)|` in the worst case (subset enumeration plus Hamiltonian checking).

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> T = nx.path_graph(6)
    >>> gc.circumference(T)
    0

    >>> G = nx.cycle_graph(7)
    >>> gc.circumference(G)
    7

    >>> G = nx.complete_graph(5)
    >>> gc.circumference(G)
    5

    >>> G = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(6))
    >>> gc.circumference(G)
    6
    """

    n = G.number_of_nodes()
    if n < 3:
        return 0
    if n > max_n:
        raise ValueError(f"circumference brute force intended for n <= {max_n}, got n={n}")

    if cycle_rank(G) == 0:
        return 0

    nodes = list(G.nodes())

    for k in range(n, 2, -1):
        for S in itertools.combinations(nodes, k):
            H = G.subgraph(S)

            # If H is exactly C_k, we can certify immediately.
            if nx.is_connected(H) and H.number_of_edges() == k and all(d == 2 for _, d in H.degree()):
                return k

            # Otherwise, brute-force search for a simple cycle of length k.
            # Warning: can explode on dense graphs.
            D = H.to_directed()
            for cyc in nx.simple_cycles(D):
                if len(cyc) == k:
                    return k

    return 0

def feedback_vertex_set(
    G: nx.Graph,
    *,
    exact: bool = True,
    time_limit_nodes: int = 60,
    return_size_only: bool = False,
) -> Any:
    r"""
    Compute a feedback vertex set (FVS) of an undirected graph :math:`G`.

    A **feedback vertex set** is a subset :math:`S \subseteq V(G)` that intersects every
    cycle of :math:`G`. Equivalently, removing :math:`S` makes the graph acyclic:

    .. math::
        S \text{ is an FVS} \iff G - S \text{ is a forest.}

    The **feedback vertex number** is the minimum size of such a set:

    .. math::
        \tau_V(G) = \min\{ |S| : G - S \text{ is acyclic} \}.

    This function can either:
    - compute a *minimum* FVS (exact mode) using branch-and-bound, or
    - compute a (usually small) FVS using a greedy heuristic.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for undirected simple graphs. (For MultiGraphs, self-loops
        and parallel edges create 1- and 2-cycles under multigraph conventions, and
        :func:`networkx.cycle_basis` is defined for simple undirected graphs.)
    exact : bool, default=True
        If ``True``, attempt to return a *minimum* feedback vertex set (exact :math:`\tau_V(G)`)
        via branch-and-bound. If ``False``, return a feedback vertex set found by a greedy
        heuristic (not guaranteed optimal).
    time_limit_nodes : int, default=60
        A size-based advisory threshold for exact search. (Note: in the current implementation,
        this value is not enforced when ``exact=True``; it is primarily a guardrail parameter
        for callers / future “auto” behavior.)
    return_size_only : bool, default=False
        If ``True``, return only the size (an ``int``). Otherwise return a vertex set.

    Returns
    -------
    set | int
        If ``return_size_only=False``, returns a set ``S`` that is a feedback vertex set.
        If ``exact=True``, ``S`` is minimum.
        If ``return_size_only=True``, returns ``|S|`` (which equals :math:`\tau_V(G)` in exact mode).

    Notes
    -----
    Exact mode (branch-and-bound)
    -----------------------------
    The exact solver repeatedly finds a cycle and branches on which vertex of that cycle to delete:

    1. If the graph is acyclic, return the empty set.
    2. Find any cycle :math:`C` (via :func:`networkx.cycle_basis` on a component).
    3. For each :math:`v \in C`, recursively solve on :math:`G - v` and take the best solution.

    Pruning uses:
    - an initial upper bound from the greedy heuristic, and
    - a lower bound from greedily packing vertex-disjoint cycles (each such cycle forces at least one vertex into any FVS).

    Heuristic mode
    --------------
    The greedy heuristic repeatedly computes a cycle basis in a cyclic component, removes the vertex
    that appears in the most basis cycles, and repeats until the graph becomes a forest.

    Caveats
    -------
    - This targets **undirected** graphs only.
    - Exact mode is exponential in the worst case (FVS is NP-hard) and can blow up on large/dense graphs. Use ``exact=False`` if you need a fast answer.

    Examples
    --------
    A cycle :math:`C_n` has :math:`\tau_V(C_n)=1`:

    >>> import networkx as nx
    >>> G = nx.cycle_graph(8)
    >>> S = feedback_vertex_set(G, exact=True)
    >>> len(S)
    1
    >>> nx.is_forest(G.subgraph(set(G) - S))
    True

    A complete graph :math:`K_n` has :math:`\tau_V(K_n)=n-2`:

    >>> G = nx.complete_graph(6)
    >>> len(feedback_vertex_set(G, exact=True))
    4

    Heuristic mode returns a valid (not necessarily minimum) FVS:

    >>> G = nx.complete_graph(7)
    >>> S = feedback_vertex_set(G, exact=False)
    >>> nx.is_forest(G.subgraph(set(G) - S))
    True

    See Also
    --------
    - Feedback edge set: for undirected graphs, the minimum feedback edge set size equals
      :math:`|E|-|V|+c` (cycle rank), whereas the feedback vertex set problem is different and NP-hard.
    """

    if G.is_directed():
        raise nx.NetworkXError("feedback_vertex_set currently supports undirected graphs only.")

    # Trivial cases
    if G.number_of_edges() == 0:
        return 0 if return_size_only else set()

    # If graph is already a forest, τ_V = 0
    if nx.is_forest(G):
        return 0 if return_size_only else set()

    # Guardrail: switch to heuristic unless user insists
    if (not exact) or (G.number_of_nodes() > time_limit_nodes and exact is False):
        S = _fvs_greedy(G)
        return len(S) if return_size_only else S

    # If the graph is big and exact=True, we still try; caller is knowingly asking for it.
    best_set = _fvs_branch_and_bound(G)
    return len(best_set) if return_size_only else best_set


def _any_cycle_nodes(H: nx.Graph) -> Optional[Tuple[Hashable, ...]]:
    """Return nodes of one simple cycle in H, or None if H is acyclic."""
    # nx.cycle_basis works per connected component; get any component with a cycle.
    for comp in nx.connected_components(H):
        sub = H.subgraph(comp)
        basis = nx.cycle_basis(sub)
        if basis:
            return tuple(basis[0])
    return None


def _lower_bound_disjoint_cycles(H: nx.Graph) -> int:
    """
    Greedy lower bound: pack vertex-disjoint cycles.
    Each packed cycle forces at least 1 vertex in any feedback vertex set.
    """
    H2 = H.copy()
    count = 0
    while True:
        cyc = _any_cycle_nodes(H2)
        if cyc is None:
            break
        count += 1
        H2.remove_nodes_from(cyc)  # enforce vertex-disjointness
    return count


def _fvs_branch_and_bound(G: nx.Graph) -> Set[Hashable]:
    """Exact minimum FVS via branch-and-bound on cycles."""
    # Upper bound from greedy (good initial pruning)
    best = _fvs_greedy(G)

    def rec(H: nx.Graph, chosen: Set[Hashable]) -> None:
        nonlocal best

        # Prune by current best
        if len(chosen) >= len(best):
            return

        # If acyclic, update best
        if nx.is_forest(H):
            best = set(chosen)
            return

        # Lower bound prune
        lb = _lower_bound_disjoint_cycles(H)
        if len(chosen) + lb >= len(best):
            return

        # Branch on a cycle
        cycle = _any_cycle_nodes(H)
        if cycle is None:
            best = set(chosen)
            return

        # A small heuristic ordering: branch on higher "cycle participation" first
        # to find good solutions earlier (tighter upper bounds).
        # Count occurrences in a cycle basis of that component.
        comp = next(c for c in nx.connected_components(H) if set(cycle).issubset(c))
        basis = nx.cycle_basis(H.subgraph(comp))
        score = {v: 0 for v in cycle}
        for C in basis:
            for v in C:
                if v in score:
                    score[v] += 1
        branch_vertices = sorted(cycle, key=lambda v: score.get(v, 0), reverse=True)

        for v in branch_vertices:
            H_next = H.copy()
            H_next.remove_node(v)
            chosen.add(v)
            rec(H_next, chosen)
            chosen.remove(v)

    rec(G.copy(), set())
    return best


def _fvs_greedy(G: nx.Graph) -> Set[Hashable]:
    """
    Greedy FVS:
    repeatedly remove the vertex that appears in the most cycles of a cycle basis.
    """
    H = G.copy()
    S: Set[Hashable] = set()

    while not nx.is_forest(H):
        cyc = _any_cycle_nodes(H)
        if cyc is None:
            break

        # Build a cycle basis in that component and count participation
        comp = next(c for c in nx.connected_components(H) if set(cyc).issubset(c))
        basis = nx.cycle_basis(H.subgraph(comp))
        counts = {}
        for C in basis:
            for v in C:
                counts[v] = counts.get(v, 0) + 1

        # If basis is empty for some reason, fall back to removing a node from cyc
        if not counts:
            v = cyc[0]
        else:
            v = max(counts, key=counts.get)

        S.add(v)
        H.remove_node(v)

    return S

def feedback_vertex_number(
    G: nx.Graph,
    *,
    exact: bool = True,
    time_limit_nodes: int = 60,
) -> int:
    r"""
    Compute the feedback vertex number :math:`\tau_V(G)` of an undirected graph :math:`G`.

    A **feedback vertex set (FVS)** is a subset :math:`S \subseteq V(G)` that intersects
    every cycle of :math:`G`. Equivalently, removing :math:`S` makes the graph acyclic:

    .. math::
        S \text{ is an FVS} \iff G - S \text{ is a forest.}

    The **feedback vertex number** is the minimum possible size of a feedback vertex set:

    .. math::
        \tau_V(G) = \min\{ |S| : G - S \text{ is acyclic} \}.

    This function returns :math:`\tau_V(G)` in exact mode (branch-and-bound) and a
    heuristic upper bound in heuristic mode.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for undirected simple graphs. (For MultiGraphs, self-loops
        and parallel edges can create 1- and 2-cycles under multigraph conventions; this
        routine is designed for simple graphs.)
    exact : bool, default=True
        If ``True``, compute :math:`\tau_V(G)` exactly via branch-and-bound. If ``False``,
        return the size of a feedback vertex set found by a greedy heuristic (not guaranteed
        optimal).
    time_limit_nodes : int, default=60
        A size-based advisory threshold. If ``exact=False`` and ``|V(G)|`` is large, this
        parameter is a reminder to prefer the heuristic. (Note: in the current implementation,
        exact mode is not automatically disabled based on this threshold.)

    Returns
    -------
    int
        The feedback vertex number :math:`\tau_V(G)` (exact mode) or the size of a valid
        feedback vertex set (heuristic mode).

    Notes
    -----
    In exact mode, the solver repeatedly finds a cycle (via :func:`networkx.cycle_basis`)
    and branches on which vertex of that cycle to delete, using:

    - an initial upper bound from the greedy heuristic, and
    - a lower bound from greedily packing vertex-disjoint cycles for pruning.

    The worst-case running time is exponential (the problem is NP-hard).

    Examples
    --------
    A cycle :math:`C_n` has :math:`\tau_V(C_n)=1`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(8)
    >>> gc.feedback_vertex_number(G)
    1

    A tree has no cycles:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> T = nx.path_graph(10)
    >>> gc.feedback_vertex_number(T)
    0

    A complete graph :math:`K_n` has :math:`\tau_V(K_n)=n-2`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.complete_graph(6)
    >>> gc.feedback_vertex_number(G)
    4

    Heuristic mode returns an upper bound (valid but not necessarily minimum):

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.complete_graph(9)
    >>> gc.feedback_vertex_number(G, exact=False) >= 7
    True
    """
    # Fast exits
    if G.is_directed():
        raise nx.NetworkXError("feedback_vertex_number supports undirected graphs only.")

    if G.number_of_edges() == 0 or nx.is_forest(G):
        return 0

    if exact:
        return len(_fvs_branch_and_bound(G))
    else:
        return len(_fvs_greedy(G))

def maximum_number_of_vertex_disjoint_cycles(G: nx.Graph) -> int:
    r"""
    Compute the maximum number of pairwise vertex-disjoint (simple) cycles in an undirected graph.

    A **cycle packing** is a collection of simple cycles :math:`C_1,\dots,C_t` such that no
    vertex appears in more than one cycle. This function returns

    .. math::
        \nu(G) = \max\{t : G \text{ contains } t \text{ vertex-disjoint cycles}\}.

    Relationship to feedback vertex sets
    ------------------------------------
    If :math:`\tau_V(G)` is the feedback vertex number (minimum size of a feedback vertex set),
    then

    .. math::
        \nu(G) \le \tau_V(G),

    because each vertex-disjoint cycle must contribute at least one distinct vertex to any
    feedback vertex set.

    Approach
    --------
    The algorithm is exact but exponential-time. It uses recursion with memoization over
    induced subgraphs represented by a bitmask of remaining vertices.

    In each recursive call, it picks a vertex :math:`v` that lies on some cycle and branches:

    - **Exclude** :math:`v` from the packing: delete :math:`v` and recurse.
    - **Include** :math:`v` in the packing: the packing must contain exactly one cycle through
      :math:`v`. Enumerate all simple cycles through :math:`v`, choose one, delete all vertices
      of that cycle, and recurse. Take the best result.

    This branching is exact because in any optimal packing, either :math:`v` is unused, or it
    lies on exactly one chosen cycle.

    Graph conventions
    -----------------
    This routine is intended for undirected **simple** graphs. If ``G`` is a multigraph, the
    implementation first converts it to a simple graph via ``nx.Graph(G)`` (collapsing parallel
    edges) and removes self-loops. The returned value therefore corresponds to the underlying
    simple graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Must be undirected.

    Returns
    -------
    int
        The maximum number of vertex-disjoint simple cycles in ``G``.

    Raises
    ------
    networkx.NetworkXError
        If ``G`` is directed.

    Examples
    --------
    A single cycle has packing number 1:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.maximum_number_of_vertex_disjoint_cycles(nx.cycle_graph(8))
    1

    Two disjoint triangles yield a packing of size 2:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    >>> gc.maximum_number_of_vertex_disjoint_cycles(G)
    2

    A tree has no cycles:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> T = nx.path_graph(10)
    >>> gc.maximum_number_of_vertex_disjoint_cycles(T)
    0
    """
    if G.is_directed():
        raise nx.NetworkXError("This function supports undirected graphs only.")
    H = nx.Graph(G)  # copy & simplify to simple graph container
    H.remove_edges_from(nx.selfloop_edges(H))
    if H.number_of_edges() == 0:
        return 0

    # Memoize by a bitmask of remaining vertices (n <= 30 typical).
    nodes = list(H.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    # adjacency bitmasks for fast induced-subgraph construction
    adj_mask = [0] * n
    for u, v in H.edges():
        iu, iv = idx[u], idx[v]
        adj_mask[iu] |= 1 << iv
        adj_mask[iv] |= 1 << iu

    memo: Dict[int, int] = {}

    def induced_cycle_exists(mask: int) -> bool:
        # quick acyclicity test via m - n + c; compute edges count in induced graph
        # for n <= 30, brute counting edges via bit operations is fine.
        vs = [i for i in range(n) if (mask >> i) & 1]
        if len(vs) < 3:
            return False
        # count edges
        e2 = 0
        for i in vs:
            e2 += (adj_mask[i] & mask).bit_count()
        m_ind = e2 // 2
        # compute components with DFS on bitmasks
        comp = 0
        unseen = mask
        while unseen:
            comp += 1
            start = (unseen & -unseen).bit_length() - 1
            stack = 1 << start
            unseen &= ~(1 << start)
            while stack:
                x = (stack & -stack).bit_length() - 1
                stack &= ~(1 << x)
                nbrs = adj_mask[x] & unseen
                unseen &= ~nbrs
                stack |= nbrs
        # cyclomatic number > 0  => has a cycle
        return m_ind - len(vs) + comp > 0

    def find_any_cycle(mask: int) -> Optional[List[int]]:
        # Find one cycle using networkx on the induced subgraph (fine for n <= 30).
        # We convert mask -> subgraph just for cycle discovery.
        vs = [nodes[i] for i in range(n) if (mask >> i) & 1]
        if len(vs) < 3:
            return None
        sub = H.subgraph(vs)
        basis = nx.cycle_basis(sub)
        if not basis:
            return None
        return [idx[v] for v in basis[0]]

    def cycles_through_vertex(mask: int, v_i: int) -> List[int]:
        """
        Return a list of cycle-vertex-masks for all simple cycles in the induced
        subgraph on `mask` that contain vertex v_i.

        Implementation: enumerate cycles by exploring simple paths from neighbors
        of v back to v. This can still be large in dense graphs, but n<=30 typical.
        """
        v = nodes[v_i]
        sub_nodes = [nodes[i] for i in range(n) if (mask >> i) & 1]
        sub = H.subgraph(sub_nodes)

        # Early exit if v not present
        if v not in sub:
            return []

        # Enumerate cycles containing v by enumerating pairs of neighbors (a,b)
        # and simple paths between them that avoid v, then adding edges (v,a),(v,b).
        nbrs = list(sub.neighbors(v))
        if len(nbrs) < 2:
            return []

        cycles_masks: Set[int] = set()
        # For each unordered pair of neighbors, enumerate all simple paths between them avoiding v.
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                a = nbrs[a_idx]
                b = nbrs[b_idx]
                # enumerate all simple paths a->b in sub with v removed
                sub_wo_v = sub.copy()
                sub_wo_v.remove_node(v)
                if a not in sub_wo_v or b not in sub_wo_v:
                    continue
                try:
                    for path in nx.all_simple_paths(sub_wo_v, a, b):
                        # cycle is v + path vertices
                        cyc_vs = set(path)
                        cyc_vs.add(v)
                        cm = 0
                        for u in cyc_vs:
                            cm |= 1 << idx[u]
                        cycles_masks.add(cm)
                except nx.NetworkXNoPath:
                    continue

        # Return as list; no need to include duplicate vertex-sets
        return list(cycles_masks)

    def rec(mask: int) -> int:
        if mask in memo:
            return memo[mask]
        if not induced_cycle_exists(mask):
            memo[mask] = 0
            return 0

        cyc = find_any_cycle(mask)
        if cyc is None:
            memo[mask] = 0
            return 0

        # Pick a vertex on a cycle to branch on
        v_i = cyc[0]

        # Branch 1: v_i unused -> delete it
        best = rec(mask & ~(1 << v_i))

        # Branch 2: v_i used -> pick a cycle through v_i, remove all its vertices
        for cyc_mask in cycles_through_vertex(mask, v_i):
            best = max(best, 1 + rec(mask & ~cyc_mask))

        memo[mask] = best
        return best

    full = (1 << n) - 1
    return rec(full)

def decycling_number(G: nx.Graph) -> int:
    r"""
    Compute the decycling number (feedback vertex number) of an undirected graph :math:`G`.

    The **decycling number** is the minimum number of vertices that must be removed to
    destroy all cycles. Equivalently, it is the size of a minimum **feedback vertex set**
    (FVS):

    .. math::
        \tau_V(G) = \min\{ |S| : S \subseteq V(G)\ \text{and}\ G - S\ \text{is a forest}\}.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for undirected simple graphs.

    Notes
    -----
    This function delegates to :func:`feedback_vertex_set` in exact mode and returns the
    size of the resulting set.

    Conventions:
    - If :math:`G` is a forest, then :math:`\tau_V(G)=0`.
    - For disconnected graphs, :math:`G-S` must be acyclic in every component.

    Complexity
    ----------
    Computing :math:`\tau_V(G)` is NP-hard in general; exact computation may take
    exponential time in the worst case.

    Returns
    -------
    int
        The decycling number :math:`\tau_V(G)`.

    Examples
    --------
    A tree has decycling number 0:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> T = nx.path_graph(8)
    >>> gc.decycling_number(T)
    0

    A cycle has decycling number 1:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(7)
    >>> gc.decycling_number(G)
    1

    A complete graph :math:`K_n` has :math:`\tau_V(K_n)=n-2`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.complete_graph(6)
    >>> gc.decycling_number(G)
    4

    See Also
    --------
    feedback_vertex_set : Compute an actual (minimum) feedback vertex set.
    maximum_induced_forest_number : Uses the identity :math:`|V(G)|-\tau_V(G)`.
    """
    return len(feedback_vertex_set(G, exact=True))


def maximum_induced_forest_number(G: nx.Graph) -> int:
    r"""
    Compute the maximum induced forest number of an undirected graph :math:`G`.

    The **maximum induced forest number** is the largest size of a vertex subset
    :math:`U \subseteq V(G)` such that the induced subgraph :math:`G[U]` is acyclic
    (a forest):

    .. math::
        f(G) = \max\{ |U| : U \subseteq V(G)\ \text{and}\ G[U]\ \text{is a forest}\}.

    Relationship to the decycling number
    ------------------------------------
    This invariant is the complement of the decycling number:

    .. math::
        f(G) = |V(G)| - \tau_V(G),

    since choosing a feedback vertex set :math:`S` with :math:`G-S` a forest is equivalent
    to choosing :math:`U = V(G)\setminus S` inducing a forest.

    This implementation uses that identity:
    ``f(G) = G.order() - decycling_number(G)``.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for undirected simple graphs.

    Notes
    -----
    Exact computation inherits the complexity of :func:`decycling_number`, which is NP-hard
    in general and may take exponential time.

    Returns
    -------
    int
        The maximum induced forest number :math:`f(G)`.

    Examples
    --------
    A forest keeps all vertices:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> T = nx.path_graph(8)
    >>> gc.maximum_induced_forest_number(T)
    8

    For a cycle :math:`C_n`, removing one vertex breaks all cycles:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(7)
    >>> gc.maximum_induced_forest_number(G)
    6

    For :math:`K_n`, the largest induced forest has size 2:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.complete_graph(6)
    >>> gc.maximum_induced_forest_number(G)
    2

    See Also
    --------
    decycling_number : The feedback vertex number :math:`\tau_V(G)`.
    feedback_vertex_set : Compute an actual optimal deletion set (when exact=True).
    """
    return G.order() - decycling_number(G)
