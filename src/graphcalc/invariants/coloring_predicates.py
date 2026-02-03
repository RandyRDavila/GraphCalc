import networkx as nx
import graphcalc as gc

__all__ = [
    "can_edge_color_with_k",
    "is_class1",
    "is_class2",
]

def _is_simple_graph(G):
    """True iff G is an undirected simple graph (no parallel edges; no direction)."""
    return (not getattr(G, "is_directed", lambda: False)()) and (not getattr(G, "is_multigraph", lambda: False)())

def can_edge_color_with_k(G, k):
    r"""
    Decide whether :math:`G` admits a proper edge-coloring with *at most* ``k`` colors.

    A **proper edge-coloring** is a map :math:`c : E(G)\to\{0,1,\dots,k-1\}` such that
    any two edges sharing an endpoint receive different colors. Equivalently, `c` is a
    proper vertex-coloring of the line graph :math:`L(G)`.

    This routine uses simple backtracking and is intended only for small graphs.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for undirected simple graphs).
    k : int
        The number of available colors.

    Notes
    -----
    - Necessary condition: ``k >= Δ(G)`` for any graph with at least one edge. This is
      checked for early rejection.
    - The implementation searches edges in a heuristic order (edges incident to
      higher-degree vertices first).
    - This is a decision procedure; it does not return a coloring.

    Conventions:
    - If ``k <= 0``, returns ``False``.
    - If ``|E(G)| = 0``, returns ``True`` (empty coloring).

    Complexity
    ----------
    Exponential in :math:`|E(G)|` in the worst case. Intended only for small graphs.

    Returns
    -------
    bool
        ``True`` iff :math:`G` has a proper edge-coloring using at most ``k`` colors.
    """
    if k <= 0:
        return False
    if G.number_of_edges() == 0:
        return True

    # Quick necessary condition: k >= Δ(G)
    if k < gc.max_degree(G):
        return False

    deg = dict(G.degree())
    edges = list(G.edges())
    edges.sort(
        key=lambda e: (deg[e[0]] + deg[e[1]], max(deg[e[0]], deg[e[1]])),
        reverse=True
    )

    # Track which colors are used on edges incident to each vertex
    used = {v: set() for v in G.nodes()}

    def backtrack(i):
        if i == len(edges):
            return True

        u, v = edges[i]
        forbidden = used[u] | used[v]

        for c in range(k):
            if c in forbidden:
                continue

            used[u].add(c)
            used[v].add(c)

            if backtrack(i + 1):
                return True

            used[u].remove(c)
            used[v].remove(c)

        return False

    return backtrack(0)

def is_class1(G, max_edges=40):
    r"""
    Test whether a simple graph is **Class 1** (edge-chromatic number equals maximum degree).

    For a simple undirected graph :math:`G`, the **edge-chromatic number**
    :math:`\chi'(G)` is the minimum number of colors in a proper edge-coloring.
    The graph is:

    - **Class 1** if :math:`\chi'(G) = \Delta(G)`,
    - **Class 2** if :math:`\chi'(G) = \Delta(G) + 1`.

    By Vizing's Theorem, every simple graph satisfies
    :math:`\chi'(G) \in \{\Delta(G), \Delta(G)+1\}`.

    This function decides Class 1 by:
    1. Returning ``True`` for bipartite graphs (Kőnig's line coloring theorem), and
    2. Otherwise running an exact backtracking test for a :math:`\Delta(G)`-edge-coloring.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Must be an undirected simple graph (``nx.Graph``).
    max_edges : int, default=40
        Safety cutoff to avoid exponential blowups in backtracking.

    Notes
    -----
    - If ``|E(G)| = 0`` then :math:`\chi'(G)=\Delta(G)=0`, so :math:`G` is Class 1.
    - Bipartite graphs satisfy :math:`\chi'(G)=\Delta(G)`.

    Returns
    -------
    bool
        ``True`` iff :math:`G` is Class 1, i.e., :math:`\chi'(G) = \Delta(G)`.

    Raises
    ------
    ValueError
        If ``G`` is not a simple undirected graph, or if ``|E(G)| > max_edges``.
    """
    if not _is_simple_graph(G):
        raise ValueError("Class 1/2 classification here is defined for undirected simple graphs (nx.Graph).")

    m = G.number_of_edges()
    if m > max_edges:
        raise ValueError(f"Edge-coloring backtracking capped at {max_edges} edges; got {m}.")

    if m == 0:
        return True

    Δ = gc.max_degree(G)
    if Δ == 0:
        return True

    # Kőnig's line coloring theorem: bipartite graphs have χ'(G)=Δ(G)
    if nx.is_bipartite(G):
        return True

    return can_edge_color_with_k(G, Δ)


def is_class2(G, max_edges=40):
    r"""
    Test whether a simple graph is **Class 2** (edge-chromatic number equals :math:`\Delta(G)+1`).

    For simple graphs, Vizing's theorem implies:
    :math:`G` is Class 2  if and only if  :math:`G` is not Class 1.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Must be an undirected simple graph (``nx.Graph``).
    max_edges : int, default=40
        Passed through to :func:`is_class1` for the backtracking cutoff.

    Returns
    -------
    bool
        ``True`` iff :math:`G` is Class 2, i.e., :math:`\chi'(G) = \Delta(G) + 1`.

    Raises
    ------
    ValueError
        If ``G`` is not a simple undirected graph, or if ``|E(G)| > max_edges``.
    """
    return not is_class1(G, max_edges=max_edges)
