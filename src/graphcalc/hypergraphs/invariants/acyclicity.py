from __future__ import annotations

from collections import deque
from typing import Optional

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.metadata import invariant_metadata

__all__ = [
    "is_alpha_acyclic",
    "berge_girth",
    "is_berge_acyclic",
]


@invariant_metadata(
    display_name="Alpha-acyclicity",
    notation=r"\alpha\text{-acyclic}(H)",
    category="acyclicity invariants",
    aliases=("is alpha acyclic", "alpha acyclic"),
    definition=(
        "A hypergraph is alpha-acyclic if it can be reduced to the empty hypergraph "
        "by repeated GYO reductions: deleting vertices contained in at most one hyperedge "
        "and deleting hyperedges contained in other hyperedges."
    ),
)
@require_hypergraph_like
def is_alpha_acyclic(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is alpha-acyclic.

    A hypergraph is **alpha-acyclic** if it can be reduced to the empty
    hypergraph by repeated application of the GYO reduction rules:

    1. remove any vertex that belongs to at most one hyperedge, deleting it
       from that hyperedge, and
    2. remove any hyperedge that is contained in another hyperedge.

    This notion is one of the standard acyclicity concepts for hypergraphs
    and database schemas.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Returns
    -------
    bool
        True if the GYO reduction eliminates all nonempty hyperedges.
        Otherwise False.

    Notes
    -----
    Empty hyperedges are ignored by this implementation when initializing the
    reduction process. This is consistent with viewing them as carrying no
    vertex-incidence structure for the GYO procedure.

    The algorithm works on mutable copies of the hyperedges and does not
    modify ``H``.

    Examples
    --------
    >>> import graphcalc.hypergraphs as hc
    >>> from graphcalc.hypergraphs.invariants.acyclicity import is_alpha_acyclic
    >>> H = hc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> is_alpha_acyclic(H)
    True
    """
    edges = [set(edge) for edge in H.E if len(edge) > 0]
    changed = True

    while changed:
        changed = False

        # Remove hyperedges contained in other hyperedges.
        edges.sort(key=len)
        to_remove = set()
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if edges[i].issubset(edges[j]):
                    to_remove.add(i)
                    break

        if to_remove:
            edges = [edge for idx, edge in enumerate(edges) if idx not in to_remove]
            changed = True
            if not edges:
                return True

        # Remove vertices of degree at most one.
        vertex_counts: dict[object, int] = {}
        for edge in edges:
            for v in edge:
                vertex_counts[v] = vertex_counts.get(v, 0) + 1

        removable_vertices = {v for v, count in vertex_counts.items() if count <= 1}
        if removable_vertices:
            new_edges = []
            for edge in edges:
                reduced = edge - removable_vertices
                if reduced:
                    new_edges.append(reduced)
                else:
                    changed = True
            edges = new_edges
            changed = True

    return len(edges) == 0


@invariant_metadata(
    display_name="Berge girth",
    notation=r"g_{\mathrm{B}}(H)",
    category="acyclicity invariants",
    aliases=("berge cycle girth",),
    definition=(
        "The Berge girth of a hypergraph is the minimum number of hyperedges in a Berge cycle. "
        "If the hypergraph has no Berge cycle, the Berge girth is undefined, and this function returns None."
    ),
)
@require_hypergraph_like
def berge_girth(H: HypergraphLike) -> Optional[int]:
    r"""
    Return the Berge girth of a hypergraph.

    The **Berge girth** is the minimum number of hyperedges in a Berge cycle
    of the hypergraph. If the hypergraph has no Berge cycle, this function
    returns ``None``.

    This implementation computes the girth through the incidence graph of the
    hypergraph. The incidence graph is the bipartite graph with one part
    corresponding to vertices, the other part corresponding to hyperedges,
    and an incidence edge joining :math:`v` to :math:`e` whenever
    :math:`v \in e`.

    A Berge cycle of length :math:`k` corresponds to a cycle of length
    :math:`2k` in the incidence graph. Therefore this function computes the
    length of the shortest cycle in the incidence graph and divides by two.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Returns
    -------
    int or None
        The length of the shortest Berge cycle, measured in number of
        hyperedges, or ``None`` if the hypergraph is Berge-acyclic.

    Notes
    -----
    - If the incidence graph is acyclic, then the hypergraph is Berge-acyclic.
    - The smallest possible cycle in a bipartite graph has length 4, so the
      smallest possible Berge cycle has length 2 under this convention.

    Examples
    --------
    >>> import graphcalc.hypergraphs as hc
    >>> from graphcalc.hypergraphs.invariants.acyclicity import berge_girth
    >>> H = hc.Hypergraph(E=[{1, 2}, {2, 3}, {1, 3}])
    >>> berge_girth(H)
    3
    """
    vertices = list(H.V)
    edges = list(H.E)
    n = len(vertices)
    m = len(edges)

    if n + m == 0:
        return None

    vertex_id = {v: i for i, v in enumerate(vertices)}

    # Nodes 0, ..., n-1 correspond to vertices.
    # Nodes n, ..., n+m-1 correspond to hyperedges.
    adjacency: list[list[int]] = [[] for _ in range(n + m)]
    for j, edge in enumerate(edges):
        edge_node = n + j
        for v in edge:
            if v in vertex_id:
                vertex_node = vertex_id[v]
                adjacency[vertex_node].append(edge_node)
                adjacency[edge_node].append(vertex_node)

    best_cycle_length: Optional[int] = None

    for start in range(n + m):
        dist = [-1] * (n + m)
        parent = [-1] * (n + m)

        q = deque([start])
        dist[start] = 0

        while q:
            u = q.popleft()
            for w in adjacency[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    q.append(w)
                elif parent[u] != w:
                    cycle_length = dist[u] + dist[w] + 1
                    if best_cycle_length is None or cycle_length < best_cycle_length:
                        best_cycle_length = cycle_length

        if best_cycle_length == 4:
            break

    if best_cycle_length is None:
        return None
    return best_cycle_length // 2


@invariant_metadata(
    display_name="Berge-acyclicity",
    notation=r"\text{Berge-acyclic}(H)",
    category="acyclicity invariants",
    aliases=("is berge acyclic", "berge acyclic"),
    definition=(
        "A hypergraph is Berge-acyclic if it contains no Berge cycle; equivalently, "
        "its incidence graph is acyclic."
    ),
)
@require_hypergraph_like
def is_berge_acyclic(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is Berge-acyclic.

    A hypergraph is **Berge-acyclic** if it contains no Berge cycle.
    Equivalently, its incidence graph is acyclic.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Returns
    -------
    bool
        True if ``H`` has no Berge cycle. Otherwise False.

    See Also
    --------
    berge_girth
        Returns the length of the shortest Berge cycle, or ``None`` if no such
        cycle exists.
    """
    return berge_girth(H) is None
