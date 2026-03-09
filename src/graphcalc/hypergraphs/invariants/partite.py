# src/graphcalc/hypergraphs/invariants/partite.py

from __future__ import annotations

from typing import Dict, Optional, Tuple

from graphcalc.hypergraphs.core.basics import Vertex
from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.metadata import invariant_metadata

__all__ = [
    "is_r_partite_r_uniform",
]


@invariant_metadata(
    display_name="r-partite r-uniform property",
    notation=r"\text{$r$-partite $r$-uniform}(H)",
    category="partite properties",
    aliases=("is r-partite r-uniform",),
    definition=(
        "A hypergraph H is r-partite and r-uniform if every hyperedge has size exactly r and the vertex set of H can be partitioned into r parts such that each hyperedge contains exactly one vertex from each part."
    ),
)
@require_hypergraph_like
def is_r_partite_r_uniform(
    H: HypergraphLike,
    r: int,
    *,
    exact_vertex_limit: int = 22,
) -> Tuple[bool, Optional[Dict[Vertex, int]]]:
    r"""
    Determine whether a hypergraph is ``r``-partite and ``r``-uniform.

    A hypergraph is **r-uniform** if every hyperedge has size exactly ``r``.

    A hypergraph is **r-partite** if its vertex set can be partitioned into
    ``r`` parts such that every hyperedge contains at most one vertex from
    each part. For an ``r``-uniform hypergraph, this is equivalent to saying
    that every hyperedge contains exactly one vertex from each part.

    This function checks both properties simultaneously and, if successful,
    returns a witness partition encoded as a map
    ``vertex -> part_index`` with values in ``{0, 1, ..., r-1}``.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    r : int
        Target uniformity and number of parts.
    exact_vertex_limit : int, default=22
        Maximum number of vertices allowed for the exact backtracking search.

    Returns
    -------
    tuple[bool, dict[Vertex, int] | None]
        A pair ``(is_partite, witness)`` where ``is_partite`` is True if and
        only if ``H`` is ``r``-partite and ``r``-uniform. If True, ``witness``
        is a partition map ``vertex -> part``. Otherwise ``witness`` is None.

    Raises
    ------
    ValueError
        If ``r <= 0``.
    ValueError
        If the number of vertices exceeds ``exact_vertex_limit``.

    Notes
    -----
    The recognition is done by exact backtracking on vertex labels.
    It is intended for small hypergraphs.

    The search orders vertices by decreasing degree to improve pruning.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> H = gc.Hypergraph(E=[{1, 3}, {2, 4}])
    >>> gc.is_r_partite_r_uniform(H, 2)
    (True, {1: 0, 2: 0, 3: 1, 4: 1})
    """
    if r <= 0:
        raise ValueError("r must be a positive integer.")

    vertices = list(H.V)
    n = len(vertices)

    if n > exact_vertex_limit:
        raise ValueError(f"r-partite check capped at n <= {exact_vertex_limit}.")

    edges = [set(edge) for edge in H.E]
    if any(len(edge) != r for edge in edges):
        return (False, None)

    vertex_id = {v: i for i, v in enumerate(vertices)}
    edge_indices = [[vertex_id[v] for v in edge] for edge in edges]

    part = [-1] * n

    degree = [0] * n
    for edge in edge_indices:
        for i in edge:
            degree[i] += 1

    order = sorted(range(n), key=lambda i: -degree[i])

    def ok_after_assign(v: int) -> bool:
        for edge in edge_indices:
            if v not in edge:
                continue
            seen = set()
            for u in edge:
                if part[u] != -1:
                    if part[u] in seen:
                        return False
                    seen.add(part[u])
        return True

    def dfs(pos: int) -> bool:
        if pos == n:
            return True

        v = order[pos]
        for c in range(r):
            part[v] = c
            if ok_after_assign(v) and dfs(pos + 1):
                return True
            part[v] = -1
        return False

    if not dfs(0):
        return (False, None)

    witness = {vertices[i]: part[i] for i in range(n)}
    return (True, witness)
