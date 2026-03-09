# src/graphcalc/hypergraphs/invariants/structure.py

from __future__ import annotations

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.metadata import invariant_metadata

__all__ = [
    "is_simple",
    "is_linear",
    "is_intersecting",
    "is_pair_covering",
    "is_sperner",
    "is_clutter",
    "is_t_intersecting",
]


@invariant_metadata(
    display_name="Simplicity",
    notation=r"\text{simple}(H)",
    category="structure properties",
    aliases=("is simple",),
    definition=(
        "A hypergraph H is simple if it has no repeated hyperedges."
    ),
)
@require_hypergraph_like
def is_simple(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is simple.
    """
    return len(H.E) == len(set(H.E))


@invariant_metadata(
    display_name="Pair-covering property",
    notation=r"\text{pair-covering}(H)",
    category="structure properties",
    aliases=("is pair covering", "2-covering"),
    definition=(
        "A hypergraph H is pair-covering if every pair of distinct vertices is contained in some hyperedge of H."
    ),
)
@require_hypergraph_like
def is_pair_covering(H: HypergraphLike) -> bool:
    r"""
    Return whether every pair of distinct vertices is contained in some hyperedge.
    """
    vertices = list(H.V)
    if len(vertices) < 2:
        return True

    covered_pairs = H.two_section_edges()
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if frozenset((vertices[i], vertices[j])) not in covered_pairs:
                return False
    return True


@invariant_metadata(
    display_name="Linearity",
    notation=r"\text{linear}(H)",
    category="structure properties",
    aliases=("is linear",),
    definition=(
        "A hypergraph H is linear if every two distinct hyperedges intersect in at most one vertex."
    ),
)
@require_hypergraph_like
def is_linear(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is linear.
    """
    edges = list(H.E)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if len(edges[i] & edges[j]) > 1:
                return False
    return True


@invariant_metadata(
    display_name="Intersecting property",
    notation=r"\text{intersecting}(H)",
    category="structure properties",
    aliases=("is intersecting",),
    definition=(
        "A hypergraph H is intersecting if every two distinct hyperedges have nonempty intersection."
    ),
)
@require_hypergraph_like
def is_intersecting(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is intersecting.
    """
    edges = list(H.E)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edges[i].isdisjoint(edges[j]):
                return False
    return True


@invariant_metadata(
    display_name="Sperner property",
    notation=r"\text{Sperner}(H)",
    category="structure properties",
    aliases=("is sperner",),
    definition=(
        "A hypergraph H is Sperner if no hyperedge is a proper subset of another hyperedge."
    ),
)
@require_hypergraph_like
def is_sperner(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is Sperner.
    """
    edges = list(H.E)
    for i in range(len(edges)):
        for j in range(len(edges)):
            if i == j:
                continue
            if edges[i] < edges[j]:
                return False
    return True


@invariant_metadata(
    display_name="Clutter property",
    notation=r"\text{clutter}(H)",
    category="structure properties",
    aliases=("is clutter",),
    definition=(
        "A hypergraph H is a clutter if no hyperedge properly contains another hyperedge; equivalently, its hyperedge family is an antichain under inclusion."
    ),
)
@require_hypergraph_like
def is_clutter(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is a clutter.

    A hypergraph is a **clutter** if no hyperedge properly contains another.
    Equivalently, the hyperedge family is an antichain under set inclusion.

    In many texts, this is the same notion as a **Sperner hypergraph**.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Returns
    -------
    bool
        True if for all distinct hyperedges :math:`e,f \in E(H)`, neither
        :math:`e \subset f` nor :math:`f \subset e` holds. Otherwise False.

    Notes
    -----
    Since the core hypergraph representation is simple, duplicate hyperedges
    are already excluded. Thus only proper containment needs to be checked.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> gc.is_clutter(H)
    True

    >>> H = gc.Hypergraph(allow_singletons=True, E=[{1}, {1, 2}])
    >>> gc.is_clutter(H)
    False
    """
    edges = list(H.E)
    for i in range(len(edges)):
        for j in range(len(edges)):
            if i == j:
                continue
            if edges[i] > edges[j]:
                return False
    return True


@invariant_metadata(
    display_name="t-intersecting property",
    notation=r"\text{$t$-intersecting}(H)",
    category="structure properties",
    aliases=("is t-intersecting",),
    definition=(
        "A hypergraph H is t-intersecting if every two distinct hyperedges intersect in at least t vertices."
    ),
)
@require_hypergraph_like
def is_t_intersecting(H: HypergraphLike, t: int = 1) -> bool:
    r"""
    Return whether a hypergraph is ``t``-intersecting.

    A hypergraph is **t-intersecting** if every two distinct hyperedges
    intersect in at least ``t`` vertices. That is,

    .. math::
        |e \cap f| \ge t
        \quad \text{for all distinct } e,f \in E(H).

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    t : int, default=1
        Required minimum intersection size.

    Returns
    -------
    bool
        True if every pair of distinct hyperedges intersects in at least
        ``t`` vertices. Otherwise False.

    Raises
    ------
    ValueError
        If ``t < 0``.

    Notes
    -----
    - ``t = 1`` recovers the usual notion of an intersecting hypergraph.
    - If ``t = 0``, every hypergraph is vacuously ``t``-intersecting.
    - Hypergraphs with at most one hyperedge are vacuously ``t``-intersecting
      for every ``t >= 0``.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> H = gc.Hypergraph(E=[{1, 2, 3}, {2, 3, 4}])
    >>> gc.is_t_intersecting(H, 2)
    True

    >>> gc.is_t_intersecting(H, 3)
    False
    """
    if t < 0:
        raise ValueError("t must be >= 0.")
    if t == 0:
        return True

    edges = list(H.E)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if len(edges[i] & edges[j]) < t:
                return False
    return True
