# src/graphcalc/hypergraphs/invariants/configurations.py

from __future__ import annotations

from itertools import combinations
from typing import FrozenSet, Hashable, Optional

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.metadata import invariant_metadata

__all__ = [
    "has_sunflower",
]


@invariant_metadata(
    display_name="Sunflower existence",
    notation=r"\text{has-sunflower}(H)",
    category="configuration properties",
    aliases=("contains sunflower", "contains delta-system"),
    definition=(
        "A hypergraph H has a sunflower with p petals if it contains distinct hyperedges "
        "e_1, ..., e_p such that there exists a set C with e_i ∩ e_j = C for all i != j. "
        "The set C is called the core."
    ),
)
@require_hypergraph_like
def has_sunflower(
    H: HypergraphLike,
    petals: int = 3,
    *,
    core_size: Optional[int] = None,
    exact_edge_limit: int = 22,
) -> Optional[tuple[FrozenSet[Hashable], list[FrozenSet[Hashable]]]]:
    r"""
    Search for a sunflower in the hyperedge family.

    A family of hyperedges :math:`e_1, \dots, e_p` is a **sunflower**
    (or **Delta-system**) if there exists a set :math:`C` such that

    .. math::
        e_i \cap e_j = C
        \quad \text{for all } i \ne j.

    The set :math:`C` is called the **core**, and the sets
    :math:`e_i \setminus C` are the **petals**.

    This function searches for a sunflower of a prescribed number of petals
    among the hyperedges of ``H``.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    petals : int, default=3
        Number of petals required in the sunflower.
    core_size : int or None, default=None
        If provided, require the sunflower core to have exactly this size.
    exact_edge_limit : int, default=22
        Maximum number of hyperedges allowed for this exact search.

    Returns
    -------
    tuple[frozenset, list[frozenset]] or None
        If a sunflower is found, returns ``(core, petal_edges)`` where
        ``core`` is the common intersection and ``petal_edges`` is a list of
        the chosen hyperedges. If no sunflower is found, returns ``None``.

    Raises
    ------
    ValueError
        If ``petals < 2``.
    ValueError
        If the number of hyperedges exceeds ``exact_edge_limit``.

    Notes
    -----
    This is an exact brute-force search over edge subsets of size ``petals``.
    It is intended for small hypergraphs.

    """
    edges = list(H.E)
    m = len(edges)

    if petals < 2:
        raise ValueError("petals must be >= 2.")
    if m > exact_edge_limit:
        raise ValueError(f"Sunflower search capped at m <= {exact_edge_limit}.")

    for idxs in combinations(range(m), petals):
        chosen = [edges[i] for i in idxs]

        core = set(chosen[0])
        for edge in chosen[1:]:
            core &= set(edge)

        if core_size is not None and len(core) != core_size:
            continue

        ok = True
        for i in range(petals):
            for j in range(i + 1, petals):
                if set(chosen[i]) & set(chosen[j]) != core:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            return (frozenset(core), chosen)

    return None
