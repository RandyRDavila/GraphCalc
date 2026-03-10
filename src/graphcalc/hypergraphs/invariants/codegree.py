# src/graphcalc/hypergraphs/invariants/codegree.py

from __future__ import annotations

from itertools import combinations
from typing import FrozenSet, Hashable, Iterable, Set

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.metadata import invariant_metadata

__all__ = [
    "codegree",
    "maximum_codegree",
    "minimum_codegree",
    "average_codegree",
    "lower_shadow",
    "lower_shadow_size",
    "upper_shadow",
    "upper_shadow_size",
]


@invariant_metadata(
    display_name="Codegree",
    notation=r"d_H(S)",
    category="codegree invariants",
    aliases=("subset codegree",),
    definition=(
        "The codegree of a vertex subset S in a hypergraph H is the number of hyperedges of H that contain S."
    ),
)
@require_hypergraph_like
def codegree(H: HypergraphLike, S: Iterable[Hashable]) -> int:
    r"""
    Return the codegree of a vertex subset in a hypergraph.
    """
    subset = frozenset(S)
    return sum(1 for edge in H.E if subset.issubset(edge))


@invariant_metadata(
    display_name="Maximum t-codegree",
    notation=r"\Delta_t(H)",
    category="codegree invariants",
    aliases=("maximum codegree",),
    definition=(
        "The maximum t-codegree of a hypergraph H is the maximum, over all t-element vertex subsets S, "
        "of the number of hyperedges of H that contain S."
    ),
)
@require_hypergraph_like
def maximum_codegree(H: HypergraphLike, t: int = 2) -> int:
    r"""
    Return the maximum ``t``-codegree of a hypergraph.
    """
    if t < 0:
        raise ValueError("t must be >= 0")

    vertices = list(H.V)
    if t > len(vertices):
        return 0

    return max((codegree(H, subset) for subset in combinations(vertices, t)), default=0)


@invariant_metadata(
    display_name="Minimum t-codegree",
    notation=r"\delta_t(H)",
    category="codegree invariants",
    aliases=("minimum codegree",),
    definition=(
        "The minimum t-codegree of a hypergraph H is the minimum, over all t-element vertex subsets S, "
        "of the number of hyperedges of H that contain S."
    ),
)
@require_hypergraph_like
def minimum_codegree(H: HypergraphLike, t: int = 2) -> int:
    r"""
    Return the minimum ``t``-codegree of a hypergraph.
    """
    if t < 0:
        raise ValueError("t must be >= 0")

    vertices = list(H.V)
    if t > len(vertices):
        return 0

    return min((codegree(H, subset) for subset in combinations(vertices, t)), default=0)


@invariant_metadata(
    display_name="Average t-codegree",
    notation=r"\overline{d}_t(H)",
    category="codegree invariants",
    aliases=("mean codegree",),
    definition=(
        "The average t-codegree of a hypergraph H is the arithmetic mean of the codegrees of all t-element vertex subsets of H."
    ),
)
@require_hypergraph_like
def average_codegree(H: HypergraphLike, t: int = 2) -> float:
    r"""
    Return the average ``t``-codegree of a hypergraph.
    """
    if t < 0:
        raise ValueError("t must be >= 0")

    vertices = list(H.V)
    if t > len(vertices):
        return 0.0

    subsets = list(combinations(vertices, t))
    if not subsets:
        return 0.0

    total = sum(codegree(H, subset) for subset in subsets)
    return float(total / len(subsets))


@invariant_metadata(
    display_name="Lower shadow",
    notation=r"\partial^{-}(H)",
    category="shadow invariants",
    aliases=("down shadow",),
    definition=(
        "The lower shadow of a hypergraph H is the family of all sets obtained by deleting one vertex from a hyperedge of H."
    ),
)
@require_hypergraph_like
def lower_shadow(H: HypergraphLike) -> Set[FrozenSet[Hashable]]:
    r"""
    Return the lower shadow of a hypergraph.
    """
    shadow: Set[FrozenSet[Hashable]] = set()
    for edge in H.E:
        for v in edge:
            shadow.add(frozenset(edge.difference({v})))
    return shadow


@invariant_metadata(
    display_name="Lower shadow size",
    notation=r"|\partial^{-}(H)|",
    category="shadow invariants",
    aliases=("size of lower shadow",),
    definition=(
        "The lower shadow size of a hypergraph H is the number of distinct sets in the lower shadow of H."
    ),
)
@require_hypergraph_like
def lower_shadow_size(H: HypergraphLike) -> int:
    r"""
    Return the size of the lower shadow of a hypergraph.
    """
    return len(lower_shadow(H))


@invariant_metadata(
    display_name="Upper shadow",
    notation=r"\partial^{+}(H)",
    category="shadow invariants",
    aliases=("up shadow",),
    definition=(
        "The upper shadow of a hypergraph H, relative to a ground set, is the family of all sets obtained by adding one vertex "
        "from the ground set outside a hyperedge of H."
    ),
)
@require_hypergraph_like
def upper_shadow(
    H: HypergraphLike,
    *,
    ground_set: Iterable[Hashable] | None = None,
) -> Set[FrozenSet[Hashable]]:
    r"""
    Return the upper shadow of a hypergraph.
    """
    universe = set(H.V if ground_set is None else ground_set)
    shadow: Set[FrozenSet[Hashable]] = set()

    for edge in H.E:
        for v in universe.difference(edge):
            shadow.add(frozenset(set(edge) | {v}))

    return shadow


@invariant_metadata(
    display_name="Upper shadow size",
    notation=r"|\partial^{+}(H)|",
    category="shadow invariants",
    aliases=("size of upper shadow",),
    definition=(
        "The upper shadow size of a hypergraph H is the number of distinct sets in the upper shadow of H."
    ),
)
@require_hypergraph_like
def upper_shadow_size(
    H: HypergraphLike,
    *,
    ground_set: Iterable[Hashable] | None = None,
) -> int:
    r"""
    Return the size of the upper shadow of a hypergraph.
    """
    return len(upper_shadow(H, ground_set=ground_set))
