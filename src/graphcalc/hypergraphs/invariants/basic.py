# src/graphcalc/hypergraphs/invariants/basic.py

from __future__ import annotations

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.metadata import invariant_metadata

__all__ = [
    "number_of_vertices",
    "number_of_edges",
    "is_empty",
    "is_trivial",
    "rank",
    "co_rank",
    "is_k_uniform",
    "maximum_degree",
    "minimum_degree",
    "average_degree",
    "degree_sequence",
    "edge_size_sequence",
    "is_regular",
    "is_d_regular",
]


@invariant_metadata(
    display_name="Number of vertices",
    notation=r"n(H)",
    category="basic invariants",
    aliases=("order", "vertex count"),
    definition=(
        "The number of vertices of a hypergraph H is the cardinality of its vertex set."
    ),
)
@require_hypergraph_like
def number_of_vertices(H: HypergraphLike) -> int:
    r"""
    Return the number of vertices of a hypergraph.
    """
    return len(H.V)


@invariant_metadata(
    display_name="Number of edges",
    notation=r"m(H)",
    category="basic invariants",
    aliases=("size", "edge count", "number of hyperedges"),
    definition=(
        "The number of edges of a hypergraph H is the cardinality of its hyperedge set."
    ),
)
@require_hypergraph_like
def number_of_edges(H: HypergraphLike) -> int:
    r"""
    Return the number of hyperedges of a hypergraph.
    """
    return len(H.E)


@invariant_metadata(
    display_name="Emptiness",
    notation=r"\text{empty}(H)",
    category="basic properties",
    aliases=("is empty",),
    definition=(
        "A hypergraph is empty if it has no hyperedges."
    ),
)
@require_hypergraph_like
def is_empty(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph has no hyperedges.
    """
    return len(H.E) == 0


@require_hypergraph_like
def is_trivial(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph has at most one vertex.
    """
    return len(H.V) <= 1


@require_hypergraph_like
def degree_sequence(H: HypergraphLike, *, nonincreasing: bool = True) -> list[int]:
    r"""
    Return the vertex degree sequence of a hypergraph.
    """
    seq = list(H.degrees().values())
    return sorted(seq, reverse=nonincreasing)


@require_hypergraph_like
def edge_size_sequence(H: HypergraphLike, *, nonincreasing: bool = True) -> list[int]:
    r"""
    Return the hyperedge-size sequence of a hypergraph.
    """
    seq = [len(edge) for edge in H.E]
    return sorted(seq, reverse=nonincreasing)


@require_hypergraph_like
def is_regular(H: HypergraphLike) -> bool:
    r"""
    Return whether a hypergraph is regular.
    """
    deg = list(H.degrees().values())
    return len(set(deg)) <= 1


@require_hypergraph_like
def is_d_regular(H: HypergraphLike, d: int) -> bool:
    r"""
    Return whether a hypergraph is ``d``-regular.
    """
    return all(deg == d for deg in H.degrees().values())


@invariant_metadata(
    display_name="Rank",
    notation=r"r(H)",
    category="basic invariants",
    aliases=("maximum edge size",),
    definition=(
        "The rank of a hypergraph H is the maximum cardinality of a hyperedge of H."
    ),
)
@require_hypergraph_like
def rank(H: HypergraphLike) -> int:
    r"""
    Return the rank of a hypergraph.
    """
    return max((len(edge) for edge in H.E), default=0)


@invariant_metadata(
    display_name="Co-rank",
    notation=r"c(H)",
    category="basic invariants",
    aliases=("minimum edge size",),
    definition=(
        "The co-rank of a hypergraph H is the minimum cardinality of a hyperedge of H."
    ),
)
@require_hypergraph_like
def co_rank(H: HypergraphLike) -> int:
    r"""
    Return the co-rank of a hypergraph.
    """
    return min((len(edge) for edge in H.E), default=0)


@require_hypergraph_like
def is_k_uniform(H: HypergraphLike, k: int) -> bool:
    r"""
    Return whether a hypergraph is ``k``-uniform.
    """
    return all(len(edge) == k for edge in H.E)


@invariant_metadata(
    display_name="Maximum degree",
    notation=r"\Delta(H)",
    category="degree invariants",
    aliases=("max degree",),
    definition=(
        "The maximum degree of a hypergraph H is the largest number of hyperedges incident with any vertex of H."
    ),
)
@require_hypergraph_like
def maximum_degree(H: HypergraphLike) -> int:
    r"""
    Return the maximum vertex degree of a hypergraph.
    """
    return max(H.degrees().values(), default=0)


@invariant_metadata(
    display_name="Minimum degree",
    notation=r"\delta(H)",
    category="degree invariants",
    aliases=("min degree",),
    definition=(
        "The minimum degree of a hypergraph H is the smallest number of hyperedges incident with any vertex of H."
    ),
)
@require_hypergraph_like
def minimum_degree(H: HypergraphLike) -> int:
    r"""
    Return the minimum vertex degree of a hypergraph.
    """
    return min(H.degrees().values(), default=0)


@invariant_metadata(
    display_name="Average degree",
    notation=r"\overline{d}(H)",
    category="degree invariants",
    aliases=("mean degree",),
    definition=(
        "The average degree of a hypergraph H is the arithmetic mean of the degrees of its vertices."
    ),
)
@require_hypergraph_like
def average_degree(H: HypergraphLike) -> float:
    r"""
    Return the average vertex degree of a hypergraph.
    """
    if not H.V:
        return 0.0
    deg = H.degrees()
    return float(sum(deg.values()) / len(H.V))
