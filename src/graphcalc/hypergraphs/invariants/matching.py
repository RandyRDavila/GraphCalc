# src/graphcalc/hypergraphs/invariants/matching.py

from __future__ import annotations

from typing import FrozenSet, Hashable, Set

import pulp

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.solvers import with_solver
from graphcalc.utils import _extract_and_report
from graphcalc.metadata import invariant_metadata

__all__ = [
    "maximum_matching",
    "matching_number",
    "fractional_matching_number",
    "minimum_edge_cover",
    "edge_cover_number",
]


@invariant_metadata(
    display_name="Maximum matching",
    notation=r"M_{\max}(H)",
    category="matching invariants",
    aliases=("largest matching",),
    definition=(
        "A maximum matching of a hypergraph H is a family of pairwise disjoint hyperedges of maximum cardinality."
    ),
)
@require_hypergraph_like
@with_solver
def maximum_matching(
    H: HypergraphLike,
    *,
    verbose: bool = False,
    solve=None,
) -> Set[FrozenSet[Hashable]]:
    r"""
    Return a largest matching of hyperedges in a hypergraph.
    """
    if not H.E:
        return set()

    edges = list(H.E)

    prob = pulp.LpProblem("MaximumMatchingHypergraph", pulp.LpMaximize)
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(len(edges))}

    prob += pulp.lpSum(y[i] for i in range(len(edges)))

    for v in H.V:
        incident = [i for i, edge in enumerate(edges) if v in edge]
        if incident:
            prob += pulp.lpSum(y[i] for i in incident) <= 1, f"vertex_{repr(v)}"

    solve(prob)

    selected_indices = _extract_and_report(prob, y, verbose=verbose)
    return {edges[i] for i in selected_indices}


@invariant_metadata(
    display_name="Matching number",
    notation=r"\nu(H)",
    category="matching invariants",
    aliases=("hypergraph matching number",),
    definition=(
        "The matching number of a hypergraph H is the maximum cardinality of a matching in H."
    ),
)
@require_hypergraph_like
def matching_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the matching number of a hypergraph.
    """
    return len(maximum_matching(H, **solver_kwargs))


@invariant_metadata(
    display_name="Fractional matching number",
    notation=r"\nu^*(H)",
    category="matching invariants",
    aliases=("hypergraph fractional matching number",),
    definition=(
        "The fractional matching number of a hypergraph H is the maximum total weight assignable to hyperedges so that, for each vertex, the sum of weights of incident hyperedges is at most 1."
    ),
)
@require_hypergraph_like
@with_solver
def fractional_matching_number(
    H: HypergraphLike,
    *,
    verbose: bool = False,
    solve=None,
) -> float:
    r"""
    Return the fractional matching number of a hypergraph.
    """
    if not H.E:
        return 0.0

    edges = list(H.E)

    prob = pulp.LpProblem("FractionalMatchingNumberHypergraph", pulp.LpMaximize)
    w = {
        i: pulp.LpVariable(f"w_{i}", lowBound=0.0, upBound=1.0, cat="Continuous")
        for i in range(len(edges))
    }

    prob += pulp.lpSum(w[i] for i in range(len(edges)))

    for v in H.V:
        incident = [i for i, edge in enumerate(edges) if v in edge]
        if incident:
            prob += pulp.lpSum(w[i] for i in incident) <= 1, f"vertex_{repr(v)}"

    solve(prob)

    value = pulp.value(prob.objective)
    if verbose:
        status = pulp.LpStatus.get(prob.status, str(prob.status))
        print(f"Solver status : {status}")
        print(f"Objective     : {value}")

    return float(value if value is not None else 0.0)


@invariant_metadata(
    display_name="Minimum edge cover",
    notation=r"C_{\min}(H)",
    category="matching invariants",
    aliases=("minimum hyperedge cover",),
    definition=(
        "A minimum edge cover of a hypergraph H is an edge cover of minimum cardinality, where an edge cover is a family of hyperedges whose union contains every vertex of H."
    ),
)
@require_hypergraph_like
@with_solver
def minimum_edge_cover(
    H: HypergraphLike,
    *,
    verbose: bool = False,
    solve=None,
) -> Set[FrozenSet[Hashable]]:
    r"""
    Return a minimum edge cover of a hypergraph.
    """
    if not H.V:
        return set()

    isolated = [v for v in H.V if all(v not in edge for edge in H.E)]
    if isolated:
        raise ValueError(
            f"Hypergraph contains isolated vertices, so no edge cover exists: {isolated}"
        )

    edges = list(H.E)

    prob = pulp.LpProblem("MinimumEdgeCoverHypergraph", pulp.LpMinimize)
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(len(edges))}

    prob += pulp.lpSum(y[i] for i in range(len(edges)))

    for v in H.V:
        incident = [i for i, edge in enumerate(edges) if v in edge]
        prob += pulp.lpSum(y[i] for i in incident) >= 1, f"cover_vertex_{repr(v)}"

    solve(prob)

    selected_indices = _extract_and_report(prob, y, verbose=verbose)
    return {edges[i] for i in selected_indices}


@invariant_metadata(
    display_name="Edge cover number",
    notation=r"\rho(H)",
    category="matching invariants",
    aliases=("hypergraph edge cover number",),
    definition=(
        "The edge cover number of a hypergraph H is the minimum cardinality of an edge cover of H."
    ),
)
@require_hypergraph_like
def edge_cover_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the edge cover number of a hypergraph.
    """
    return len(minimum_edge_cover(H, **solver_kwargs))
