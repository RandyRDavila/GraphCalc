# src/graphcalc/hypergraphs/invariants/transversals.py

from __future__ import annotations

from typing import Dict, Hashable, Optional, Set

import pulp

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.solvers import with_solver
from graphcalc.utils import _extract_and_report
from graphcalc.metadata import invariant_metadata

__all__ = [
    "minimum_transversal",
    "transversal_number",
    "fractional_transversal_number",
]


def _check_no_empty_edges_for_transversal(H: HypergraphLike) -> None:
    """
    Validate that a hypergraph admits a transversal.
    """
    for edge in H.E:
        if len(edge) == 0:
            raise ValueError(
                "Hypergraph contains an empty hyperedge; no transversal exists."
            )


@invariant_metadata(
    display_name="Minimum transversal",
    notation=r"T_{\min}(H)",
    category="transversal invariants",
    aliases=("minimum hitting set",),
    definition=(
        "A minimum transversal of a hypergraph H is a transversal of minimum cardinality, "
        "where a transversal is a vertex set that intersects every hyperedge of H."
    ),
)
@require_hypergraph_like
@with_solver
def minimum_transversal(
    H: HypergraphLike,
    *,
    weights: Optional[Dict[Hashable, float]] = None,
    verbose: bool = False,
    solve=None,
) -> Set[Hashable]:
    r"""
    Return a minimum transversal of a hypergraph.
    """
    _check_no_empty_edges_for_transversal(H)

    if not H.E:
        return set()

    vertices = list(H.V)
    w = {v: float(weights.get(v, 1.0)) if weights is not None else 1.0 for v in vertices}

    prob = pulp.LpProblem("MinimumTransversal", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{i}", cat="Binary") for i, v in enumerate(vertices)}

    prob += pulp.lpSum(w[v] * x[v] for v in vertices)

    for i, edge in enumerate(H.E):
        prob += pulp.lpSum(x[v] for v in edge) >= 1, f"hit_edge_{i}"

    solve(prob)

    return _extract_and_report(prob, x, verbose=verbose)


@invariant_metadata(
    display_name="Transversal number",
    notation=r"\tau(H)",
    category="transversal invariants",
    aliases=("hitting number", "vertex cover number"),
    definition=(
        "The transversal number of a hypergraph H is the minimum cardinality of a transversal of H."
    ),
)
@require_hypergraph_like
def transversal_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int | float:
    r"""
    Return the transversal number of a hypergraph.
    """
    weights = solver_kwargs.get("weights", None)
    T = minimum_transversal(H, **solver_kwargs)

    if weights is None:
        return len(T)

    return float(sum(float(weights.get(v, 1.0)) for v in T))


@invariant_metadata(
    display_name="Fractional transversal number",
    notation=r"\tau^*(H)",
    category="transversal invariants",
    aliases=("fractional hitting number",),
    definition=(
        "The fractional transversal number of a hypergraph H is the minimum total weight assignable to vertices so that, for each hyperedge, the sum of weights on its vertices is at least 1."
    ),
)
@require_hypergraph_like
@with_solver
def fractional_transversal_number(
    H: HypergraphLike,
    *,
    verbose: bool = False,
    solve=None,
) -> float:
    r"""
    Return the fractional transversal number of a hypergraph.
    """
    for edge in H.E:
        if len(edge) == 0:
            raise ValueError(
                "Hypergraph contains an empty hyperedge; no fractional transversal exists."
            )

    if not H.E:
        return 0.0

    vertices = list(H.V)

    prob = pulp.LpProblem("FractionalTransversalNumberHypergraph", pulp.LpMinimize)
    x = {
        v: pulp.LpVariable(f"x_{i}", lowBound=0.0, upBound=1.0, cat="Continuous")
        for i, v in enumerate(vertices)
    }

    prob += pulp.lpSum(x[v] for v in vertices)

    for i, edge in enumerate(H.E):
        prob += pulp.lpSum(x[v] for v in edge) >= 1, f"hit_edge_{i}"

    solve(prob)

    value = pulp.value(prob.objective)
    if verbose:
        status = pulp.LpStatus.get(prob.status, str(prob.status))
        print(f"Solver status : {status}")
        print(f"Objective     : {value}")

    return float(value if value is not None else 0.0)
