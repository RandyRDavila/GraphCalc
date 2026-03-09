# src/graphcalc/hypergraphs/invariants/independence.py

from __future__ import annotations

from typing import Hashable, Set

import pulp

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.solvers import with_solver
from graphcalc.utils import _extract_and_report
from graphcalc.metadata import invariant_metadata

__all__ = [
    "maximum_independent_set",
    "independence_number",
]


def _check_no_empty_edges_for_independence(H: HypergraphLike) -> None:
    """
    Validate that the usual independence-set formulation is well-defined.
    """
    for edge in H.E:
        if len(edge) == 0:
            raise ValueError(
                "Hypergraph contains an empty hyperedge; independence set is undefined."
            )


@invariant_metadata(
    display_name="Maximum independent set",
    notation=r"I_{\max}(H)",
    category="independence invariants",
    aliases=("largest independent set",),
    definition=(
        "A maximum independent set of a hypergraph H is an independent vertex set of maximum cardinality, "
        "where a set is independent if it contains no hyperedge of H as a subset."
    ),
)
@require_hypergraph_like
@with_solver
def maximum_independent_set(
    H: HypergraphLike,
    *,
    verbose: bool = False,
    solve=None,
) -> Set[Hashable]:
    r"""
    Return a largest independent set of vertices in a hypergraph.
    """
    _check_no_empty_edges_for_independence(H)

    vertices = list(H.V)

    if not H.E:
        return set(vertices)

    prob = pulp.LpProblem("MaximumIndependentSetHypergraph", pulp.LpMaximize)
    x = {v: pulp.LpVariable(f"x_{i}", cat="Binary") for i, v in enumerate(vertices)}

    prob += pulp.lpSum(x[v] for v in vertices)

    for i, edge in enumerate(H.E):
        prob += pulp.lpSum(x[v] for v in edge) <= len(edge) - 1, f"avoid_edge_{i}"

    solve(prob)

    return _extract_and_report(prob, x, verbose=verbose)


@invariant_metadata(
    display_name="Independence number",
    notation=r"\alpha(H)",
    category="independence invariants",
    aliases=("hypergraph independence number",),
    definition=(
        "The independence number of a hypergraph H is the maximum cardinality of an independent vertex set of H."
    ),
)
@require_hypergraph_like
def independence_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the independence number of a hypergraph.
    """
    return len(maximum_independent_set(H, **solver_kwargs))
