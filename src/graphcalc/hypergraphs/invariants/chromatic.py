# src/graphcalc/hypergraphs/invariants/chromatic.py

from __future__ import annotations

from typing import Dict, FrozenSet, Hashable

import pulp

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.solvers import with_solver
from graphcalc.metadata import invariant_metadata

__all__ = [
    "weak_coloring",
    "weak_chromatic_number",
    "strong_coloring",
    "strong_chromatic_number",
    "edge_coloring",
    "edge_chromatic_number",
]


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer.")


def _solve_status_ok(prob: pulp.LpProblem) -> bool:
    return pulp.LpStatus.get(prob.status, "") == "Optimal"


def _extract_vertex_coloring(
    vertices: list[Hashable],
    x: Dict[tuple[Hashable, int], pulp.LpVariable],
    k: int,
) -> Dict[Hashable, int]:
    coloring: Dict[Hashable, int] = {}
    for v in vertices:
        assigned = [c for c in range(k) if pulp.value(x[(v, c)]) is not None and pulp.value(x[(v, c)]) > 0.5]
        if len(assigned) != 1:
            raise ValueError(f"Failed to extract a unique color for vertex {v!r}.")
        coloring[v] = assigned[0]
    return coloring


def _extract_edge_coloring(
    edges: list[FrozenSet[Hashable]],
    y: Dict[tuple[int, int], pulp.LpVariable],
    k: int,
) -> Dict[FrozenSet[Hashable], int]:
    coloring: Dict[FrozenSet[Hashable], int] = {}
    for i, edge in enumerate(edges):
        assigned = [c for c in range(k) if pulp.value(y[(i, c)]) is not None and pulp.value(y[(i, c)]) > 0.5]
        if len(assigned) != 1:
            raise ValueError(f"Failed to extract a unique color for edge {edge!r}.")
        coloring[edge] = assigned[0]
    return coloring


@invariant_metadata(
    display_name="Weak coloring",
    notation=r"\phi_w(H)",
    category="chromatic invariants",
    aliases=("weak proper coloring",),
    definition=(
        "A weak coloring of a hypergraph H is a vertex coloring such that no hyperedge "
        "of size at least 2 is monochromatic."
    ),
)
@require_hypergraph_like
@with_solver
def weak_coloring(
    H: HypergraphLike,
    *,
    k: int,
    verbose: bool = False,
    solve=None,
) -> Dict[Hashable, int]:
    r"""
    Return a weak proper vertex coloring of a hypergraph using at most ``k`` colors.

    A weak coloring is a vertex coloring such that no hyperedge of size at least 2
    is monochromatic.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int
        Number of available colors.
    verbose : bool, default=False
        If True, print basic solver information.

    Returns
    -------
    dict[hashable, int]
        Mapping ``vertex -> color`` with colors in ``{0, ..., k-1}``.

    Raises
    ------
    ValueError
        If ``k <= 0`` or if no weak ``k``-coloring exists.

    Notes
    -----
    Empty and singleton edges impose no weak-coloring restriction.
    """
    _validate_k(k)

    vertices = list(H.V)
    if not vertices:
        return {}

    prob = pulp.LpProblem("WeakColoringHypergraph", pulp.LpMinimize)
    x = {
        (v, c): pulp.LpVariable(f"x_{i}_{c}", cat="Binary")
        for i, v in enumerate(vertices)
        for c in range(k)
    }

    prob += 0

    for v in vertices:
        prob += pulp.lpSum(x[(v, c)] for c in range(k)) == 1, f"assign_{repr(v)}"

    for ei, edge in enumerate(H.E):
        if len(edge) < 2:
            continue
        for c in range(k):
            prob += pulp.lpSum(x[(v, c)] for v in edge) <= len(edge) - 1, f"weak_edge_{ei}_color_{c}"

    solve(prob)

    if not _solve_status_ok(prob):
        raise ValueError(f"No weak {k}-coloring exists.")

    if verbose:
        print(f"Solver status : {pulp.LpStatus.get(prob.status, str(prob.status))}")

    return _extract_vertex_coloring(vertices, x, k)


@invariant_metadata(
    display_name="Weak chromatic number",
    notation=r"\chi_w(H)",
    category="chromatic invariants",
    aliases=("weak coloring number",),
    definition=(
        "The weak chromatic number of a hypergraph H is the minimum number of colors "
        "needed for a weak coloring of H."
    ),
)
@require_hypergraph_like
def weak_chromatic_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the weak chromatic number of a hypergraph.

    The weak chromatic number is the minimum number of colors needed to color
    vertices so that no hyperedge of size at least 2 is monochromatic.
    """
    n = len(H.V)
    if n == 0:
        return 0
    if not H.E:
        return 1 if n > 0 else 0

    for k in range(1, n + 1):
        try:
            weak_coloring(H, k=k, **solver_kwargs)
            return k
        except ValueError:
            pass

    raise ValueError("Failed to determine weak chromatic number.")


@invariant_metadata(
    display_name="Strong coloring",
    notation=r"\phi_s(H)",
    category="chromatic invariants",
    aliases=("strong proper coloring",),
    definition=(
        "A strong coloring of a hypergraph H is a vertex coloring such that any two "
        "distinct vertices contained in a common hyperedge receive different colors; "
        "equivalently, every hyperedge is rainbow."
    ),
)
@require_hypergraph_like
@with_solver
def strong_coloring(
    H: HypergraphLike,
    *,
    k: int,
    verbose: bool = False,
    solve=None,
) -> Dict[Hashable, int]:
    r"""
    Return a strong proper vertex coloring of a hypergraph using at most ``k`` colors.

    A strong coloring is a vertex coloring such that any two distinct vertices
    contained in a common hyperedge receive different colors. Equivalently, each
    hyperedge is rainbow.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int
        Number of available colors.
    verbose : bool, default=False
        If True, print basic solver information.

    Returns
    -------
    dict[hashable, int]
        Mapping ``vertex -> color`` with colors in ``{0, ..., k-1}``.

    Raises
    ------
    ValueError
        If ``k <= 0`` or if no strong ``k``-coloring exists.
    """
    _validate_k(k)

    vertices = list(H.V)
    if not vertices:
        return {}

    if any(len(edge) > k for edge in H.E):
        raise ValueError(f"No strong {k}-coloring exists because some edge has size > {k}.")

    prob = pulp.LpProblem("StrongColoringHypergraph", pulp.LpMinimize)
    x = {
        (v, c): pulp.LpVariable(f"x_{i}_{c}", cat="Binary")
        for i, v in enumerate(vertices)
        for c in range(k)
    }

    prob += 0

    for v in vertices:
        prob += pulp.lpSum(x[(v, c)] for c in range(k)) == 1, f"assign_{repr(v)}"

    for ei, edge in enumerate(H.E):
        edge_list = list(edge)
        for c in range(k):
            prob += pulp.lpSum(x[(v, c)] for v in edge_list) <= 1, f"strong_edge_{ei}_color_{c}"

    solve(prob)

    if not _solve_status_ok(prob):
        raise ValueError(f"No strong {k}-coloring exists.")

    if verbose:
        print(f"Solver status : {pulp.LpStatus.get(prob.status, str(prob.status))}")

    return _extract_vertex_coloring(vertices, x, k)


@invariant_metadata(
    display_name="Strong chromatic number",
    notation=r"\chi_s(H)",
    category="chromatic invariants",
    aliases=("strong coloring number",),
    definition=(
        "The strong chromatic number of a hypergraph H is the minimum number of colors "
        "needed for a strong coloring of H."
    ),
)
@require_hypergraph_like
def strong_chromatic_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the strong chromatic number of a hypergraph.

    The strong chromatic number is the minimum number of colors needed to color
    vertices so that every hyperedge is rainbow.
    """
    n = len(H.V)
    if n == 0:
        return 0
    if not H.E:
        return 1 if n > 0 else 0

    lower = max((len(edge) for edge in H.E), default=1)
    for k in range(lower, n + 1):
        try:
            strong_coloring(H, k=k, **solver_kwargs)
            return k
        except ValueError:
            pass

    raise ValueError("Failed to determine strong chromatic number.")


@invariant_metadata(
    display_name="Edge coloring",
    notation=r"\phi'(H)",
    category="chromatic invariants",
    aliases=("proper edge coloring",),
    definition=(
        "An edge coloring of a hypergraph H is a coloring of its hyperedges such that "
        "any two intersecting hyperedges receive different colors."
    ),
)
@require_hypergraph_like
@with_solver
def edge_coloring(
    H: HypergraphLike,
    *,
    k: int,
    verbose: bool = False,
    solve=None,
) -> Dict[FrozenSet[Hashable], int]:
    r"""
    Return a proper edge coloring of a hypergraph using at most ``k`` colors.

    A proper edge coloring assigns colors to hyperedges so that intersecting
    hyperedges receive different colors.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int
        Number of available colors.
    verbose : bool, default=False
        If True, print basic solver information.

    Returns
    -------
    dict[frozenset, int]
        Mapping ``edge -> color`` with colors in ``{0, ..., k-1}``.

    Raises
    ------
    ValueError
        If ``k <= 0`` or if no proper edge ``k``-coloring exists.
    """
    _validate_k(k)

    edges = list(H.E)
    if not edges:
        return {}

    prob = pulp.LpProblem("EdgeColoringHypergraph", pulp.LpMinimize)
    y = {
        (i, c): pulp.LpVariable(f"y_{i}_{c}", cat="Binary")
        for i in range(len(edges))
        for c in range(k)
    }

    prob += 0

    for i in range(len(edges)):
        prob += pulp.lpSum(y[(i, c)] for c in range(k)) == 1, f"assign_edge_{i}"

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edges[i] & edges[j]:
                for c in range(k):
                    prob += y[(i, c)] + y[(j, c)] <= 1, f"edge_conflict_{i}_{j}_{c}"

    solve(prob)

    if not _solve_status_ok(prob):
        raise ValueError(f"No edge {k}-coloring exists.")

    if verbose:
        print(f"Solver status : {pulp.LpStatus.get(prob.status, str(prob.status))}")

    return _extract_edge_coloring(edges, y, k)


@invariant_metadata(
    display_name="Edge chromatic number",
    notation=r"\chi'(H)",
    category="chromatic invariants",
    aliases=("hyperedge chromatic number", "edge coloring number"),
    definition=(
        "The edge chromatic number of a hypergraph H is the minimum number of colors "
        "needed to color its hyperedges so that intersecting hyperedges receive different colors."
    ),
)
@require_hypergraph_like
def edge_chromatic_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the edge chromatic number of a hypergraph.

    This is the minimum number of colors needed to color hyperedges so that
    intersecting hyperedges receive different colors.
    """
    m = len(H.E)
    if m == 0:
        return 0

    for k in range(1, m + 1):
        try:
            edge_coloring(H, k=k, **solver_kwargs)
            return k
        except ValueError:
            pass

    raise ValueError("Failed to determine edge chromatic number.")
