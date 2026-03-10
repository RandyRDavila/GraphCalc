# src/graphcalc/hypergraphs/invariants/domination.py

"""
Domination and total domination in hypergraphs via the 2-section graph.

In this module, hypergraph domination is defined in the standard
adjacency-based sense used by Michael A. Henning and collaborators:
a vertex set ``D`` dominates a hypergraph ``H`` if every vertex outside
``D`` shares a hyperedge with some vertex in ``D``. Equivalently,
``D`` is a dominating set of the 2-section (primal) graph of ``H``. The
same equivalence holds for total domination.

Definitions
-----------
Let ``H = (V, E)`` be a finite hypergraph.

The **2-section** (or **primal graph**) of ``H`` is the graph on vertex
set ``V`` in which two distinct vertices ``u`` and ``v`` are adjacent
whenever there exists a hyperedge ``e in E`` such that
``{u, v} subseteq e``.

A set ``D subseteq V`` is a **dominating set** of ``H`` if every vertex
``v in V \\ D`` has a neighbor in ``D`` in the 2-section. Equivalently,

    for every v in V,
        v in D  or  there exists u in D such that u and v lie together in
        some hyperedge.

A set ``D subseteq V`` is a **total dominating set** of ``H`` if every
vertex ``v in V`` has a *distinct* neighbor in ``D`` in the 2-section.
Equivalently,

    for every v in V,
        there exists u in D, u != v, such that u and v lie together in
        some hyperedge.

Thus total domination does not allow self-domination and is infeasible
whenever the 2-section contains an isolated vertex.

Functions
---------
minimum_dominating_set
    Compute a minimum dominating set of a hypergraph.
domination_number
    Compute the domination number gamma(H).
minimum_total_dominating_set
    Compute a minimum total dominating set of a hypergraph.
total_domination_number
    Compute the total domination number gamma_t(H).

Notes
-----
These routines formulate domination and total domination as 0-1 linear
programs and solve them using the shared GraphCalc solver framework via
``graphcalc.solvers.with_solver``.

On 2-uniform hypergraphs (ordinary simple graphs), these notions reduce
exactly to the usual graph domination and total domination problems.
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Set

import pulp

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.solvers import with_solver
from graphcalc.utils import _extract_and_report
from graphcalc.metadata import invariant_metadata

__all__ = [
    "minimum_dominating_set",
    "domination_number",
    "minimum_total_dominating_set",
    "total_domination_number",
]


def _check_k_uniform(H: HypergraphLike, k: int) -> None:
    """
    Validate that a hypergraph is k-uniform.

    Parameters
    ----------
    H : HypergraphLike
        Hypergraph to check.
    k : int
        Target uniformity.

    Raises
    ------
    ValueError
        If ``k`` is negative or if some hyperedge has size different from
        ``k``.
    """
    if k < 0:
        raise ValueError("k must be >= 0.")
    if any(len(edge) != int(k) for edge in H.E):
        raise ValueError("H is not k-uniform.")


def _two_section_neighbors(H: HypergraphLike) -> Dict[Hashable, Set[Hashable]]:
    """
    Build open neighborhoods in the 2-section graph of a hypergraph.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Returns
    -------
    dict[hashable, set[hashable]]
        Mapping ``v -> N(v)``, where ``N(v)`` is the set of vertices that
        share at least one hyperedge with ``v``.
    """
    neighbors: Dict[Hashable, Set[Hashable]] = {v: set() for v in H.V}
    for edge in H.E:
        edge_list = list(edge)
        for i, v in enumerate(edge_list):
            for j, u in enumerate(edge_list):
                if i != j:
                    neighbors[v].add(u)
    return neighbors


def _isolated_vertices_in_two_section(H: HypergraphLike) -> list[Hashable]:
    """
    Return the isolated vertices of the 2-section graph.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Returns
    -------
    list[hashable]
        Vertices with empty open neighborhood in the 2-section.
    """
    neighbors = _two_section_neighbors(H)
    return [v for v in H.V if not neighbors[v]]


@invariant_metadata(
    display_name="Minimum dominating set",
    notation=r"D_{\min}(H)",
    category="domination invariants",
    aliases=("minimum domination set",),
    definition=(
        "A minimum dominating set of a hypergraph H is a dominating set of minimum cardinality, "
        "where domination is defined in the 2-section graph of H."
    ),
)
@require_hypergraph_like
@with_solver
def minimum_dominating_set(
    H: HypergraphLike,
    *,
    k: Optional[int] = None,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Return a minimum dominating set of a hypergraph.

    This function uses domination in the 2-section graph of the hypergraph.
    A set ``D subseteq V(H)`` is a dominating set if every vertex either
    belongs to ``D`` or has a neighbor in ``D`` in the 2-section.
    Equivalently,

    .. math::
        \forall v \in V(H), \quad
        v \in D \;\; \text{or} \;\;
        \exists u \in D \text{ such that } \{u,v\} \subseteq e
        \text{ for some } e \in E(H).

    The optimization problem solved is:

    .. math::
        \min \sum_{v \in V(H)} x_v

    subject to

    .. math::
        x_v + \sum_{u \in N_H(v)} x_u \ge 1
        \quad \text{for all } v \in V(H),

    where ``N_H(v)`` denotes the open neighborhood of ``v`` in the
    2-section graph and each ``x_v`` is binary.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int, optional
        If provided, first verify that ``H`` is ``k``-uniform. This is only
        a validation check and is not required for correctness.
    verbose : bool, default=False
        If True, print solver status, objective value, and extracted
        solution.

    Other Parameters
    ----------------
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver specification handled by
        :func:`graphcalc.solvers.resolve_solver`.
    solver_options : dict, optional
        Extra keyword arguments used when constructing the solver.

    Returns
    -------
    set of hashable
        A minimum dominating set of ``H``.

    Raises
    ------
    ValueError
        If ``k`` is provided and ``H`` is not ``k``-uniform.
    ValueError
        If no optimal solution is found by the solver.

    Notes
    -----
    Vertices isolated in the 2-section must belong to every dominating set.
    This includes vertices that are not contained in any hyperedge, as well
    as vertices that appear only in singleton or empty-incidence contexts.

    If ``H`` has no vertices, the empty set is returned.

    Examples
    --------
    >>> import graphcalc.hypergraphs as hc
    >>> from graphcalc.hypergraphs.invariants.domination import minimum_dominating_set
    >>> H = hc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> D = minimum_dominating_set(H)
    >>> D == {2}
    True

    >>> from graphcalc.hypergraphs.invariants.domination import domination_number
    >>> domination_number(H)
    1
    """
    if k is not None:
        _check_k_uniform(H, k)

    vertices = list(H.V)
    if not vertices:
        return set()

    neighbors = _two_section_neighbors(H)

    prob = pulp.LpProblem("MinimumDominatingSetHypergraph", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{i}", cat="Binary") for i, v in enumerate(vertices)}

    prob += pulp.lpSum(x[v] for v in vertices)

    for i, v in enumerate(vertices):
        prob += x[v] + pulp.lpSum(x[u] for u in neighbors[v]) >= 1, f"dominate_{i}"

    solve(prob)

    return _extract_and_report(prob, x, verbose=verbose)


@invariant_metadata(
    display_name="Domination number",
    notation=r"\gamma(H)",
    category="domination invariants",
    aliases=("hypergraph domination number",),
    definition=(
        "The domination number of a hypergraph H is the minimum cardinality of a dominating set of H, "
        "where domination is defined in the 2-section graph of H."
    ),
)
@require_hypergraph_like
def domination_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the domination number of a hypergraph.

    The **domination number** of a hypergraph, under the 2-section
    definition, is

    .. math::
        \gamma(H) = \min\{ |D| : D \subseteq V(H),\ D \text{ dominates } H \}.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Other Parameters
    ----------------
    k : int, optional
        Passed through to :func:`minimum_dominating_set`.
    verbose : bool, default=False
        Passed through to :func:`minimum_dominating_set`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Passed through to :func:`minimum_dominating_set`.
    solver_options : dict, optional
        Passed through to :func:`minimum_dominating_set`.

    Returns
    -------
    int
        The domination number :math:`\gamma(H)`.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> from graphcalc.hypergraphs.invariants.domination import domination_number
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> domination_number(H)
    1
    """
    return len(minimum_dominating_set(H, **solver_kwargs))


@invariant_metadata(
    display_name="Minimum total dominating set",
    notation=r"D^t_{\min}(H)",
    category="domination invariants",
    aliases=("minimum total domination set",),
    definition=(
        "A minimum total dominating set of a hypergraph H is a total dominating set of minimum cardinality, "
        "where total domination is defined in the 2-section graph of H."
    ),
)
@require_hypergraph_like
@with_solver
def minimum_total_dominating_set(
    H: HypergraphLike,
    *,
    k: Optional[int] = None,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Return a minimum total dominating set of a hypergraph.

    This function uses total domination in the 2-section graph of the
    hypergraph. A set ``D subseteq V(H)`` is a total dominating set if every
    vertex has a *distinct* neighbor in ``D`` in the 2-section. Equivalently,

    .. math::
        \forall v \in V(H), \quad
        \exists u \in D,\; u \ne v,
        \text{ such that } \{u,v\} \subseteq e
        \text{ for some } e \in E(H).

    Thus self-domination is not allowed.

    The optimization problem solved is:

    .. math::
        \min \sum_{v \in V(H)} x_v

    subject to

    .. math::
        \sum_{u \in N_H(v)} x_u \ge 1
        \quad \text{for all } v \in V(H),

    where ``N_H(v)`` denotes the open neighborhood of ``v`` in the
    2-section graph and each ``x_v`` is binary.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int, optional
        If provided, first verify that ``H`` is ``k``-uniform. This is only
        a validation check and is not required for correctness.
    verbose : bool, default=False
        If True, print solver status, objective value, and extracted
        solution.

    Other Parameters
    ----------------
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver specification handled by
        :func:`graphcalc.solvers.resolve_solver`.
    solver_options : dict, optional
        Extra keyword arguments used when constructing the solver.

    Returns
    -------
    set of hashable
        A minimum total dominating set of ``H``.

    Raises
    ------
    ValueError
        If ``k`` is provided and ``H`` is not ``k``-uniform.
    ValueError
        If the 2-section contains an isolated vertex, in which case no total
        dominating set exists.
    ValueError
        If no optimal solution is found by the solver.

    Notes
    -----
    If ``H`` has no vertices, the empty set is returned.

    Hypergraphs with isolated vertices in the 2-section have no total
    dominating set, since such vertices have no distinct neighbor available
    to dominate them.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> from graphcalc.hypergraphs.invariants.domination import minimum_total_dominating_set
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> T = minimum_total_dominating_set(H)
    >>> len(T)
    2
    """
    if k is not None:
        _check_k_uniform(H, k)

    vertices = list(H.V)
    if not vertices:
        return set()

    neighbors = _two_section_neighbors(H)
    isolated = [v for v in vertices if not neighbors[v]]
    if isolated:
        raise ValueError(
            f"Total domination infeasible: isolated vertices in 2-section: {isolated}"
        )

    prob = pulp.LpProblem("MinimumTotalDominatingSetHypergraph", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{i}", cat="Binary") for i, v in enumerate(vertices)}

    prob += pulp.lpSum(x[v] for v in vertices)

    for i, v in enumerate(vertices):
        prob += pulp.lpSum(x[u] for u in neighbors[v]) >= 1, f"total_dominate_{i}"

    solve(prob)

    return _extract_and_report(prob, x, verbose=verbose)


@invariant_metadata(
    display_name="Total domination number",
    notation=r"\gamma_t(H)",
    category="domination invariants",
    aliases=("hypergraph total domination number",),
    definition=(
        "The total domination number of a hypergraph H is the minimum cardinality of a total dominating set of H, "
        "where total domination is defined in the 2-section graph of H."
    ),
)
@require_hypergraph_like
def total_domination_number(
    H: HypergraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the total domination number of a hypergraph.

    The **total domination number** of a hypergraph, under the 2-section
    definition, is

    .. math::
        \gamma_t(H) =
        \min\{ |D| : D \subseteq V(H),\ D \text{ totally dominates } H \}.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Other Parameters
    ----------------
    k : int, optional
        Passed through to :func:`minimum_total_dominating_set`.
    verbose : bool, default=False
        Passed through to :func:`minimum_total_dominating_set`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Passed through to :func:`minimum_total_dominating_set`.
    solver_options : dict, optional
        Passed through to :func:`minimum_total_dominating_set`.

    Returns
    -------
    int
        The total domination number :math:`\gamma_t(H)`.

    Raises
    ------
    ValueError
        If no total dominating set exists.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> from graphcalc.hypergraphs.invariants.domination import total_domination_number
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> total_domination_number(H)
    2
    """
    return len(minimum_total_dominating_set(H, **solver_kwargs))
