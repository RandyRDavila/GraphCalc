from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple, List, Set

import networkx as nx
import pulp

from ..core import SimpleGraph
from graphcalc.utils import enforce_type, GraphLike
from graphcalc.solvers import with_solver
from graphcalc.metadata import invariant_metadata


__all__ = [
    "SolveResult",
    "EdgeColoringSolveResult",
    "open_neighborhood_conflict_free_coloring",
    "proper_open_neighborhood_conflict_free_coloring",
    "open_neighborhood_odd_coloring",
    "open_neighborhood_conflict_free_chromatic_number",
    "proper_open_neighborhood_conflict_free_chromatic_number",
    "open_neighborhood_odd_chromatic_number",
    "has_open_neighborhood_conflict_free_coloring",
    "has_proper_open_neighborhood_conflict_free_coloring",
    "has_open_neighborhood_odd_coloring",
    "has_rainbow_connection_coloring",
    "rainbow_connection_coloring",
    "rainbow_connection_number",
    "has_strong_rainbow_connection_coloring",
    "strong_rainbow_connection_coloring",
    "strong_rainbow_connection_number",
]


Coloring = Dict[Hashable, int]


@dataclass(frozen=True)
class SolveResult:
    r"""
    Result of a coloring feasibility computation.

    Parameters
    ----------
    feasible : bool
        Whether a feasible coloring satisfying the requested constraints exists.
    coloring : dict, optional
        A witness coloring when one is found. The dictionary maps each vertex
        to a positive integer color label.
    """
    feasible: bool
    coloring: Optional[Coloring] = None


def _extract_coloring(
    x: Dict[Tuple[Hashable, int], pulp.LpVariable],
    nodes: List[Hashable],
    k: int,
) -> Coloring:
    r"""
    Extract a vertex coloring from binary assignment variables.

    Parameters
    ----------
    x : dict
        A dictionary of binary variables indexed by ``(vertex, color)``.
    nodes : list
        The nodes of the graph.
    k : int
        The number of available colors.

    Returns
    -------
    dict
        A coloring mapping each node to an integer in ``{1, \dots, k}``.

    Raises
    ------
    RuntimeError
        If a color cannot be extracted for some vertex.
    """
    col: Coloring = {}
    for v in nodes:
        chosen = None
        for c in range(1, k + 1):
            val = pulp.value(x[(v, c)])
            if val is not None and val > 0.5:
                chosen = c
                break
        if chosen is None:
            raise RuntimeError(f"Failed to extract color for vertex {v}")
        col[v] = chosen
    return col


@with_solver
def _has_open_neighborhood_conflict_free_coloring_k(
    G: GraphLike,
    k: int,
    *,
    verbose: bool = False,
    solve=None,
) -> SolveResult:
    r"""
    Solve the open-neighborhood conflict-free coloring feasibility problem.

    A coloring is feasible if every non-isolated vertex has some color that
    appears exactly once in its open neighborhood.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        Number of available colors.
    verbose : bool, default=False
        If True, enable solver output when supported.
    solve : callable, optional
        Solver hook injected by :func:`graphcalc.solvers.with_solver`.

    Returns
    -------
    SolveResult
        Feasibility status together with a witness coloring when feasible.
    """
    if k < 1:
        return SolveResult(False, None)

    nodes = list(G.nodes())
    prob = pulp.LpProblem("OpenNeighborhoodConflictFreeColoring", pulp.LpMinimize)

    x = {
        (v, c): pulp.LpVariable(f"x_{v}_{c}", 0, 1, cat="Binary")
        for v in nodes for c in range(1, k + 1)
    }

    for v in nodes:
        prob += pulp.lpSum(x[(v, c)] for c in range(1, k + 1)) == 1

    for v in nodes:
        Nv = list(G.neighbors(v))
        deg = len(Nv)
        if deg == 0:
            continue

        y = {
            c: pulp.LpVariable(f"y_{v}_{c}", 0, 1, cat="Binary")
            for c in range(1, k + 1)
        }

        prob += pulp.lpSum(y[c] for c in range(1, k + 1)) >= 1

        for c in range(1, k + 1):
            s = pulp.lpSum(x[(u, c)] for u in Nv)
            prob += s >= y[c]
            prob += s <= 1 + deg * (1 - y[c])

    prob += 0
    solve(prob)

    feasible = pulp.LpStatus[prob.status] == "Optimal"
    if not feasible:
        return SolveResult(False, None)
    return SolveResult(True, _extract_coloring(x, nodes, k))


@with_solver
def _has_proper_open_neighborhood_conflict_free_coloring_k(
    G: GraphLike,
    k: int,
    *,
    verbose: bool = False,
    solve=None,
) -> SolveResult:
    r"""
    Solve the proper open-neighborhood conflict-free coloring feasibility problem.

    A coloring is feasible if it is proper and every non-isolated vertex has
    some color appearing exactly once in its open neighborhood.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        Number of available colors.
    verbose : bool, default=False
        If True, enable solver output when supported.
    solve : callable, optional
        Solver hook injected by :func:`graphcalc.solvers.with_solver`.

    Returns
    -------
    SolveResult
        Feasibility status together with a witness coloring when feasible.
    """
    if k < 1:
        return SolveResult(False, None)

    nodes = list(G.nodes())
    edges = list(G.edges())
    prob = pulp.LpProblem("ProperOpenNeighborhoodConflictFreeColoring", pulp.LpMinimize)

    x = {
        (v, c): pulp.LpVariable(f"x_{v}_{c}", 0, 1, cat="Binary")
        for v in nodes for c in range(1, k + 1)
    }

    for v in nodes:
        prob += pulp.lpSum(x[(v, c)] for c in range(1, k + 1)) == 1

    for u, v in edges:
        for c in range(1, k + 1):
            prob += x[(u, c)] + x[(v, c)] <= 1

    for v in nodes:
        Nv = list(G.neighbors(v))
        deg = len(Nv)
        if deg == 0:
            continue

        y = {
            c: pulp.LpVariable(f"y_{v}_{c}", 0, 1, cat="Binary")
            for c in range(1, k + 1)
        }
        prob += pulp.lpSum(y[c] for c in range(1, k + 1)) >= 1

        for c in range(1, k + 1):
            s = pulp.lpSum(x[(u, c)] for u in Nv)
            prob += s >= y[c]
            prob += s <= 1 + deg * (1 - y[c])

    prob += 0
    solve(prob)

    feasible = pulp.LpStatus[prob.status] == "Optimal"
    if not feasible:
        return SolveResult(False, None)
    return SolveResult(True, _extract_coloring(x, nodes, k))


@with_solver
def _has_open_neighborhood_odd_coloring_k(
    G: GraphLike,
    k: int,
    *,
    verbose: bool = False,
    solve=None,
) -> SolveResult:
    r"""
    Solve the open-neighborhood odd coloring feasibility problem.

    A coloring is feasible if it is proper and every non-isolated vertex has
    some color that appears an odd number of times in its open neighborhood.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        Number of available colors.
    verbose : bool, default=False
        If True, enable solver output when supported.
    solve : callable, optional
        Solver hook injected by :func:`graphcalc.solvers.with_solver`.

    Returns
    -------
    SolveResult
        Feasibility status together with a witness coloring when feasible.
    """
    if k < 1:
        return SolveResult(False, None)

    nodes = list(G.nodes())
    edges = list(G.edges())
    prob = pulp.LpProblem("OpenNeighborhoodOddColoring", pulp.LpMinimize)

    x = {
        (v, c): pulp.LpVariable(f"x_{v}_{c}", 0, 1, cat="Binary")
        for v in nodes for c in range(1, k + 1)
    }

    for v in nodes:
        prob += pulp.lpSum(x[(v, c)] for c in range(1, k + 1)) == 1

    for u, v in edges:
        for c in range(1, k + 1):
            prob += x[(u, c)] + x[(v, c)] <= 1

    for v in nodes:
        Nv = list(G.neighbors(v))
        deg = len(Nv)
        if deg == 0:
            continue

        y = {
            c: pulp.LpVariable(f"y_{v}_{c}", 0, 1, cat="Binary")
            for c in range(1, k + 1)
        }
        t = {
            c: pulp.LpVariable(f"t_{v}_{c}", 0, deg // 2, cat="Integer")
            for c in range(1, k + 1)
        }

        prob += pulp.lpSum(y[c] for c in range(1, k + 1)) >= 1

        for c in range(1, k + 1):
            s = pulp.lpSum(x[(u, c)] for u in Nv)
            prob += s == 2 * t[c] + y[c]

    prob += 0
    solve(prob)

    feasible = pulp.LpStatus[prob.status] == "Optimal"
    if not feasible:
        return SolveResult(False, None)
    return SolveResult(True, _extract_coloring(x, nodes, k))


def _minimum_k(
    G: GraphLike,
    kind: str,
    *,
    k_max: Optional[int] = None,
    **solver_kwargs,
) -> int:
    r"""
    Compute the minimum number of colors needed for a specialized coloring type.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    kind : {'oncf', 'poncf', 'onodd'}
        The coloring variant to test.
    k_max : int, optional
        Upper bound on the number of colors tested. If omitted, uses
        :math:`|V(G)|`.

    Returns
    -------
    int
        The minimum feasible number of colors.

    Raises
    ------
    ValueError
        If ``kind`` is invalid.
    RuntimeError
        If no feasible coloring is found up to ``k_max``.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0
    if k_max is None:
        k_max = n

    kind = kind.lower().strip()
    if kind not in {"oncf", "poncf", "onodd"}:
        raise ValueError("kind must be one of: 'oncf', 'poncf', 'onodd'")

    for k in range(1, k_max + 1):
        try:
            if kind == "oncf":
                res = _has_open_neighborhood_conflict_free_coloring_k(G, k, **solver_kwargs)
            elif kind == "poncf":
                res = _has_proper_open_neighborhood_conflict_free_coloring_k(G, k, **solver_kwargs)
            else:
                res = _has_open_neighborhood_odd_coloring_k(G, k, **solver_kwargs)
        except ValueError:
            continue

        if res.feasible:
            return k

    raise RuntimeError(f"No feasible {kind} coloring found up to k_max={k_max}")


@enforce_type(0, (nx.Graph, SimpleGraph))
def has_open_neighborhood_conflict_free_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> bool:
    r"""
    Decide whether a graph admits an open-neighborhood conflict-free coloring with ``k`` colors.

    A vertex coloring :math:`\varphi : V(G) \to \{1,\dots,k\}` is called an
    **open-neighborhood conflict-free coloring** if, for every non-isolated
    vertex :math:`v`, there exists a color appearing exactly once in the open
    neighborhood :math:`N(v)`.

    Unlike proper colorings, adjacent vertices are allowed to receive the same
    color.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    bool
        ``True`` if such a coloring exists, and ``False`` otherwise.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.has_open_neighborhood_conflict_free_coloring(G, 2)
    True
    """
    try:
        return _has_open_neighborhood_conflict_free_coloring_k(G, k, **solver_kwargs).feasible
    except ValueError:
        return False


@enforce_type(0, (nx.Graph, SimpleGraph))
def has_proper_open_neighborhood_conflict_free_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> bool:
    r"""
    Decide whether a graph admits a proper open-neighborhood conflict-free coloring with ``k`` colors.

    A vertex coloring :math:`\varphi : V(G) \to \{1,\dots,k\}` is a
    **proper open-neighborhood conflict-free coloring** if:

    - adjacent vertices receive different colors, and
    - for every non-isolated vertex :math:`v`, some color appears exactly once
      in the open neighborhood :math:`N(v)`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    bool
        ``True`` if such a coloring exists, and ``False`` otherwise.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.has_proper_open_neighborhood_conflict_free_coloring(G, 2)
    True
    """
    try:
        return _has_proper_open_neighborhood_conflict_free_coloring_k(G, k, **solver_kwargs).feasible
    except ValueError:
        return False


@enforce_type(0, (nx.Graph, SimpleGraph))
def has_open_neighborhood_odd_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> bool:
    r"""
    Decide whether a graph admits an open-neighborhood odd coloring with ``k`` colors.

    A vertex coloring :math:`\varphi : V(G) \to \{1,\dots,k\}` is an
    **open-neighborhood odd coloring** if:

    - adjacent vertices receive different colors, and
    - for every non-isolated vertex :math:`v`, there exists a color whose
      frequency in :math:`N(v)` is odd.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    bool
        ``True`` if such a coloring exists, and ``False`` otherwise.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.has_open_neighborhood_odd_coloring(G, 2)
    True
    """
    try:
        return _has_open_neighborhood_odd_coloring_k(G, k, **solver_kwargs).feasible
    except ValueError:
        return False


@enforce_type(0, (nx.Graph, SimpleGraph))
def open_neighborhood_conflict_free_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> Coloring:
    r"""
    Return an open-neighborhood conflict-free coloring using ``k`` colors.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    dict
        A dictionary mapping each vertex of :math:`G` to a color in
        ``{1,\dots,k}``.

    Raises
    ------
    ValueError
        If no feasible coloring exists with ``k`` colors.
    """
    res = _has_open_neighborhood_conflict_free_coloring_k(G, k, **solver_kwargs)
    if not res.feasible or res.coloring is None:
        raise ValueError(f"No open-neighborhood conflict-free coloring exists with k={k}.")
    return res.coloring


@enforce_type(0, (nx.Graph, SimpleGraph))
def proper_open_neighborhood_conflict_free_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> Coloring:
    r"""
    Return a proper open-neighborhood conflict-free coloring using ``k`` colors.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    dict
        A dictionary mapping each vertex of :math:`G` to a color in
        ``{1,\dots,k}``.

    Raises
    ------
    ValueError
        If no feasible coloring exists with ``k`` colors.
    """
    res = _has_proper_open_neighborhood_conflict_free_coloring_k(G, k, **solver_kwargs)
    if not res.feasible or res.coloring is None:
        raise ValueError(f"No proper open-neighborhood conflict-free coloring exists with k={k}.")
    return res.coloring


@enforce_type(0, (nx.Graph, SimpleGraph))
def open_neighborhood_odd_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> Coloring:
    r"""
    Return an open-neighborhood odd coloring using ``k`` colors.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    dict
        A dictionary mapping each vertex of :math:`G` to a color in
        ``{1,\dots,k}``.

    Raises
    ------
    ValueError
        If no feasible coloring exists with ``k`` colors.
    """
    res = _has_open_neighborhood_odd_coloring_k(G, k, **solver_kwargs)
    if not res.feasible or res.coloring is None:
        raise ValueError(f"No open-neighborhood odd coloring exists with k={k}.")
    return res.coloring


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Open-neighborhood conflict-free chromatic number",
    notation=r"\chi_{\mathrm{ONCF}}(G)",
    category="advanced colorings",
    aliases=("CFON number", "conflict-free open-neighborhood number"),
    definition=(
        "The open-neighborhood conflict-free chromatic number is the smallest "
        "integer k such that G admits an open-neighborhood conflict-free "
        "coloring with k colors."
    ),
)
def open_neighborhood_conflict_free_chromatic_number(
    G: GraphLike,
    k_max: Optional[int] = None,
    **solver_kwargs,
) -> int:
    r"""
    Compute the open-neighborhood conflict-free chromatic number of a graph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k_max : int, optional
        Upper bound on the number of colors tested. If omitted, uses
        :math:`|V(G)|`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    int
        The minimum number of colors required.
    """
    return _minimum_k(G, "oncf", k_max=k_max, **solver_kwargs)


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Proper open-neighborhood conflict-free chromatic number",
    notation=r"\chi_{\mathrm{PONCF}}(G)",
    category="advanced colorings",
    aliases=("proper CFON number", "proper conflict-free open-neighborhood number"),
    definition=(
        "The proper open-neighborhood conflict-free chromatic number is the "
        "smallest integer k such that G admits a proper open-neighborhood "
        "conflict-free coloring with k colors."
    ),
)
def proper_open_neighborhood_conflict_free_chromatic_number(
    G: GraphLike,
    k_max: Optional[int] = None,
    **solver_kwargs,
) -> int:
    r"""
    Compute the proper open-neighborhood conflict-free chromatic number of a graph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k_max : int, optional
        Upper bound on the number of colors tested. If omitted, uses
        :math:`|V(G)|`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    int
        The minimum number of colors required.
    """
    return _minimum_k(G, "poncf", k_max=k_max, **solver_kwargs)


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Open-neighborhood odd chromatic number",
    notation=r"\chi_{\mathrm{ONO}}(G)",
    category="advanced colorings",
    aliases=("odd open-neighborhood chromatic number",),
    definition=(
        "The open-neighborhood odd chromatic number is the smallest integer k "
        "such that G admits an open-neighborhood odd coloring with k colors."
    ),
)
def open_neighborhood_odd_chromatic_number(
    G: GraphLike,
    k_max: Optional[int] = None,
    **solver_kwargs,
) -> int:
    r"""
    Compute the open-neighborhood odd chromatic number of a graph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k_max : int, optional
        Upper bound on the number of colors tested. If omitted, uses
        :math:`|V(G)|`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    int
        The minimum number of colors required.
    """
    return _minimum_k(G, "onodd", k_max=k_max, **solver_kwargs)


EdgeColoring = Dict[Tuple[Hashable, Hashable], int]


@dataclass(frozen=True)
class EdgeColoringSolveResult:
    r"""
    Result of an edge-coloring feasibility computation.

    Parameters
    ----------
    feasible : bool
        Whether a feasible coloring satisfying the requested constraints exists.
    coloring : dict, optional
        A witness edge-coloring when one is found. Each key is an undirected edge
        represented by a normalized 2-tuple, and each value is a positive integer
        color label.
    """
    feasible: bool
    coloring: Optional[EdgeColoring] = None


def _normalize_edge(u: Hashable, v: Hashable) -> Tuple[Hashable, Hashable]:
    r"""
    Return a canonical representation of an undirected edge.
    """
    return (u, v) if repr(u) <= repr(v) else (v, u)


def _all_shortest_paths_by_pair(G: GraphLike) -> Dict[Tuple[Hashable, Hashable], List[List[Hashable]]]:
    r"""
    Enumerate all shortest paths between unordered vertex pairs.
    """
    nodes = list(G.nodes())
    pair_paths: Dict[Tuple[Hashable, Hashable], List[List[Hashable]]] = {}
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            pair_paths[(u, v)] = list(nx.all_shortest_paths(G, u, v))
    return pair_paths


def _all_simple_paths_by_pair(
    G: GraphLike,
    *,
    cutoff: Optional[int] = None,
) -> Dict[Tuple[Hashable, Hashable], List[List[Hashable]]]:
    r"""
    Enumerate simple paths between unordered vertex pairs.
    """
    nodes = list(G.nodes())
    pair_paths: Dict[Tuple[Hashable, Hashable], List[List[Hashable]]] = {}
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            pair_paths[(u, v)] = list(nx.all_simple_paths(G, u, v, cutoff=cutoff))
    return pair_paths


def _path_edges(path: List[Hashable]) -> List[Tuple[Hashable, Hashable]]:
    r"""
    Return the normalized edge list of a path.
    """
    return [_normalize_edge(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _extract_edge_coloring(
    x: Dict[Tuple[Tuple[Hashable, Hashable], int], pulp.LpVariable],
    edges: List[Tuple[Hashable, Hashable]],
    k: int,
) -> EdgeColoring:
    r"""
    Extract an edge-coloring from binary assignment variables.
    """
    coloring: EdgeColoring = {}
    for e in edges:
        chosen = None
        for c in range(1, k + 1):
            val = pulp.value(x[(e, c)])
            if val is not None and val > 0.5:
                chosen = c
                break
        if chosen is None:
            raise RuntimeError(f"Failed to extract color for edge {e}")
        coloring[e] = chosen
    return coloring


@with_solver
def _has_rainbow_connection_coloring_k(
    G: GraphLike,
    k: int,
    *,
    strong: bool = False,
    path_cutoff: Optional[int] = None,
    verbose: bool = False,
    solve=None,
) -> EdgeColoringSolveResult:
    r"""
    Solve the fixed-``k`` rainbow connection feasibility problem by MILP.

    For ``strong=False``, this searches for an edge-coloring with ``k`` colors such
    that every pair of distinct vertices is joined by a rainbow path.

    For ``strong=True``, every pair must be joined by a rainbow shortest path.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        Number of available edge colors.
    strong : bool, default=False
        If True, solve the strong rainbow connection variant.
    path_cutoff : int, optional
        Upper bound on the length of simple paths enumerated when ``strong=False``.
        If omitted, all simple paths are enumerated up to length ``|V(G)|-1``.
    verbose : bool, default=False
        If True, enable solver output when supported.
    solve : callable, optional
        Solver hook injected by :func:`graphcalc.solvers.with_solver`.

    Returns
    -------
    EdgeColoringSolveResult
        Feasibility status together with a witness edge-coloring when feasible.
    """
    n = G.number_of_nodes()
    if k < 1:
        return EdgeColoringSolveResult(False, None)
    if n <= 1:
        return EdgeColoringSolveResult(True, {})
    if not nx.is_connected(G):
        return EdgeColoringSolveResult(False, None)

    edges = [_normalize_edge(u, v) for u, v in G.edges()]
    if strong:
        pair_paths = _all_shortest_paths_by_pair(G)
    else:
        cutoff = (n - 1) if path_cutoff is None else path_cutoff
        pair_paths = _all_simple_paths_by_pair(G, cutoff=cutoff)

    for pair, plist in pair_paths.items():
        if not plist:
            return EdgeColoringSolveResult(False, None)

    prob = pulp.LpProblem(
        "StrongRainbowConnectionColoring" if strong else "RainbowConnectionColoring",
        pulp.LpMinimize,
    )

    x = {
        (e, c): pulp.LpVariable(f"x_{e[0]}_{e[1]}_{c}", 0, 1, cat="Binary")
        for e in edges for c in range(1, k + 1)
    }

    y = {}
    for pair, plist in pair_paths.items():
        for idx, _ in enumerate(plist):
            y[(pair, idx)] = pulp.LpVariable(
                f"y_{repr(pair[0])}_{repr(pair[1])}_{idx}", 0, 1, cat="Binary"
            )

    for e in edges:
        prob += pulp.lpSum(x[(e, c)] for c in range(1, k + 1)) == 1

    for pair, plist in pair_paths.items():
        prob += pulp.lpSum(y[(pair, idx)] for idx in range(len(plist))) >= 1

        for idx, path in enumerate(plist):
            pedges = _path_edges(path)
            m = len(pedges)
            if m <= 1:
                continue
            for c in range(1, k + 1):
                prob += (
                    pulp.lpSum(x[(e, c)] for e in pedges)
                    <= 1 + m * (1 - y[(pair, idx)])
                )

    prob += 0
    solve(prob)

    feasible = pulp.LpStatus[prob.status] == "Optimal"
    if not feasible:
        return EdgeColoringSolveResult(False, None)

    return EdgeColoringSolveResult(True, _extract_edge_coloring(x, edges, k))


def _minimum_rainbow_k(
    G: GraphLike,
    *,
    strong: bool = False,
    k_max: Optional[int] = None,
    path_cutoff: Optional[int] = None,
    **solver_kwargs,
) -> int:
    r"""
    Compute the minimum number of colors needed for a rainbow connection variant.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0
    if not nx.is_connected(G):
        raise ValueError("Rainbow connection is defined only for connected graphs.")
    if k_max is None:
        k_max = G.number_of_edges()

    for k in range(1, k_max + 1):
        try:
            res = _has_rainbow_connection_coloring_k(
                G,
                k,
                strong=strong,
                path_cutoff=path_cutoff,
                **solver_kwargs,
            )
        except ValueError:
            continue

        if res.feasible:
            return k

    kind = "strong rainbow connection" if strong else "rainbow connection"
    raise RuntimeError(f"No feasible {kind} coloring found up to k_max={k_max}.")


@enforce_type(0, (nx.Graph, SimpleGraph))
def has_rainbow_connection_coloring(
    G: GraphLike,
    k: int,
    *,
    path_cutoff: Optional[int] = None,
    **solver_kwargs,
) -> bool:
    r"""
    Decide whether a connected graph admits a rainbow connection coloring with ``k`` edge colors.

    An edge-coloring of a connected graph :math:`G` is a **rainbow connection
    coloring** if every pair of distinct vertices is joined by a path whose edges
    have pairwise distinct colors. The minimum such number of colors is the
    **rainbow connection number**, denoted :math:`\mathrm{rc}(G)`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available edge colors.
    path_cutoff : int, optional
        Upper bound on the length of simple paths enumerated during the exact
        feasibility search. If omitted, paths are enumerated up to length
        :math:`|V(G)|-1`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    bool
        ``True`` if such an edge-coloring exists, and ``False`` otherwise.

    Notes
    -----
    This implementation is exact and MILP-based, but path enumeration may become
    expensive on larger graphs.
    """
    if G.number_of_nodes() > 1 and not nx.is_connected(G):
        return False
    try:
        return _has_rainbow_connection_coloring_k(
            G,
            k,
            strong=False,
            path_cutoff=path_cutoff,
            **solver_kwargs,
        ).feasible
    except ValueError:
        return False


@enforce_type(0, (nx.Graph, SimpleGraph))
def has_strong_rainbow_connection_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> bool:
    r"""
    Decide whether a connected graph admits a strong rainbow connection coloring with ``k`` edge colors.

    An edge-coloring of a connected graph :math:`G` is a **strong rainbow
    connection coloring** if every pair of distinct vertices is joined by a
    rainbow shortest path. The minimum such number of colors is the
    **strong rainbow connection number**, denoted :math:`\mathrm{src}(G)`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    k : int
        The number of available edge colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    bool
        ``True`` if such an edge-coloring exists, and ``False`` otherwise.

    Notes
    -----
    This implementation is exact and MILP-based.
    """
    if G.number_of_nodes() > 1 and not nx.is_connected(G):
        return False
    try:
        return _has_rainbow_connection_coloring_k(
            G,
            k,
            strong=True,
            **solver_kwargs,
        ).feasible
    except ValueError:
        return False


@enforce_type(0, (nx.Graph, SimpleGraph))
def rainbow_connection_coloring(
    G: GraphLike,
    k: int,
    *,
    path_cutoff: Optional[int] = None,
    **solver_kwargs,
) -> EdgeColoring:
    r"""
    Return a rainbow connection edge-coloring using ``k`` colors.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        A connected undirected simple graph.
    k : int
        The number of available edge colors.
    path_cutoff : int, optional
        Upper bound on the length of simple paths enumerated during the exact
        feasibility search.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    dict
        A dictionary mapping each undirected edge of :math:`G` to a color in
        ``{1,\dots,k}``.

    Raises
    ------
    ValueError
        If ``G`` is disconnected or no feasible coloring exists with ``k`` colors.
    """
    if G.number_of_nodes() > 1 and not nx.is_connected(G):
        raise ValueError("Rainbow connection is defined only for connected graphs.")

    res = _has_rainbow_connection_coloring_k(
        G,
        k,
        strong=False,
        path_cutoff=path_cutoff,
        **solver_kwargs,
    )
    if not res.feasible or res.coloring is None:
        raise ValueError(f"No rainbow connection coloring exists with k={k}.")
    return res.coloring


@enforce_type(0, (nx.Graph, SimpleGraph))
def strong_rainbow_connection_coloring(
    G: GraphLike,
    k: int,
    **solver_kwargs,
) -> EdgeColoring:
    r"""
    Return a strong rainbow connection edge-coloring using ``k`` colors.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        A connected undirected simple graph.
    k : int
        The number of available edge colors.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    dict
        A dictionary mapping each undirected edge of :math:`G` to a color in
        ``{1,\dots,k}``.

    Raises
    ------
    ValueError
        If ``G`` is disconnected or no feasible coloring exists with ``k`` colors.
    """
    if G.number_of_nodes() > 1 and not nx.is_connected(G):
        raise ValueError("Strong rainbow connection is defined only for connected graphs.")

    res = _has_rainbow_connection_coloring_k(
        G,
        k,
        strong=True,
        **solver_kwargs,
    )
    if not res.feasible or res.coloring is None:
        raise ValueError(f"No strong rainbow connection coloring exists with k={k}.")
    return res.coloring


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Rainbow connection number",
    notation=r"\mathrm{rc}(G)",
    category="advanced colorings",
    aliases=("edge rainbow connection number",),
    definition=(
        "The rainbow connection number rc(G) is the minimum number of colors "
        "in an edge-coloring of G such that every pair of vertices is joined "
        "by a path whose edges have pairwise distinct colors."
    ),
)
def rainbow_connection_number(
    G: GraphLike,
    *,
    k_max: Optional[int] = None,
    path_cutoff: Optional[int] = None,
    **solver_kwargs,
) -> int:
    r"""
    Compute the rainbow connection number of a connected graph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        A connected undirected simple graph.
    k_max : int, optional
        Upper bound on the number of colors tested. If omitted, uses
        :math:`|E(G)|`.
    path_cutoff : int, optional
        Upper bound on the length of simple paths enumerated during the exact
        feasibility search.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    int
        The rainbow connection number :math:`\mathrm{rc}(G)`.

    Raises
    ------
    ValueError
        If ``G`` is disconnected.
    """
    return _minimum_rainbow_k(
        G,
        strong=False,
        k_max=k_max,
        path_cutoff=path_cutoff,
        **solver_kwargs,
    )


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Strong rainbow connection number",
    notation=r"\mathrm{src}(G)",
    category="advanced colorings",
    aliases=("strong edge rainbow connection number",),
    definition=(
        "The strong rainbow connection number src(G) is the minimum number of "
        "colors in an edge-coloring of G such that every pair of vertices is "
        "joined by a rainbow shortest path."
    ),
)
def strong_rainbow_connection_number(
    G: GraphLike,
    *,
    k_max: Optional[int] = None,
    **solver_kwargs,
) -> int:
    r"""
    Compute the strong rainbow connection number of a connected graph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        A connected undirected simple graph.
    k_max : int, optional
        Upper bound on the number of colors tested. If omitted, uses
        :math:`|E(G)|`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Optional solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra options for the solver backend.
    verbose : bool, default=False
        If True, show solver output when supported.

    Returns
    -------
    int
        The strong rainbow connection number :math:`\mathrm{src}(G)`.

    Raises
    ------
    ValueError
        If ``G`` is disconnected.
    """
    return _minimum_rainbow_k(
        G,
        strong=True,
        k_max=k_max,
        **solver_kwargs,
    )
