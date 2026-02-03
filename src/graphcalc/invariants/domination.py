
from typing import Union, Set, Hashable, Dict, List, Tuple
import networkx as nx
from itertools import combinations
import pulp
from pulp import value

import graphcalc as gc
from graphcalc.core.neighborhoods import neighborhood, closed_neighborhood
from graphcalc.utils import get_default_solver, enforce_type, GraphLike, _extract_and_report, SimpleGraph
from graphcalc.solvers import with_solver

__all__ = [
    "is_dominating_set",
    "minimum_dominating_set",
    "domination_number",
    "minimum_total_domination_set",
    "total_domination_number",
    "minimum_connected_dominating_set",
    "connected_domination_number",
    "minimum_independent_dominating_set",
    "independent_domination_number",
    "complement_is_connected",
    "is_outer_connected_dominating_set",
    "outer_connected_domination_number",
    "minimum_roman_dominating_function",
    "roman_domination_number",
    "minimum_double_roman_dominating_function",
    "double_roman_domination_number",
    "minimum_rainbow_dominating_function",
    "rainbow_domination_number",
    "two_rainbow_domination_number",
    "three_rainbow_domination_number",
    "min_maximal_matching_number",
    "restrained_domination_number",
    "minimum_restrained_dominating_set",
    "minimum_outer_connected_dominating_set",
]

@enforce_type(0, (nx.Graph, SimpleGraph))
def is_dominating_set(
    G: GraphLike,
    S: Union[Set[Hashable], List[Hashable]],
) -> bool:
    r"""
    Checks if a given set of nodes, :math:`S\subseteq V`, is a dominating set in the graph :math:`G`.

    A dominating set of a graph :math:`G = (V, E)` is a subset of nodes :math:`S \subseteq V` such that every node in :math:`V` is either in :math:`S` or
    adjacent to a node in :math:`S`. In other words, every node in the graph is either part of the dominating set or is
    "dominated" by a node in the dominating set.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    S : set
        A subset of nodes in the graph to check for domination.

    Returns
    -------
    bool
        True if :math:`S` is a dominating set, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> S = {0, 2}
    >>> print(gc.is_dominating_set(G, S))
    True

    >>> S = {0}
    >>> print(gc.is_dominating_set(G, S))
    False
    """
    return all(any(u in S for u in closed_neighborhood(G, v)) for v in G.nodes())

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_dominating_set(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Find a minimum dominating set of :math:`G` via integer programming.

    Let :math:`x_v \in \{0,1\}` indicate whether :math:`v` is chosen. We solve:

    .. math::
        \min \sum_{v \in V} x_v \quad \text{s.t. } \sum_{u \in N[v]} x_u \ge 1 \;\; \forall v\in V,

    where :math:`N[v]` is the **closed** neighborhood of :math:`v`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Notes
    -----
    Accepts the standard solver kwargs from :func:`graphcalc.solvers.with_solver`
    (e.g., ``solver="highs"`` or ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Returns
    -------
    set of hashable
        A minimum dominating set of nodes.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> S = gc.minimum_dominating_set(G)
    >>> len(S)
    2
    """
    prob = pulp.LpProblem("MinDominatingSet", pulp.LpMinimize)

    # One binary var per node
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}

    # Objective
    prob += pulp.lpSum(x.values())

    # Coverage constraints using CLOSED neighborhood
    for v in G.nodes():
        Nclosed = closed_neighborhood(G, v)  # ensure this is imported
        prob += pulp.lpSum(x[u] for u in Nclosed) >= 1, f"cover_{v}"

    # Solve (raises if not Optimal)
    solve(prob)

    # Extract solution
    return _extract_and_report(prob, x, verbose=verbose)

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def domination_number(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> int:
    r"""
    Calculates the domination number of the graph :math:`G`.

    The domination number is the size of the smallest dominating set in :math:`G`. It represents the minimum number of nodes
    required such that every node in the graph is either in the dominating set or adjacent to a node in the set.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The domination number of G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.domination_number(G)
    2
    """
    return len(minimum_dominating_set(G, verbose=verbose))

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_total_domination_set(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Find a minimum **total** dominating set of :math:`G` via integer programming.

    Let :math:`x_v \in \{0,1\}` indicate whether :math:`v` is chosen. We solve

    .. math::
        \min \sum_{v \in V} x_v
        \quad \text{s.t.} \quad
        \sum_{u \in N(v)} x_u \ge 1 \;\; \forall v \in V,

    where :math:`N(v)` is the **open** neighborhood of :math:`v`
    (no vertex dominates itself).

    Notes
    -----
    * If :math:`G` has an isolated vertex (degree 0), no total dominating set exists.
      This function raises ``ValueError`` in that case.
    * Accepts standard solver kwargs via :func:`graphcalc.solvers.with_solver`
      (e.g., ``solver="highs"``, ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Returns
    -------
    set of hashable
        A minimum total dominating set.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> S = gc.minimum_total_domination_set(G)
    >>> len(S)
    2
    """
    # Infeasible if any isolated vertices exist
    iso = [v for v, d in G.degree() if d == 0]
    if iso:
        raise ValueError(f"Total domination is undefined: graph has isolated vertices {iso!r}.")

    prob = pulp.LpProblem("MinTotalDominatingSet", pulp.LpMinimize)

    # Binary variable per vertex
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}

    # Objective
    prob += pulp.lpSum(x.values())

    # Coverage constraints using OPEN neighborhood
    for v in G.nodes():
        Nv = neighborhood(G, v)  # must be open neighborhood: neighbors of v
        # Nv is non-empty because we ruled out isolates
        prob += pulp.lpSum(x[u] for u in Nv) >= 1, f"cover_{v}"

    # Solve (raises if not Optimal)
    solve(prob)

    # Extract chosen vertices
    return _extract_and_report(prob, x, verbose=verbose)

@enforce_type(0, (nx.Graph, SimpleGraph))
def total_domination_number(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options) to the MIP
) -> int:
    r"""
    Return the **total domination number** of :math:`G`.

    The total domination number is the size of the smallest set :math:`S`
    such that every vertex is adjacent to some vertex in :math:`S`
    (no vertex may dominate itself).

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_total_domination_set`.

    Returns
    -------
    int
        The total domination number.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> gc.total_domination_number(path_graph(4))
    2
    """
    return len(minimum_total_domination_set(G, **solver_kwargs))

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_independent_dominating_set(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Find a minimum **independent dominating set** of :math:`G` via integer programming.

    An **independent dominating set** is a set :math:`S \subseteq V(G)` such that:

    - (**Independent**) no two vertices of :math:`S` are adjacent, and
    - (**Dominating**) every vertex of :math:`G` is in :math:`S` or adjacent to a vertex in :math:`S`.

    Equivalently, :math:`S` is both an independent set and a dominating set.

    Let :math:`x_v \in \{0,1\}` indicate whether :math:`v` is chosen. We solve

    .. math::
        \min \sum_{v \in V} x_v

    subject to the **independence** constraints

    .. math::
        x_u + x_v \le 1 \quad \forall \{u,v\}\in E,

    and the **domination** constraints (using the closed neighborhood)

    .. math::
        \sum_{u \in N[v]} x_u \ge 1 \quad \forall v \in V.

    Notes
    -----
    Accepts the standard solver kwargs from :func:`graphcalc.solvers.with_solver`
    (e.g., ``solver="highs"``, ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Returns
    -------
    set of hashable
        A minimum independent dominating set.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> S = gc.minimum_independent_dominating_set(G)
    >>> len(S)
    2
    """

    prob = pulp.LpProblem("MinIndependentDominatingSet", pulp.LpMinimize)

    # One binary var per vertex
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}

    # Objective
    prob += pulp.lpSum(x.values())

    # Independence: adjacent vertices cannot both be chosen
    for u, v in G.edges():
        prob += x[u] + x[v] <= 1, f"indep_{u}_{v}"

    # Domination: closed neighborhood covers each vertex
    for v in G.nodes():
        Nv_closed = closed_neighborhood(G, v)  # includes v itself
        prob += pulp.lpSum(x[u] for u in Nv_closed) >= 1, f"dom_{v}"

    # Solve (raises if not Optimal)
    solve(prob)

    # Extract chosen vertices
    return _extract_and_report(prob, x, verbose=verbose)

@enforce_type(0, (nx.Graph, SimpleGraph))
def independent_domination_number(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options)
) -> int:
    r"""
    Return the **independent domination number** of :math:`G`.

    An independent dominating set is a dominating set that is also an
    independent set. This wraps :func:`minimum_independent_dominating_set`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_independent_dominating_set`.

    Returns
    -------
    int
        The independent domination number of :math:`G`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> gc.independent_domination_number(path_graph(4))
    2
    """
    return len(minimum_independent_dominating_set(G, **solver_kwargs))

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_connected_dominating_set(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Find a minimum connected dominating set of :math:`G` via integer programming.

    A **connected dominating set** is a set :math:`S \subseteq V(G)` such that:

    - (**Dominating**) every vertex is in :math:`S` or adjacent to a vertex in :math:`S`.
    - (**Connected**) the induced subgraph :math:`G[S]` is connected.

    Let :math:`x_v \in \{0,1\}` indicate whether :math:`v` is selected. Domination is enforced by

    .. math::
        \sum_{u \in N[v]} x_u \ge 1 \quad \forall v \in V,

    where :math:`N[v]` is the closed neighborhood of :math:`v`.

    Connectivity is enforced with a single-commodity flow formulation that chooses a root
    among the selected vertices and sends one unit of flow to each other selected vertex.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Notes
    -----
    Accepts the standard solver kwargs from :func:`graphcalc.solvers.with_solver`
    (e.g., ``solver="highs"`` or ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Conventions
    -----------
    - If :math:`|V(G)|=0`, returns the empty set.
    - If :math:`G` is disconnected and nonempty, no connected dominating set exists and this function raises ``ValueError``.

    Returns
    -------
    set of hashable
        A minimum connected dominating set.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, cycle_graph
    >>> len(gc.minimum_connected_dominating_set(path_graph(4)))
    2
    >>> len(gc.minimum_connected_dominating_set(cycle_graph(6)))
    4
    """
    import pulp

    n = G.number_of_nodes()
    if n == 0:
        return set()

    # For disconnected graphs, no connected dominating set exists.
    if not nx.is_connected(G):
        raise ValueError(
            "minimum_connected_dominating_set is defined only for connected graphs (or the empty graph)."
        )

    nodes = list(G.nodes())

    prob = pulp.LpProblem("MinConnectedDominatingSet", pulp.LpMinimize)

    # Selection variables
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in nodes}

    # Root choice among selected vertices
    r = {v: pulp.LpVariable(f"r_{v}", cat="Binary") for v in nodes}
    prob += pulp.lpSum(r.values()) == 1, "one_root"
    for v in nodes:
        prob += r[v] <= x[v], f"root_implies_selected_{v}"

    # K = |S|
    K = pulp.LpVariable("K", lowBound=1, upBound=n, cat="Integer")
    prob += K == pulp.lpSum(x.values()), "K_def"

    # Flow variables on directed arcs
    arcs = []
    for u, v in G.edges():
        arcs.append((u, v))
        arcs.append((v, u))

    M = n  # big-M
    f = {(u, v): pulp.LpVariable(f"f_{u}_{v}", lowBound=0, upBound=n, cat="Continuous") for (u, v) in arcs}

    # Flow can traverse only selected vertices
    for u, v in arcs:
        prob += f[(u, v)] <= M * x[u], f"cap_tail_{u}_{v}"
        prob += f[(u, v)] <= M * x[v], f"cap_head_{u}_{v}"

    # Linearize z_v = K * r_v
    z = {v: pulp.LpVariable(f"z_{v}", lowBound=0, upBound=n, cat="Continuous") for v in nodes}
    for v in nodes:
        prob += z[v] <= K, f"z_le_K_{v}"
        prob += z[v] <= n * r[v], f"z_le_nrv_{v}"
        prob += z[v] >= K - n * (1 - r[v]), f"z_ge_K_minus_n_{v}"
        prob += z[v] >= 0, f"z_ge_0_{v}"

    # Objective
    prob += pulp.lpSum(x.values())

    # Domination constraints (closed neighborhoods)
    for v in nodes:
        Nclosed = closed_neighborhood(G, v)
        prob += pulp.lpSum(x[u] for u in Nclosed) >= 1, f"dom_{v}"

    # Flow conservation: inflow - outflow = x[v] - z[v]
    for v in nodes:
        inflow = pulp.lpSum(f[(u, v)] for u in G.neighbors(v))
        outflow = pulp.lpSum(f[(v, u)] for u in G.neighbors(v))
        prob += inflow - outflow == x[v] - z[v], f"flow_{v}"

    # Solve (raises if not Optimal)
    solve(prob)

    return _extract_and_report(prob, x, verbose=verbose)

@enforce_type(0, (nx.Graph, SimpleGraph))
def connected_domination_number(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options)
) -> int:
    r"""
    Return the **connected domination number** :math:`\gamma_c(G)`.

    The connected domination number is the minimum size of a connected dominating set:

    .. math::
        \gamma_c(G) = \min\{ |S| : S \subseteq V(G),\ S \text{ dominates } G,\ \text{and } G[S]\text{ is connected}\}.

    This wraps :func:`minimum_connected_dominating_set`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_connected_dominating_set`.

    Returns
    -------
    int
        The connected domination number :math:`\gamma_c(G)`.

    Raises
    ------
    ValueError
        If :math:`G` is disconnected and nonempty.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, cycle_graph
    >>> gc.connected_domination_number(path_graph(3))
    1
    >>> gc.connected_domination_number(path_graph(4))
    2
    >>> gc.connected_domination_number(cycle_graph(6))
    4
    """
    return len(minimum_connected_dominating_set(G, **solver_kwargs))

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def complement_is_connected(G: GraphLike, S: Union[Set[Hashable], List[Hashable]]) -> bool:
    r"""
    Checks if the complement of a set :math:`S` in the graph :math:`G` induces a connected subgraph.

    The complement of :math:`S` is defined as the set of all nodes in :math:`G` that are not in :math:`S`. This function verifies
    whether the subgraph induced by the complement of :math:`S` is connected.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    S : set
        A subset of nodes in the graph.

    Returns
    -------
    bool
        True if the subgraph induced by the complement of S is connected, otherwise False.

    Examples
    --------

    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> S = {0}
    >>> gc.complement_is_connected(G, S)
    True

    >>> S = {0, 2}
    >>> gc.complement_is_connected(G, S)
    False
    """
    X = G.nodes() - S
    return nx.is_connected(G.subgraph(X))

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def is_outer_connected_dominating_set(G: GraphLike, S: Union[Set[Hashable], List[Hashable]]) -> bool:
    r"""
    Checks if a given set :math:`S` is an outer-connected dominating set in the graph :math:`G`.

    An outer-connected dominating set :math:`S \subseteq V` of a graph :math:`G = (V, E)` is a dominating set such that the subgraph
    induced by the complement of :math:`S` is connected.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    S : set
        A subset of nodes in the graph.

    Returns
    -------
    bool
        True if S is an outer-connected dominating set, otherwise False.

    Examples
    --------

    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> S = {0, 2, 4}
    >>> gc.is_outer_connected_dominating_set(G, S)
    False

    >>> S = {0, 1, 2}
    >>> gc.is_outer_connected_dominating_set(G, S)
    True
    """
    return is_dominating_set(G, S) and complement_is_connected(G, S)

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def minimum_outer_connected_dominating_set(G: GraphLike) -> Set[Hashable]:
    r"""
    Finds a minimum outer-connected dominating set for the graph :math:`G` by trying all subset sizes.

    Parameters
    ----------
    G : networkx.Graph

    Returns
    -------
    set
        A minimum outer-connected dominating set of nodes in G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> optimal_set = gc.minimum_outer_connected_dominating_set(G)
    """
    n = len(G.nodes())

    for r in range(1, n + 1):  # Try all subset sizes
        for S in combinations(G.nodes(), r):
            S = set(S)
            if is_outer_connected_dominating_set(G, S):
                return S

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def outer_connected_domination_number(G: GraphLike) -> int:
    r"""
    Finds a minimum outer-connected dominating set for the graph :math:`G`.

    A minimum outer-connected dominating set is the smallest subset :math:`S \subseteq V` of the graph :math:`G` such that:
      1. :math:`S` is a dominating set.
      2. The subgraph induced by the complement of :math:`S` is connected.

    This function tries all subset sizes to find the smallest outer-connected dominating set.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    set
        A minimum outer-connected dominating set of nodes in G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.outer_connected_domination_number(G)
    2

    Notes
    -----
    This implementation is exponential in complexity :math:`O(2^n)`, as it tries all subsets of nodes in the graph.
    It is not suitable for large graphs.
    """
    return len(minimum_outer_connected_dominating_set(G))

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_roman_dominating_function(
    graph: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Dict[str, Dict[Hashable, int] | float]:
    r"""
    Compute a minimum Roman dominating function (RDF) via integer programming.

    A **Roman dominating function** on :math:`G=(V,E)` is a map :math:`f:V\to\{0,1,2\}`
    such that every vertex with label 0 has a neighbor with label 2. The **weight** of
    :math:`f` is :math:`\sum_{v\in V} f(v)`.

    We minimize the weight using binary indicators:

    - :math:`x_v=1` iff :math:`f(v)=1`
    - :math:`y_v=1` iff :math:`f(v)=2`

    so that :math:`f(v)=x_v+2y_v`.

    Formulation
    -----------
    .. math::
        \min \sum_{v\in V} (x_v + 2y_v)

    .. math::
        x_v + y_v + \sum_{u\in N(v)} y_u \ge 1 \quad \forall v\in V

    .. math::
        x_v + y_v \le 1 \quad \forall v\in V

    Parameters
    ----------
    graph : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Notes
    -----
    Accepts the standard solver kwargs from :func:`graphcalc.solvers.with_solver`
    (e.g., ``solver="highs"``, ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"x"``: dict mapping ``v`` to 0/1 indicating whether :math:`f(v)=1`
        - ``"y"``: dict mapping ``v`` to 0/1 indicating whether :math:`f(v)=2`
        - ``"objective"``: float, the minimum weight :math:`\sum_v (x_v + 2y_v)`

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> sol = gc.minimum_roman_dominating_function(G)
    >>> isinstance(sol["objective"], float)
    True
    """

    prob = pulp.LpProblem("RomanDomination", pulp.LpMinimize)

    # Binary variables per vertex
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in graph.nodes()}
    y = {v: pulp.LpVariable(f"y_{v}", cat="Binary") for v in graph.nodes()}

    # Objective: min sum(x_v + 2*y_v)
    prob += pulp.lpSum(x[v] + 2 * y[v] for v in graph.nodes())

    # Roman domination constraints: x_v + y_v + sum_{u∈N(v)} y_u ≥ 1
    for v in graph.nodes():
        Nv = list(graph.neighbors(v))  # open neighborhood
        prob += x[v] + y[v] + pulp.lpSum(y[u] for u in Nv) >= 1, f"dom_{v}"

    # Exclusivity: x_v + y_v ≤ 1
    for v in graph.nodes():
        prob += x[v] + y[v] <= 1, f"excl_{v}"

    # Solve (raises if not Optimal)
    solve(prob)

    # Extract solution
    sol_x = {v: int(round(pulp.value(x[v]) or 0)) for v in graph.nodes()}
    sol_y = {v: int(round(pulp.value(y[v]) or 0)) for v in graph.nodes()}
    obj = float(pulp.value(prob.objective))

    if verbose:
        print(f"Objective: {obj}")
        # (Optional) print a compact labeling summary
        # print({v: sol_x[v] + 2*sol_y[v] for v in graph.nodes()})

    return {"x": sol_x, "y": sol_y, "objective": obj}


@enforce_type(0, (nx.Graph, SimpleGraph))
def roman_domination_number(
    graph: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options) to the MIP
) -> int:
    r"""
    Return the Roman domination number :math:`\gamma_R(G)`.

    Parameters
    ----------
    graph : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_roman_dominating_function`.

    Returns
    -------
    int
        :math:`\gamma_R(G)`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> gc.roman_domination_number(path_graph(4))
    3
    """
    sol = minimum_roman_dominating_function(graph, **solver_kwargs)
    # objective is integer-valued; guard against tiny float noise
    return int(round(sol["objective"]))

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_double_roman_dominating_function(
    graph: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Dict[str, Dict[Hashable, int] | float]:
    r"""
    Compute a minimum double Roman dominating function (DRDF) via integer programming.

    A **double Roman dominating function** is a labeling
    :math:`f:V(G)\to\{0,1,2,3\}` such that:

    1. If :math:`f(v)=0`, then either some neighbor of :math:`v` has label 3, or at least
    two neighbors of :math:`v` have label 2.
    2. If :math:`f(v)=1`, then some neighbor of :math:`v` has label at least 2.

    We use binary indicators for each vertex :math:`v`:

    - :math:`x_v=1` iff :math:`f(v)=1`
    - :math:`y_v=1` iff :math:`f(v)=2`
    - :math:`z_v=1` iff :math:`f(v)=3`

    with exclusivity :math:`x_v+y_v+z_v\le 1`. The objective is

    .. math::
        \min \sum_{v\in V} (x_v + 2y_v + 3z_v).

    Formulation
    -----------
    Domination constraint for vertices that may be labeled 0 (linearized):

    .. math::
        x_v + y_v + z_v
        \;+\; \tfrac{1}{2}\sum_{u\in N(v)} y_u
        \;+\; \sum_{u\in N(v)} z_u
        \;\ge\; 1
        \quad \forall v\in V.

    Domination constraint for vertices labeled 1:

    .. math::
        \sum_{u\in N(v)} (y_u + z_u) \;\ge\; x_v
        \quad \forall v\in V.

    Exclusivity:

    .. math::
        x_v + y_v + z_v \;\le\; 1
        \quad \forall v\in V.

    Parameters
    ----------
    graph : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Notes
    -----
    Accepts standard solver kwargs via :func:`graphcalc.solvers.with_solver`
    (e.g., ``solver="highs"`` or ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"x"``: dict mapping ``v`` to 0/1 indicating whether :math:`f(v)=1`
        - ``"y"``: dict mapping ``v`` to 0/1 indicating whether :math:`f(v)=2`
        - ``"z"``: dict mapping ``v`` to 0/1 indicating whether :math:`f(v)=3`
        - ``"objective"``: float, the minimum value :math:`\sum_v (x_v + 2y_v + 3z_v)`

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> sol = gc.minimum_double_roman_dominating_function(G)  # doctest: +ELLIPSIS
    >>> isinstance(sol["objective"], float)
    True
    """
    prob = pulp.LpProblem("DoubleRomanDomination", pulp.LpMinimize)

    # Binary indicators per vertex
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in graph.nodes()}
    y = {v: pulp.LpVariable(f"y_{v}", cat="Binary") for v in graph.nodes()}
    z = {v: pulp.LpVariable(f"z_{v}", cat="Binary") for v in graph.nodes()}

    # Objective
    prob += pulp.lpSum(x[v] + 2 * y[v] + 3 * z[v] for v in graph.nodes())

    # Constraints
    for v in graph.nodes():
        Nv = list(graph.neighbors(v))  # open neighborhood
        # (1) 0-labeled coverage linearization
        prob += (
            x[v] + y[v] + z[v]
            + 0.5 * pulp.lpSum(y[u] for u in Nv)
            + pulp.lpSum(z[u] for u in Nv)
            >= 1
        ), f"dom0_{v}"
        # (2) 1-labeled coverage
        prob += pulp.lpSum(y[u] + z[u] for u in Nv) >= x[v], f"dom1_{v}"
        # (3) exclusivity
        prob += x[v] + y[v] + z[v] <= 1, f"excl_{v}"

    # Solve (raises if not Optimal)
    solve(prob)

    # Extract solution (guarding against tiny float noise)
    sol_x = {v: int(round(pulp.value(x[v]) or 0)) for v in graph.nodes()}
    sol_y = {v: int(round(pulp.value(y[v]) or 0)) for v in graph.nodes()}
    sol_z = {v: int(round(pulp.value(z[v]) or 0)) for v in graph.nodes()}
    obj = float(pulp.value(prob.objective))

    if verbose:
        print(f"Objective: {obj}")

    return {"x": sol_x, "y": sol_y, "z": sol_z, "objective": obj}

@enforce_type(0, (nx.Graph, SimpleGraph))
def double_roman_domination_number(
    graph: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options)
) -> int:
    r"""
    Return the double Roman domination number :math:`\gamma_{dR}(G)`.

    Parameters
    ----------
    graph : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_double_roman_dominating_function`.

    Returns
    -------
    int
        :math:`\gamma_{dR}(G)`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> gc.double_roman_domination_number(path_graph(4))
    5
    """
    sol = minimum_double_roman_dominating_function(graph, **solver_kwargs)
    return int(round(sol["objective"]))


# Type alias for return
ColoredPairs = List[Tuple[Hashable, int]]
UncoloredList = List[Hashable]

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_rainbow_dominating_function(
    G: GraphLike,
    k: int,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Tuple[ColoredPairs, UncoloredList]:
    r"""
    Compute a minimum $k$-rainbow dominating function (RDF) via integer programming.

    A rainbow dominating function on $G=(V,E)$ with parameter $k$ assigns to each
    vertex either one of $k$ colors or leaves it uncolored, such that every
    uncolored vertex is adjacent to at least one vertex of each of the $k$ colors.
    The objective is to minimize the number of colored vertices.

    Variables
    ---------
    - $f_{v,i} \in \{0,1\}$ — vertex $v$ is colored with color $i \in \{1,\dots,k\}$.
    - $x_v \in \{0,1\}$ — vertex $v$ is uncolored.

    Objective
    ---------
    .. math:: \min \sum_{v\in V} \sum_{i=1}^k f_{v,i}

    Constraints
    -----------
    - Exactly one choice per vertex:
      .. math:: \sum_{i=1}^k f_{v,i} + x_v = 1 \quad \forall v \in V.
    - Rainbow domination (only applies when uncolored):
      .. math:: \sum_{u \in N(v)} f_{u,i} \ge x_v \quad \forall v\in V,\; i=1,\dots,k.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.
    k : int
        Number of colors (must be >= 1).
    verbose : bool, default=False
        If True, print solver output (when supported).

    Notes
    -----
    Accepts the standard solver kwargs via :func:`graphcalc.solvers.with_solver`
    (e.g., ``solver="highs"``, ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Returns
    -------
    (colored_vertices, uncolored_vertices) : (list[tuple], list)
        ``colored_vertices`` is a list of ``(vertex, color)`` pairs;
        ``uncolored_vertices`` is a list of vertices with no color.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> colored, uncolored = gc.minimum_rainbow_dominating_function(G, 2)
    >>> len(colored) + len(uncolored) == gc.order(G)
    True
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer (k >= 1).")

    nodes = list(G.nodes())
    colors = range(1, k + 1)

    prob = pulp.LpProblem("Rainbow_Domination", pulp.LpMinimize)

    # Decision variables
    f = {(v, i): pulp.LpVariable(f"f_{v}_{i}", cat="Binary") for v in nodes for i in colors}
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in nodes}

    # Objective: minimize number of colored vertices
    prob += pulp.lpSum(f[v_i] for v_i in f.keys())

    # Each vertex either colored with exactly one color or uncolored
    for v in nodes:
        prob += pulp.lpSum(f[(v, i)] for i in colors) + x[v] == 1, f"choice_{v}"

    # Uncolored vertex must see all colors in its open neighborhood
    for v in nodes:
        Nv = list(G.neighbors(v))
        for i in colors:
            prob += pulp.lpSum(f[(u, i)] for u in Nv) >= x[v], f"rainbow_{v}_{i}"

    # Solve (raises if not Optimal)
    solve(prob)

    # Extract solution
    colored_vertices: ColoredPairs = [
        (v, i) for v in nodes for i in colors if (pulp.value(f[(v, i)]) or 0) > 0.5
    ]
    uncolored_vertices: UncoloredList = [v for v in nodes if (pulp.value(x[v]) or 0) > 0.5]

    if verbose:
        obj = float(pulp.value(prob.objective))
        print(f"Objective (number of colored vertices): {obj}")

    return colored_vertices, uncolored_vertices


@enforce_type(0, (nx.Graph, SimpleGraph))
def rainbow_domination_number(
    G: GraphLike,
    k: int,
    **solver_kwargs,  # forwards (verbose, solver, solver_options) to the MIP
) -> int:
    r"""
    Return the rainbow domination number $\gamma_{r,k}(G)$.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.
    k : int
        Number of colors (>= 1).

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_rainbow_dominating_function`.

    Returns
    -------
    int
        $\gamma_{r,k}(G)$ — the minimum number of colored vertices.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.rainbow_domination_number(G, 2)
    3
    """
    colored, _ = minimum_rainbow_dominating_function(G, k, **solver_kwargs)
    return len(colored)


@enforce_type(0, (nx.Graph, SimpleGraph))
def two_rainbow_domination_number(
    G: GraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the 2-rainbow domination number $\gamma_{r,2}(G)$.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_rainbow_dominating_function`.

    Returns
    -------
    int
        $\gamma_{r,2}(G)$.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> gc.two_rainbow_domination_number(path_graph(4))
    3
    """
    return rainbow_domination_number(G, 2, **solver_kwargs)


@enforce_type(0, (nx.Graph, SimpleGraph))
def three_rainbow_domination_number(
    G: GraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Return the 3-rainbow domination number $\gamma_{r,3}(G)$.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_rainbow_dominating_function`.

    Returns
    -------
    int
        $\gamma_{r,3}(G)$.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> gc.three_rainbow_domination_number(path_graph(4))
    4
    """
    return rainbow_domination_number(G, 3, **solver_kwargs)

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def minimum_restrained_dominating_set(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Compute a minimum **restrained dominating set** (RDS) via integer programming.

    Let :math:`x_v \in \{0,1\}` indicate whether :math:`v` is in the set :math:`S`.
    We minimize :math:`\sum_v x_v` subject to:

    - **Domination:** every vertex is dominated:
      .. math:: x_v + \sum_{u \in N(v)} x_u \ge 1 \quad \forall v\in V.
    - **Restraint:** no isolates in the complement :math:`V\setminus S`:
      .. math:: \sum_{u \in N(v)} (1-x_u) \ge 1 - x_v \quad \forall v\in V.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Notes
    -----
    Accepts standard solver kwargs via :func:`graphcalc.solvers.with_solver`
    (e.g., ``solver="highs"``, ``solver={"name":"GUROBI_CMD","options":{...}}``).

    Returns
    -------
    set of hashable
        A minimum restrained dominating set.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(5)
    >>> S = gc.minimum_restrained_dominating_set(G)
    >>> len(S) >= 1
    True
    """
    prob = pulp.LpProblem("MinimumRestrainedDomination", pulp.LpMinimize)

    # Decision vars: x_v = 1 if v in S
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}

    # Objective
    prob += pulp.lpSum(x.values())

    # Domination constraints
    for v in G.nodes():
        prob += x[v] + pulp.lpSum(x[u] for u in G.neighbors(v)) >= 1, f"dom_{v}"

    # Restraint constraints: no isolates in complement
    # (equivalently: deg(v) - sum_{u in N(v)} x_u >= 1 - x_v)
    for v in G.nodes():
        prob += pulp.lpSum(1 - x[u] for u in G.neighbors(v)) >= (1 - x[v]), f"rest_{v}"

    # Solve (raises if not Optimal)
    solve(prob)

    # Extract chosen vertices
    return _extract_and_report(prob, x, verbose=verbose)

@enforce_type(0, (nx.Graph, SimpleGraph))
def restrained_domination_number(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options)
) -> int:
    r"""
    Return the restrained domination number :math:`\gamma_r(G)`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        Undirected input graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
    solver_options : dict, optional
        Forwarded to :func:`minimum_restrained_dominating_set`.

    Returns
    -------
    int
        :math:`\gamma_r(G)`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> gc.restrained_domination_number(path_graph(5))
    3
    """
    return len(minimum_restrained_dominating_set(G, **solver_kwargs))


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def min_maximal_matching_number(G: GraphLike) -> int:
    r"""
    Calculates the minimum maximal matching number of the graph G.

    The minimum maximal matching number of G is the size of a minimum maximal matching
    in G. This is equivalent to finding the domination number of the line graph of G.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The minimum maximal matching number of G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.min_maximal_matching_number(G)
    1
    """
    return domination_number(nx.line_graph(G))
