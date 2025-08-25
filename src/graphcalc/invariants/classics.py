from typing import Set, Hashable, Dict, Tuple, Hashable
import pulp
import itertools
import networkx as nx

from graphcalc.core import SimpleGraph
from graphcalc.utils import (
    get_default_solver, enforce_type, GraphLike, _extract_and_report
)
from graphcalc.solvers import with_solver

__all__ = [
    "maximum_independent_set",
    "independence_number",
    "maximum_clique",
    "clique_number",
    "optimal_proper_coloring",
    "chromatic_number",
    "minimum_vertex_cover",
    "minimum_edge_cover",
    "vertex_cover_number",
    "edge_cover_number",
    "maximum_matching",
    "matching_number",
    "triameter",
]

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def maximum_independent_set(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""Return a largest independent set of nodes in *G*.

    This method formulates the maximum independent set problem as an integer
    linear program:

    .. math::
        \max \sum_{v \in V} x_v

    subject to

    .. math::
        x_u + x_v \leq 1 \quad \text{for all } \{u, v\} \in E,

    where *E* and *V* are the edge and vertex sets of *G*, and
    :math:`x_v \in \{0,1\}` for each vertex.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    verbose : bool, default=False
        If True, print solver output (when supported) and intermediate results
        during optimization. If False, run silently.

    Notes
    -----
    This function also accepts the standard solver kwargs provided by
    :func:`graphcalc.solvers.with_solver`, e.g. ``solver="highs"`` or
    ``solver={"name":"GUROBI_CMD","options":{"timeLimit":10}}``.

    Returns
    -------
    set of hashable
        A set of nodes comprising a largest independent set in *G*.

    Raises
    ------
    ValueError
        If no optimal solution is found by the solver.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> S = gc.maximum_independent_set(G)
    >>> len(S)
    1
    >>> # Optionally choose a specific solver
    >>> S = gc.maximum_independent_set(G, solver="cbc")  # doctest: +SKIP
    >>> len(S)
    1
    """
    # Build IP
    prob = pulp.LpProblem("MaximumIndependentSet", pulp.LpMaximize)

    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}
    prob += pulp.lpSum(x[v] for v in G.nodes())

    for u, v in G.edges():
        prob += x[u] + x[v] <= 1, f"edge_{u}_{v}"

    # Uniform solve (provided by @with_solver)
    solve(prob)

    return _extract_and_report(prob, x, verbose=verbose)

@enforce_type(0, (nx.Graph, SimpleGraph))
def independence_number(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options) to MIS
) -> int:
    r"""
    Return the size of a largest independent set in *G*.

    An **independent set** in a graph :math:`G=(V,E)` is a subset
    :math:`S \subseteq V` such that no two vertices in :math:`S`
    are adjacent. The **independence number** :math:`\alpha(G)`
    is defined as

    .. math::
        \alpha(G) = \max \{ |S| : S \subseteq V, \, S \text{ is independent} \}.

    Notes
    -----
    * The independence number is NP-hard to compute in general.
    * This implementation calls :func:`maximum_independent_set`,
      which formulates the problem as a mixed integer program (MIP).
    * Relations:

      - Complement: :math:`\alpha(G) = \omega(\overline{G})`.
      - Bound: :math:`\alpha(G) \ge \frac{|V|}{\Delta(G)+1}` (Caro–Wei).

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
        Passed through to the solver via :func:`graphcalc.solvers.with_solver`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver spec handled by :func:`graphcalc.solvers.resolve_solver`.
    solver_options : dict, optional
        Extra kwargs used when constructing the solver if needed.

    Returns
    -------
    int
        The independence number :math:`\alpha(G)` of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph, cycle_graph
    >>> gc.independence_number(complete_graph(4))
    1
    >>> gc.independence_number(cycle_graph(5))
    2
    """
    return len(maximum_independent_set(G, **solver_kwargs))

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def maximum_clique(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Return a maximum clique of nodes in *G* using integer programming.

    We choose binary variables :math:`x_v \in \{0,1\}` for each vertex :math:`v`,
    maximize the selected vertices, and forbid selecting non-adjacent pairs:

    Objective
    ---------
    .. math::
        \max \sum_{v \in V} x_v

    Constraints
    -----------
    .. math::
        x_u + x_v \le 1 \quad \text{for every non-edge } \{u,v\} \notin E,

    which ensures the chosen vertices induce a clique.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    verbose : bool, default=False
        If True, print solver output (when supported) and intermediate details.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Specification of the solver backend. Same accepted forms as in
        :func:`maximum_independent_set`. If None, uses :func:`get_default_solver`.
    solver_options : dict, optional
        Extra keyword arguments when constructing the solver if ``solver`` is a
        string or class. Ignored if ``solver`` is an existing object.

    Returns
    -------
    set of hashable
        A set of nodes forming a maximum clique in *G*.

    Raises
    ------
    ValueError
        If no optimal solution is found by the solver.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> gc.maximum_clique(complete_graph(4)) == {0, 1, 2, 3}
    True

    Optionally specify a solver (skipped in doctest since availability varies):

    >>> from pulp import HiGHS_CMD
    >>> gc.maximum_clique(complete_graph(4), solver=HiGHS_CMD) # doctest: +SKIP
    {0, 1, 2, 3}
    """
    # MILP model
    prob = pulp.LpProblem("MaximumClique", pulp.LpMaximize)

    # Binary decision variables x_v for each vertex
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}

    # Objective: maximize number of selected vertices
    prob += pulp.lpSum(x.values())

    # For every non-edge {u,v}, forbid selecting both: x_u + x_v <= 1
    E = {frozenset((u, v)) for (u, v) in G.edges()}
    nodes = list(G.nodes())
    for u, v in itertools.combinations(nodes, 2):
        if frozenset((u, v)) not in E:
            prob += x[u] + x[v] <= 1, f"nonedge_{u}_{v}"

    # Solve (same flexible API as MIS)
    solve(prob)

    # Check status
    if pulp.LpStatus[prob.status] != "Optimal":
        raise ValueError(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")

    # Extract solution
    return _extract_and_report(prob, x, verbose=verbose)

@enforce_type(0, (nx.Graph, SimpleGraph))
def clique_number(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options) to MIS
) -> int:
    r"""
    Compute the clique number :math:`\omega(G)`.

    A **clique** in :math:`G=(V,E)` is a subset :math:`C \subseteq V` such that
    every pair of vertices in :math:`C` is adjacent. The **clique number** is

    .. math::
        \omega(G) = \max \{ |C| : C \subseteq V, \, C \text{ induces a clique} \}.

    Notes
    -----
    * NP-hard in general.
    * This implementation calls :func:`maximum_clique`, which solves a MIP.
    * Relations:
      - Complement: :math:`\omega(G) = \alpha(\overline{G})`.
      - Trivial bound: :math:`\omega(G) \le \Delta(G)+1`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Same solver options as :func:`maximum_clique`.
    solver_options : dict, optional
        Extra keyword arguments used when constructing the solver if needed.

    Returns
    -------
    int
        The clique number :math:`\omega(G)`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph, cycle_graph
    >>> gc.clique_number(complete_graph(4))
    4
    >>> gc.clique_number(cycle_graph(5))
    2
    """
    return len(maximum_clique(G, **solver_kwargs))

@enforce_type(0, (nx.Graph, SimpleGraph))
def optimal_proper_coloring(G: GraphLike) -> Dict:
    r"""Finds the optimal proper coloring of a graph using linear programming.

    This function uses integer linear programming to find the optimal (minimum) number of colors
    required to color the graph :math:`G` such that no two adjacent nodes have the same color. Each node
    is assigned a color represented by a binary variable.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    dict:
        A dictionary where keys are color indices and values are lists of nodes in :math:`G` assigned that color.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> coloring = gc.optimal_proper_coloring(G)
    """
    # Set up the optimization model
    prob = pulp.LpProblem("OptimalProperColoring", pulp.LpMinimize)

    # Define decision variables
    colors = {i: pulp.LpVariable(f"x_{i}", 0, 1, pulp.LpBinary) for i in range(G.order())}
    node_colors = {
        node: [pulp.LpVariable(f"c_{node}_{i}", 0, 1, pulp.LpBinary) for i in range(G.order())] for node in G.nodes()
    }

    # Set the min proper coloring objective function
    prob += pulp.lpSum([colors[i] for i in colors])

    # Set constraints
    for node in G.nodes():
        prob += sum(node_colors[node]) == 1

    for edge, i in itertools.product(G.edges(), range(G.order())):
        prob += sum(node_colors[edge[0]][i] + node_colors[edge[1]][i]) <= 1

    for node, i in itertools.product(G.nodes(), range(G.order())):
        prob += node_colors[node][i] <= colors[i]

    solver = get_default_solver()
    prob.solve(solver)

    # Raise value error if solution not found
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")

    solution_set = {color: [node for node in node_colors if node_colors[node][color].value() == 1] for color in colors}
    return solution_set

@enforce_type(0, (nx.Graph, SimpleGraph))
def chromatic_number(G: GraphLike) -> int:
    r"""
    The chromatic number of a graph is the smallest number of colors needed to color the vertices of :math:`G` so that no two
    adjacent vertices share the same color.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    int
        The chromatic number of :math:`G`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.chromatic_number(G)
    4
    """
    coloring = optimal_proper_coloring(G)
    colors = [color for color in coloring if len(coloring[color]) > 0]
    return len(colors)

@enforce_type(0, (nx.Graph, SimpleGraph))
def minimum_vertex_cover(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options) to MIS
) -> Set[Hashable]:
    r"""Return a smallest vertex cover of :math:`G`.

    A set :math:`X \subseteq V` is a **vertex cover** if every edge has at least
    one endpoint in :math:`X`. By complementarity with independent sets,
    a smallest vertex cover has size :math:`|V| - \alpha(G)` and equals
    :math:`V \setminus S` for any maximum independent set :math:`S`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
        Passed through to the solver used by :func:`maximum_independent_set`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver spec handled by :func:`graphcalc.solvers.resolve_solver`.
    solver_options : dict, optional
        Extra kwargs used when constructing the solver if needed.

    Returns
    -------
    set of hashable
        A smallest vertex cover of :math:`G`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> len(gc.minimum_vertex_cover(G))  # any 3 vertices form a minimum cover
    3
    """
    S = set(maximum_independent_set(G, **solver_kwargs))
    return set(G.nodes()) - S

@enforce_type(0, (nx.Graph, SimpleGraph))
def vertex_cover_number(
    G: GraphLike,
    **solver_kwargs,  # forwards to independence_number (which forwards to MIS)
) -> int:
    r"""Return the size of a smallest vertex cover of :math:`G`.

    Uses :math:`\tau(G) = |V| - \alpha(G)`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
        Passed through to the solver used by :func:`independence_number`.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver spec handled by :func:`graphcalc.solvers.resolve_solver`.
    solver_options : dict, optional
        Extra kwargs used when constructing the solver if needed.

    Returns
    -------
    int
        The vertex cover number :math:`\tau(G)`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> gc.vertex_cover_number(complete_graph(4))
    3
    """
    return G.order() - independence_number(G, **solver_kwargs)

@enforce_type(0, (nx.Graph, SimpleGraph))
def minimum_edge_cover(G: GraphLike):
    r"""Return a smallest edge cover of the graph :math:`G`.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    set
        A smallest edge cover of :math:`G`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> solution = gc.minimum_edge_cover(G)
    """
    return nx.min_edge_cover(G)

@enforce_type(0, (nx.Graph, SimpleGraph))
def edge_cover_number(G: GraphLike) -> int:
    r"""Return the size of a smallest edge cover in the graph :math:`G`.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    number
        The size of a smallest edge cover of :math:`G`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.edge_cover_number(G)
    2
    """
    return len(nx.min_edge_cover(G))

@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def maximum_matching(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Tuple[Hashable, Hashable]]:
    r"""Return a maximum matching in :math:`G` via integer programming.

    A matching is a set of edges with no shared endpoint. We solve:

    .. math::
        \max \sum_{e \in E} x_e \quad \text{s.t. } \sum_{e \in \delta(v)} x_e \le 1 \;\; \forall v\in V,\;
        x_e \in \{0,1\}.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph.
    verbose : bool, default=False
        If True, print solver output (when supported).

    Notes
    -----
    This function accepts the standard solver kwargs provided by
    :func:`graphcalc.solvers.with_solver`, e.g. ``solver="highs"`` or
    ``solver={"name":"GUROBI_CMD","options":{"timeLimit":10}}``.

    Returns
    -------
    set of tuple
        A maximum matching as a set of edges ``(u, v)``.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> M = gc.maximum_matching(G)
    >>> len(M)
    2
    """
    prob = pulp.LpProblem("MaximumMatching", pulp.LpMaximize)

    # Decision variables: one binary per edge (normalize key order for stability)
    def ek(u, v):
        a, b = sorted((u, v))
        return (a, b)

    edges = [ek(u, v) for (u, v) in G.edges()]
    x = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}", cat="Binary") for e in edges}

    # Objective
    prob += pulp.lpSum(x[e] for e in edges)

    # Degree constraints: each vertex incident to at most one chosen edge
    inc = {v: [] for v in G.nodes()}
    for (u, v) in G.edges():
        e = ek(u, v)
        inc[u].append(e)
        inc[v].append(e)
    for v in G.nodes():
        prob += pulp.lpSum(x[e] for e in inc[v]) <= 1, f"deg_{v}"

    # Solve via the uniform hook
    solve(prob)

    # Extract selected edges
    return {e for e in edges if pulp.value(x[e]) > 0.5}

@enforce_type(0, (nx.Graph, SimpleGraph))
def matching_number(
    G: GraphLike,
    **solver_kwargs,  # forwards (verbose, solver, solver_options) to maximum_matching
) -> int:
    r"""Return the size of a maximum matching in :math:`G`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph.

    Other Parameters
    ----------------
    verbose : bool, default=False
        Passed through to the solver.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver spec handled by :func:`graphcalc.solvers.resolve_solver`.
    solver_options : dict, optional
        Extra kwargs used when constructing the solver if needed.

    Returns
    -------
    int
        The matching number :math:`\nu(G)`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> gc.matching_number(complete_graph(4))
    2
    """
    return len(maximum_matching(G, **solver_kwargs))


@enforce_type(0, (nx.Graph, SimpleGraph))
def triameter(G: GraphLike) -> int:
    r"""
    Compute the triameter of a connected graph :math:`G`.

    The triameter is defined as:

    .. math::

        \text{max} \{ d(u,v) + d(v,w) + d(u,w) : u, v, w \in V \}

    where :math:`d(u,v)` is the shortest-path distance between :math:`u` and :math:`v`.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected, connected graph.

    Returns
    -------
    int
        The triameter of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(10)
    >>> gc.triameter(G)
    10
    """
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected to have a finite triameter.")

    # Precompute all-pairs shortest-path distances
    dist = dict(nx.all_pairs_shortest_path_length(G))

    tri = 0
    for u, v, w in itertools.combinations(G.nodes(), 3):
        s = dist[u][v] + dist[v][w] + dist[u][w]
        if s > tri:
            tri = s
    return tri
