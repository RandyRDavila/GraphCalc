from typing import Set, Hashable, Dict, Tuple, Hashable, List
import pulp
import itertools
import networkx as nx
import math
from dataclasses import dataclass

from ..core import SimpleGraph
from graphcalc.utils import (
    get_default_solver,
    enforce_type,
    GraphLike,
    _extract_and_report,
)
from graphcalc.solvers import with_solver
from graphcalc.metadata import invariant_metadata


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
    "vertex_clique_cover_partition",
    "vertex_clique_cover_number",
    "arboricity",
    "linear_arboricity",
    "maximum_induced_bipartite_subgraph",
    "bipartite_number",
    "average_distance",
    "path_cover_number",
    "is_hamiltonian",
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
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
@invariant_metadata(
    display_name="Independence number",
    notation=r"\alpha(G)",
    category="independence and covers",
    aliases=("stable set number", "graph independence number"),
    definition=(
        "The independence number alpha(G) is the maximum cardinality of an "
        "independent set of vertices in G, where an independent set is a set of "
        "pairwise nonadjacent vertices."
    ),
)
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph, cycle_graph
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
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
@invariant_metadata(
    display_name="Clique number",
    notation=r"\omega(G)",
    category="cliques",
    aliases=("graph clique number",),
    definition=(
        "The clique number omega(G) is the maximum cardinality of a clique in G, "
        "where a clique is a set of vertices that are pairwise adjacent."
    ),
)
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph, cycle_graph
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph

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

    for (u, v), i in itertools.product(G.edges(), range(G.order())):
        prob += node_colors[u][i] + node_colors[v][i] <= 1

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
@invariant_metadata(
    display_name="Chromatic number",
    notation=r"\chi(G)",
    category="colorings",
    aliases=("vertex chromatic number", "graph chromatic number"),
    definition=(
        "The chromatic number chi(G) is the minimum number of colors needed to "
        "color the vertices of G so that adjacent vertices receive distinct colors."
    ),
)
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.chromatic_number(G)
    4
    """
    coloring = optimal_proper_coloring(G)
    colors = [color for color in coloring if len(coloring[color]) > 0]
    return len(colors)

@enforce_type(0, (nx.Graph, SimpleGraph))
def vertex_clique_cover_partition(G: GraphLike) -> Dict[int, List[Hashable]]:
    r"""
    Partition \(V(G)\) into the fewest number of cliques (a vertex clique cover),
    returning the actual parts.

    This uses the identity \(\theta(G) = \chi(\overline{G})\): we compute an
    optimal proper coloring of the complement \(\overline{G}\), then interpret
    each color class as a clique in \(G\).

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    dict[int, list[hashable]]
        A dictionary mapping color indices to vertex lists. Only nonempty parts
        are returned. Each part induces a clique in \(G\).

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(5)  # C5
    >>> parts = gc.vertex_clique_cover_partition(G)
    >>> sum(len(vs) for vs in parts.values()) == G.order()
    True
    """
    G_comp = nx.complement(G)
    coloring_comp = optimal_proper_coloring(G_comp)
    # Keep only nonempty color classes
    partition = {k: vs for k, vs in coloring_comp.items() if len(vs) > 0}
    return partition


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Vertex clique cover number",
    notation=r"\theta(G)",
    category="cliques",
    aliases=("clique cover number", "vertex clique partition number"),
    definition=(
        "The vertex clique cover number theta(G) is the minimum number of cliques "
        "whose vertex sets partition the vertex set of G."
    ),
)
def vertex_clique_cover_number(G: GraphLike) -> int:
    r"""
    Vertex clique cover number \(\theta(G)\): the fewest cliques needed to partition \(V(G)\).

    Uses \(\theta(G) = \chi(\overline{G})\), i.e., the chromatic number of the complement.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    int
        The vertex clique cover number \(\theta(G)\).

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph, cycle_graph
    >>> gc.vertex_clique_cover_number(complete_graph(4))
    1
    >>> gc.vertex_clique_cover_number(cycle_graph(5))  # C5, complement is C5, χ=3
    3
    """
    # If you prefer to reuse your chromatic_number API:
    # return chromatic_number(nx.complement(G))
    parts = vertex_clique_cover_partition(G)
    return len(parts)

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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
    >>> G = complete_graph(4)
    >>> len(gc.minimum_vertex_cover(G))  # any 3 vertices form a minimum cover
    3
    """
    S = set(maximum_independent_set(G, **solver_kwargs))
    return set(G.nodes()) - S

@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Vertex cover number",
    notation=r"\tau(G)",
    category="independence and covers",
    aliases=("minimum vertex cover size",),
    definition=(
        "The vertex cover number tau(G) is the minimum cardinality of a vertex "
        "cover of G, where a vertex cover is a set of vertices incident to every edge."
    ),
)
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
    >>> G = complete_graph(4)
    >>> solution = gc.minimum_edge_cover(G)
    """
    return nx.min_edge_cover(G)

@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Edge cover number",
    notation=r"\rho(G)",
    category="matchings and covers",
    aliases=("minimum edge cover size",),
    definition=(
        "The edge cover number rho(G) is the minimum cardinality of an edge cover "
        "of G, where an edge cover is a set of edges incident to every vertex of G."
    ),
)
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
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
@invariant_metadata(
    display_name="Matching number",
    notation=r"\nu(G)",
    category="matchings",
    aliases=("maximum matching size",),
    definition=(
        "The matching number nu(G) is the maximum cardinality of a matching in G, "
        "where a matching is a set of pairwise nonincident edges."
    ),
)
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import complete_graph
    >>> gc.matching_number(complete_graph(4))
    2
    """
    return len(maximum_matching(G, **solver_kwargs))


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Triameter",
    notation=r"\operatorname{tr}(G)",
    category="distances",
    aliases=("graph triameter",),
    definition=(
        "The triameter tr(G) of a connected graph G is the maximum value of "
        "d(u,v) + d(v,w) + d(u,w) taken over all triples of vertices u, v, w in G, "
        "where d denotes shortest-path distance."
    ),
)
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
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph

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

# def bipartite_number(G):
#     r"""
#     Compute the **bipartite number** of a graph :math:`G`.

#     The bipartite number :math:`b(G)` is the order (number of vertices) of a largest
#     **induced bipartite subgraph** of :math:`G`. Equivalently, it is the maximum size of a
#     vertex subset whose induced subgraph is bipartite:

#     .. math::

#         b(G) \;=\; \max\{\, |S| : S \subseteq V(G)\ \text{and}\ G[S]\ \text{is bipartite}\,\},

#     where :math:`G[S]` denotes the subgraph of :math:`G` induced by :math:`S`.

#     This invariant is complementary to the **odd cycle transversal number** (also called the
#     **vertex bipartization number**), which is the minimum number of vertices that must be
#     deleted to make the graph bipartite. Writing :math:`\tau_{\mathrm{odd}}(G)` for that minimum,

#     .. math::

#         b(G) \;=\; |V(G)| - \tau_{\mathrm{odd}}(G).

#     Parameters
#     ----------
#     G : networkx.Graph-like
#         A finite undirected graph. The bipartiteness test is performed with
#         :func:`networkx.algorithms.bipartite.basic.is_bipartite` on induced subgraphs.

#         For best-defined behavior, use a simple graph (no self-loops, no parallel edges).
#         A self-loop makes a graph non-bipartite; parallel edges do not affect bipartiteness.

#     Returns
#     -------
#     int
#         The bipartite number :math:`b(G)`. If :math:`G` is empty (has no vertices), returns 0.

#     Notes
#     -----
#     - Since bipartite graphs are exactly those containing **no odd cycle**, this function
#       searches for the largest induced subgraph that is free of odd cycles.
#     - The empty graph is bipartite; hence the optimization problem is always feasible, and
#       :math:`b(G)` is well-defined for all finite graphs.
#     - The implementation below is **exact** but **brute force**: it checks induced subgraphs
#       in decreasing order of size and returns the first bipartite one found. This is intended
#       only for small graphs.

#     Complexity
#     ----------
#     Exponential in :math:`n = |V(G)|`. In the worst case it may examine :math:`\Theta(2^n)`
#     vertex subsets, and each check runs a bipartiteness test on the induced subgraph.
#     Practical only for small :math:`n` (often :math:`n \lesssim 20`, depending on density).

#     Examples
#     --------
#     >>> import graphcalc.graphs as gc
#     >>> from graphcalc.graphs.generators import cycle_graph
#     >>> G = cycle_graph(5)  # C5 is not bipartite; deleting any 1 vertex gives P4
#     >>> gc.bipartite_number(G)
#     4

#     >>> from graphcalc.graphs.generators import complete_graph
#     >>> H = complete_graph(4)  # K4: largest induced bipartite subgraph is K2,2 on 4 vertices? no, K4 has triangles
#     >>> gc.bipartite_number(H)
#     2
#     """
#     n = G.number_of_nodes()
#     if n == 0:
#         return 0

#     nodes = list(G.nodes())

#     # Search from large to small; stop at first feasible size.
#     for k in range(n, 0, -1):
#         for S in itertools.combinations(nodes, k):
#             if nx.is_bipartite(G.subgraph(S)):
#                 return k

#     # k = 0 always works (empty graph is bipartite), but keep conventionally:
#     return 0

@invariant_metadata(
    display_name="Average distance",
    notation=r"\operatorname{avgdist}(G)",
    category="distances",
    aliases=("mean distance", "average shortest-path distance"),
    definition=(
        "The average distance avgdist(G) is the mean of the finite shortest-path "
        "distances between unordered pairs of distinct vertices of G."
    ),
)
def average_distance(G):
    r"""
    Compute the **average finite shortest-path distance** of an undirected graph.

    This function returns the mean of the unweighted shortest-path distance
    :math:`d_G(u,v)` over all **unordered vertex pairs** :math:`\{u,v\}` for which
    the distance is finite (i.e., :math:`u` and :math:`v` lie in the same connected
    component). Formally, let

    .. math::

        P \;=\; \bigl\{ \{u,v\} \subseteq V(G) : u \neq v,\ d_G(u,v) < \infty \bigr\},

    then the returned value is

    .. math::

        \operatorname{avgdist}(G) \;=\;
        \begin{cases}
        \dfrac{1}{|P|}\sum_{\{u,v\}\in P} d_G(u,v), & |P|>0,\\[6pt]
        0, & |P|=0.
        \end{cases}

    Conventions
    -----------
    - If :math:`|V(G)| < 2`, the function returns ``0.0``.
    - If :math:`G` is disconnected, only pairs of vertices within the same connected
      component are included; pairs in different components are **ignored** (they are
      not treated as having infinite distance).
    - With these conventions, the only way :math:`|P|=0` is when :math:`|V(G)|<2`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite undirected graph. Distances are computed as **unweighted**
        shortest-path lengths (BFS distances). If you require weighted distances,
        use Dijkstra-based routines (e.g. ``nx.single_source_dijkstra_path_length``)
        and adjust the definition accordingly.

    Returns
    -------
    float
        The average finite shortest-path distance over all connected unordered vertex pairs.

    Notes
    -----
    - For a connected graph, this equals the standard **average shortest-path length**
      (also called the mean distance).
    - Some authors define the average distance only for connected graphs (and otherwise
      return :math:`\infty` or raise an exception). This implementation instead computes
      an *intra-component* mean, which stays finite on disconnected graphs.

    Complexity
    ----------
    Let the connected components of :math:`G` be :math:`C_1,\dots,C_t`. The implementation
    runs a BFS from each vertex within each component and counts each unordered pair once.
    The time complexity is

    .. math::

        O\!\left(\sum_{i=1}^t |V(C_i)|\bigl(|V(C_i)| + |E(C_i)|\bigr)\right),

    and the additional memory used by each BFS is :math:`O(|V(C_i)|)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> # Path on 4 vertices: distances are 1,2,3,1,2,1 (sum 10 over 6 pairs)
    >>> G = nx.path_graph(4)
    >>> gc.average_distance(G)
    1.6666666666666667

    >>> # Disconnected: average over pairs within components only
    >>> H = nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))
    >>> gc.average_distance(H)
    1.25
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0.0

    total = 0
    count = 0

    for comp in nx.connected_components(G):
        nodes = list(comp)
        k = len(nodes)
        if k < 2:
            continue

        # Count each unordered pair {u,v} exactly once by only adding distances to "later" vertices.
        index = {v: i for i, v in enumerate(nodes)}
        for u in nodes:
            dist_u = nx.single_source_shortest_path_length(G, u)
            iu = index[u]
            for v, d in dist_u.items():
                iv = index.get(v)
                if iv is not None and iv > iu:
                    total += d
                    count += 1

    return (total / count) if count > 0 else 0.0

def _all_simple_paths_vertex_sets(G):
    r"""
    Enumerate candidate path vertex-sets for an exact **vertex-disjoint path cover** search.

    This helper constructs a collection of subsets :math:`P \subseteq V(G)` such that the vertices
    in :math:`P` can be ordered to form a **simple undirected path** in :math:`G`. Each candidate
    path is represented only by its **vertex set** (a ``frozenset``), not by an ordered sequence.

    In particular, all singleton sets :math:`\{v\}` are included, corresponding to paths of length 0.

    Parameters
    ----------
    G : networkx.Graph-like
        Intended for finite undirected graphs (typically simple graphs). Paths are interpreted in
        the usual undirected sense and enumerated using :func:`networkx.all_simple_paths`.

    Returns
    -------
    list[frozenset]
        A list of distinct vertex sets. Each returned set :math:`P` has the property that
        :math:`G` contains at least one simple path whose vertex set is exactly :math:`P`.

    Notes
    -----
    - Many different simple paths can share the same vertex set (e.g. the path and its reverse,
      or multiple embeddings in graphs with symmetries). Collapsing them to sets is **sound** for
      the downstream solver used in :func:`path_cover_number`, because that solver only needs:
        (i) which vertices are covered, and (ii) whether candidate paths are vertex-disjoint.
      The internal traversal order of a path does not matter for these constraints.
    - This generator is intentionally brute-force and can become enormous quickly, since the number
      of simple paths in a graph can be exponential in :math:`|V|`.

    Complexity
    ----------
    Potentially exponential (and in dense graphs, extremely large) in :math:`n=|V(G)|`, because it
    enumerates all simple paths between all unordered pairs of vertices (up to cutoff :math:`n-1`).
    This helper is intended only for very small graphs.

    See Also
    --------
    path_cover_number : Uses these candidates in an exact backtracking search for a minimum path cover.
    """
    nodes = list(G.nodes())
    path_sets = {frozenset([v]) for v in nodes}

    # Enumerate simple paths between unordered pairs to avoid duplication.
    for s, t in itertools.combinations(nodes, 2):
        for path in nx.all_simple_paths(G, s, t, cutoff=len(nodes) - 1):
            path_sets.add(frozenset(path))

    return list(path_sets)


@invariant_metadata(
    display_name="Path cover number",
    notation=r"\operatorname{pc}(G)",
    category="decompositions",
    aliases=("path partition number",),
    definition=(
        "The path cover number pc(G) of an undirected graph G is the minimum number "
        "of pairwise vertex-disjoint simple paths whose vertex sets partition V(G)."
    ),
)
def path_cover_number(G, max_n=20):
    r"""
    Compute the **path cover number** of an undirected graph: the minimum size of a
    vertex-disjoint path cover.

    This function solves the following optimization problem. Find the smallest integer :math:`k`
    for which there exist **pairwise vertex-disjoint** simple paths
    :math:`P_1,\dots,P_k` in :math:`G` such that their vertex sets partition the vertex set:

    .. math::

        V(G) \;=\; V(P_1)\,\dot\cup\,V(P_2)\,\dot\cup\,\cdots\,\dot\cup\,V(P_k),

    where each :math:`P_i` is a simple undirected path (and paths of length 0, i.e. single vertices,
    are allowed).

    In other words, this is the minimum number of disjoint paths whose union covers all vertices.
    This quantity is also called a **path partition number** in some sources.

    Conventions
    -----------
    - Paths are **simple** undirected paths.
    - **Singleton paths** :math:`\{v\}` are allowed (paths of length 0).
    - The chosen paths must be **vertex-disjoint**, so the cover is a partition of :math:`V(G)`.
    - If :math:`G` has no vertices, the value is 0.

    Parameters
    ----------
    G : networkx.Graph-like
        Intended for finite undirected graphs (typically simple graphs).
    max_n : int, optional
        Safety cutoff on :math:`|V(G)|`. This implementation is exact but brute-force and can
        become infeasible quickly. If :math:`|V(G)| > \texttt{max_n}`, a ``ValueError`` is raised.

    Returns
    -------
    int
        The minimum number of vertex-disjoint paths whose union is :math:`V(G)`.

    Raises
    ------
    ValueError
        If :math:`|V(G)| > \texttt{max_n}`.

    Notes
    -----
    - This is **not** the standard “minimum path cover” problem for DAGs (which is polynomial-time via maximum matching). Here the input is an **undirected** graph and the paths must be vertex-disjoint and cover all vertices; this variant is NP-hard in general.
    - Implementation strategy:
        1. Enumerate candidate paths by their vertex sets using :func:`_all_simple_paths_vertex_sets`.
        2. Backtrack to choose a minimum number of these sets that form a partition of :math:`V(G)`.
    - The backtracking branches on an uncovered vertex :math:`v` and tries all candidate path-sets containing :math:`v` that fit inside the remaining uncovered vertices.

    Complexity
    ----------
    Exponential in :math:`n = |V(G)|`. The helper that enumerates simple paths can itself be
    exponential, and the subsequent set-packing/partition backtracking is also exponential.
    Intended only for small graphs (your cutoff parameter controls this).

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> # A path is coverable by a single path
    >>> G = nx.path_graph(6)
    >>> gc.path_cover_number(G)
    1

    >>> # An edgeless graph on n vertices needs n singleton paths
    >>> H = nx.empty_graph(5)
    >>> gc.path_cover_number(H)
    5

    >>> # Two disjoint paths need two paths in the cover
    >>> J = nx.disjoint_union(nx.path_graph(3), nx.path_graph(4))
    >>> gc.path_cover_number(J)
    2
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0
    if n > max_n:
        raise ValueError(f"path_cover_number brute force intended for n <= {max_n}, got n={n}")

    U = set(G.nodes())
    path_sets = _all_simple_paths_vertex_sets(G)

    # Group candidate path-sets by a vertex they contain (branching heuristic).
    by_vertex = {v: [] for v in U}
    for P in path_sets:
        for v in P:
            by_vertex[v].append(P)

    best = n  # trivial cover by singletons

    def backtrack(remaining, used_count):
        nonlocal best
        if used_count >= best:
            return
        if not remaining:
            best = used_count
            return

        v = next(iter(remaining))

        for P in by_vertex[v]:
            if P.issubset(remaining):
                backtrack(remaining - set(P), used_count + 1)

    backtrack(U, 0)
    return best

@invariant_metadata(
    display_name="Arboricity",
    notation=r"ab(G)",
    category="decompositions",
    aliases=("graph arboricity", "Nash-Williams arboricity"),
    definition=(
        "The arboricity ab(G) of a graph G is the minimum number of forests whose "
        "edge sets partition the edge set of G."
    ),
)
def arboricity(G: nx.Graph) -> int:
    r"""
    Compute the (undirected) arboricity :math:`a(G)` exactly.

    Arboricity measures how many forests are needed to cover the edges of a graph.
    Formally, :math:`a(G)` is the minimum integer :math:`k` such that :math:`E(G)` can be
    partitioned into :math:`k` forests.

    Nash–Williams / Tutte characterization
    --------------------------------------
    A classical theorem gives the exact formula

    .. math::
        a(G) \;=\; \max_{H \subseteq G,\; |V(H)| \ge 2}\;
        \left\lceil \frac{|E(H)|}{|V(H)| - 1} \right\rceil.

    Equivalently, :math:`a(G)` is the smallest :math:`k` such that for every vertex subset
    :math:`S \subseteq V(G)` with :math:`|S| \ge 2`,

    .. math::
        |E(G[S])| \;\le\; k\,(|S| - 1).

    This function computes arboricity exactly by testing candidate values of :math:`k` via a
    min-cut reduction that detects whether there exists a violating subset :math:`S` with

    .. math::
        |E(G[S])| \;>\; k\,(|S| - 1).

    Min-cut oracle
    --------------
    For a fixed :math:`k`, consider the objective

    .. math::
        \max_{S \subseteq V(G)} \bigl(|E(G[S])| - k|S|\bigr).

    Since
    :math:`|E(G[S])| - k(|S|-1) = (|E(G[S])| - k|S|) + k`,
    a violation exists if and only if

    .. math::
        \max_{S} \bigl(|E(G[S])| - k|S|\bigr) \;>\; -k.

    This maximum can be obtained via an :math:`s`–:math:`t` min-cut construction:

    - Create a node for each original vertex (V-nodes).
    - Create a node for each original edge (E-nodes).
    - Add an arc ``source -> E-node`` with capacity 1.
    - Add arcs ``E-node ->`` its two endpoint V-nodes with capacity ``INF``.
    - Add an arc ``V-node -> sink`` with capacity :math:`k`.

    If the :math:`s`-side contains a vertex subset :math:`S`, then the cut value is

    .. math::
        \text{cut} \;=\; m - |E(G[S])| + k|S|,

    so minimizing the cut is equivalent to maximizing :math:`|E(G[S])| - k|S|`.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph. Self-loops are ignored. For a MultiGraph, parallel edges
        increase :math:`|E(G[S])|`; if you want multigraph arboricity, be explicit about
        the convention or pass a simple projection.

    Returns
    -------
    int
        The exact arboricity :math:`a(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> gc.arboricity(nx.path_graph(10))
    1
    >>> gc.arboricity(nx.complete_graph(6))
    3
    >>> gc.arboricity(nx.complete_bipartite_graph(3, 4))
    2
    """
    if G.is_directed():
        raise nx.NetworkXError("arboricity() here is for undirected graphs only.")

    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    n = H.number_of_nodes()
    m = H.number_of_edges()
    if m == 0:
        return 0
    if n <= 1:
        return 0

    # Quick lower/upper bounds
    delta = max((d for _, d in H.degree()), default=0)
    lo = 1
    hi = max(1, delta)  # arboricity <= Δ for simple graphs

    # Monotone property: if k works, larger k works.
    def violates(k: int) -> bool:
        # Returns True iff there exists S with |S| >= 2 and |E(S)| > k(|S|-1).
        # Equivalently, iff max_{|S|>=2} (|E(S)| - k|S|) > -k.
        nodes = list(H.nodes())
        if len(nodes) < 2:
            return False

        # Build once-per-pair (n<=30 so fine).
        for a_i in range(len(nodes)):
            a = nodes[a_i]
            for b_i in range(a_i + 1, len(nodes)):
                b = nodes[b_i]

                DG = nx.DiGraph()
                s, t = "_s", "_t"
                DG.add_node(s)
                DG.add_node(t)

                # Vertex nodes -> sink
                for v in H.nodes():
                    DG.add_edge(v, t, capacity=float(k))

                INF = float(m + 1)  # big enough
                # Force a,b into the s-side
                DG.add_edge(s, a, capacity=INF)
                DG.add_edge(s, b, capacity=INF)

                # Edge nodes
                for ei, (u, v) in enumerate(H.edges()):
                    en = ("e", ei)
                    DG.add_edge(s, en, capacity=1.0)
                    DG.add_edge(en, u, capacity=INF)
                    DG.add_edge(en, v, capacity=INF)

                cut_value, _ = nx.minimum_cut(DG, s, t, capacity="capacity")
                max_val = m - cut_value  # = max_{S ⊇ {a,b}} (|E(S)| - k|S|)

                if (max_val + k) > 1e-9:
                    return True

        return False


    # Binary search smallest k with no violation
    while lo < hi:
        mid = (lo + hi) // 2
        if violates(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


@dataclass
class _DSURollback:
    parent: List[int]
    size: List[int]
    history: List[Tuple[int, int, int]]  # (b, parent_b_old, size_a_old)

    @classmethod
    def create(cls, n: int) -> "_DSURollback":
        return cls(parent=list(range(n)), size=[1] * n, history=[])

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        # attach rb -> ra
        self.history.append((rb, self.parent[rb], self.size[ra]))
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def snapshot(self) -> int:
        return len(self.history)

    def rollback(self, snap: int) -> None:
        while len(self.history) > snap:
            rb, parent_rb_old, size_ra_old = self.history.pop()
            ra = self.parent[rb]
            self.parent[rb] = parent_rb_old
            self.size[ra] = size_ra_old

@invariant_metadata(
    display_name="Linear arboricity",
    notation=r"\operatorname{la}(G)",
    category="decompositions",
    aliases=("graph linear arboricity",),
    definition=(
        "The linear arboricity la(G) of a graph G is the minimum number of linear "
        "forests whose edge sets partition the edge set of G."
    ),
)
def linear_arboricity(G: nx.Graph) -> int:
    r"""
    Compute the **linear arboricity** :math:`\mathrm{la}(G)` exactly (intended for small simple graphs).

    A *linear forest* is a forest whose connected components are paths (including isolated vertices).
    Equivalently, a graph is a linear forest if and only if it is acyclic and has maximum degree at
    most 2.

    The *linear arboricity* :math:`\mathrm{la}(G)` is the minimum integer :math:`k` such that the edge
    set can be partitioned into :math:`k` linear forests.

    .. math::

        \mathrm{la}(G)
        \;=\;
        \min\{\, k : E(G)=E(L_1)\,\dot\cup\,\cdots\,\dot\cup\,E(L_k)\ \text{with each } L_i
        \text{ a linear forest}\,\}.

    Exactness and search strategy
    -----------------------------
    Computing linear arboricity is NP-hard in general. This implementation is intended for small
    graphs (typically :math:`n \le 20\text{--}30`) and performs an exact incremental feasibility search:

    - Lower bound: :math:`\lceil \Delta(G)/2 \rceil`, since in any linear forest each vertex has degree
      at most 2, so across :math:`k` forests a vertex can support at most :math:`2k` incident edges.
    - Upper bound: :math:`|E(G)|`, since assigning each edge its own color class yields a linear forest.

    For each :math:`k` from the lower bound upward, the algorithm attempts to assign each edge one of
    :math:`k` colors so that each color class induces a linear forest. The first feasible :math:`k` is
    returned.

    Feasibility checking (fixed k)
    ------------------------------
    Backtracking assigns edges to colors subject to:

    - Per-color degree constraint: for every color class and vertex, degree is at most 2.
    - Per-color acyclicity: adding an edge may not create a cycle in that color class.
      This is enforced via a rollback disjoint-set union (DSU) structure per color.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph. This routine is intended for **simple** graphs.
        Self-loops are ignored.

        Note: if a MultiGraph is provided, it is first projected to a simple graph via ``nx.Graph(G)``,
        which collapses parallel edges; thus multiplicity is not preserved under this implementation.

    Returns
    -------
    int
        The exact linear arboricity :math:`\mathrm{la}(G)` (under the above conventions).
        Returns 0 if :math:`|E(G)|=0`.

    Complexity
    ----------
    Exponential in :math:`|E(G)|` in the worst case due to backtracking; practical only for small graphs.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> gc.linear_arboricity(nx.path_graph(10))
    1
    >>> gc.linear_arboricity(nx.cycle_graph(10))
    2
    """
    if G.is_directed():
        raise nx.NetworkXError("linear_arboricity() here is for undirected graphs only.")

    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    n = H.number_of_nodes()
    m = H.number_of_edges()
    if m == 0:
        return 0

    nodes = list(H.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in H.edges()]
    # Heuristic: assign edges in an order that tends to prune earlier (high-degree endpoints first)
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    edges.sort(key=lambda e: (deg[e[0]] + deg[e[1]]), reverse=True)

    Delta = max(deg)
    lb = math.ceil(Delta / 2)
    ub = max(1, Delta)

    def feasible(k: int) -> bool:
        # per color DSU + per color degrees
        dsus = [_DSURollback.create(n) for _ in range(k)]
        deg_c = [[0] * n for _ in range(k)]

        # Quick necessary condition: total “degree capacity” per vertex across colors is 2k.
        # If some vertex has degree > 2k, impossible.
        if any(d > 2 * k for d in deg):
            return False

        # Backtracking
        def bt(i: int) -> bool:
            if i == len(edges):
                return True
            a, b = edges[i]

            # Try colors in an order that is likely to work: those with more remaining capacity on a,b
            candidates = list(range(k))
            candidates.sort(
                key=lambda c: (2 - deg_c[c][a]) + (2 - deg_c[c][b]),
                reverse=True,
            )

            for c in candidates:
                if deg_c[c][a] >= 2 or deg_c[c][b] >= 2:
                    continue
                # cycle check: adding edge (a,b) creates cycle iff find(a)==find(b) in that color
                dsu = dsus[c]
                if dsu.find(a) == dsu.find(b):
                    continue

                snap = dsu.snapshot()
                # apply
                dsu.union(a, b)
                deg_c[c][a] += 1
                deg_c[c][b] += 1

                if bt(i + 1):
                    return True

                # undo
                deg_c[c][a] -= 1
                deg_c[c][b] -= 1
                dsu.rollback(snap)

            return False

        return bt(0)

    for k in range(lb, ub + 1):
        if feasible(k):
            return k

    # Theoretically should never happen with ub=Δ, but keep a safe fallback:
    return ub


@enforce_type(0, (nx.Graph, SimpleGraph))
@with_solver
def maximum_induced_bipartite_subgraph(
    G: GraphLike,
    *,
    verbose: bool = False,
    solve=None,  # injected by @with_solver
) -> Set[Hashable]:
    r"""
    Return a maximum-cardinality vertex set inducing a bipartite subgraph of ``G``.

    The **bipartite number** of a graph :math:`G=(V,E)` is the maximum order of an
    induced bipartite subgraph. Equivalently, it is the largest size of a subset
    :math:`S \subseteq V` such that :math:`G[S]` is bipartite.

    This function computes such a subset exactly using a mixed integer linear program.

    Formulation
    -----------
    For each vertex :math:`v \in V`, introduce binary variables
    :math:`a_v, b_v \in \{0,1\}` where:

    - :math:`a_v = 1` means ``v`` is selected and placed in part A,
    - :math:`b_v = 1` means ``v`` is selected and placed in part B.

    We maximize the number of selected vertices:

    .. math::
        \max \sum_{v \in V} (a_v + b_v)

    subject to:

    .. math::
        a_v + b_v \le 1 \qquad \text{for all } v \in V,

    and for each edge :math:`uv \in E`,

    .. math::
        a_u + a_v \le 1,
        \qquad
        b_u + b_v \le 1.

    These constraints ensure that the selected vertices induce a graph whose
    vertex set can be partitioned into two independent sets, hence an induced
    bipartite subgraph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    verbose : bool, default=False
        If True, print solver output when supported.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra keyword arguments when constructing the solver if needed.

    Returns
    -------
    set of hashable
        A maximum-cardinality subset of vertices inducing a bipartite subgraph.

    Raises
    ------
    ValueError
        If no optimal solution is found by the solver.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(5)
    >>> S = gc.maximum_induced_bipartite_subgraph(G)
    >>> len(S)
    4

    >>> from graphcalc.graphs.generators import complete_graph
    >>> H = complete_graph(4)
    >>> len(gc.maximum_induced_bipartite_subgraph(H))
    2
    """
    prob = pulp.LpProblem("MaximumInducedBipartiteSubgraph", pulp.LpMaximize)

    a = {v: pulp.LpVariable(f"a_{v}", cat="Binary") for v in G.nodes()}
    b = {v: pulp.LpVariable(f"b_{v}", cat="Binary") for v in G.nodes()}

    prob += pulp.lpSum(a[v] + b[v] for v in G.nodes())

    for v in G.nodes():
        prob += a[v] + b[v] <= 1, f"select_side_{v}"

    for u, v in G.edges():
        prob += a[u] + a[v] <= 1, f"same_A_{u}_{v}"
        prob += b[u] + b[v] <= 1, f"same_B_{u}_{v}"

    solve(prob)

    if pulp.LpStatus[prob.status] != "Optimal":
        raise ValueError(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")

    S = {v for v in G.nodes() if pulp.value(a[v]) + pulp.value(b[v]) > 0.5}

    if verbose:
        print(f"Maximum induced bipartite subgraph size: {len(S)}")
        print(f"Selected vertices: {S}")

    return S


@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Bipartite number",
    notation=r"b(G)",
    category="bipartite structure",
    aliases=("maximum induced bipartite subgraph order",),
    definition=(
        "The bipartite number b(G) is the maximum cardinality of a vertex subset "
        "S of G such that the induced subgraph G[S] is bipartite."
    ),
)
def bipartite_number(
    G: GraphLike,
    **solver_kwargs,
) -> int:
    r"""
    Compute the **bipartite number** of a graph :math:`G`.

    The bipartite number :math:`b(G)` is the order of a largest induced
    bipartite subgraph of :math:`G`. Equivalently,

    .. math::
        b(G) = \max\{\, |S| : S \subseteq V(G)\ \text{and}\ G[S]\ \text{is bipartite}\,\}.

    Since bipartite graphs are exactly the graphs whose vertices can be
    partitioned into two independent sets, this implementation formulates the
    problem as a mixed integer linear program and solves it exactly.

    Notes
    -----
    This function calls :func:`maximum_induced_bipartite_subgraph` and returns
    the size of the selected vertex set.

    The bipartite number is complementary to the odd cycle transversal number
    :math:`\tau_{\mathrm{odd}}(G)`:

    .. math::
        b(G) = |V(G)| - \tau_{\mathrm{odd}}(G).

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    solver : str or dict or pulp.LpSolver or type or callable or None, optional
        Flexible solver specification handled by
        :func:`graphcalc.solvers.with_solver`.
    solver_options : dict, optional
        Extra keyword arguments when constructing the solver if needed.
    verbose : bool, default=False
        Passed through to the solver wrapper.

    Returns
    -------
    int
        The bipartite number :math:`b(G)`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph, complete_graph
    >>> gc.bipartite_number(cycle_graph(5))
    4
    >>> gc.bipartite_number(complete_graph(4))
    2
    """
    return len(maximum_induced_bipartite_subgraph(G, **solver_kwargs))

@enforce_type(0, (nx.Graph, SimpleGraph))
@invariant_metadata(
    display_name="Hamiltonian",
    notation=r"Hamiltonian",
    category="boolean property",
    aliases=("Is Hamiltonian",),
    definition=(
        "True if the graph contains a Hamiltonian cycle."
    ),
)
def is_hamiltonian(G: GraphLike) -> bool:
    r"""
    Return True iff the undirected simple graph G has a Hamiltonian cycle.

    This is an exact exponential-time backtracking algorithm, so it is only
    practical for small to medium graphs.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    bool
        True if G is Hamiltonian, else False.
    """
    # Basic validation
    if not isinstance(G, nx.Graph) or G.is_directed():
        raise TypeError("G must be a NetworkX undirected graph.")

    n = G.number_of_nodes()

    # By the usual convention, graphs with fewer than 3 vertices
    # cannot contain a Hamiltonian cycle.
    if n < 3:
        return False

    # A Hamiltonian graph must be connected.
    if not nx.is_connected(G):
        return False

    # Every vertex in a Hamiltonian graph has degree at least 2.
    if any(G.degree(v) < 2 for v in G.nodes):
        return False

    nodes = list(G.nodes)
    start = nodes[0]
    visited = {start}
    path = [start]

    def backtrack(current):
        # If all vertices are used, check whether we can close the cycle.
        if len(path) == n:
            return G.has_edge(current, start)

        # Try unvisited neighbors
        for nbr in G.neighbors(current):
            if nbr not in visited:
                visited.add(nbr)
                path.append(nbr)

                if backtrack(nbr):
                    return True

                path.pop()
                visited.remove(nbr)

        return False

    return backtrack(start)
