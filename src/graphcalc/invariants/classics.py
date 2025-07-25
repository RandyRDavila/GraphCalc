from typing import Set, Hashable, Dict
import pulp
import itertools

from typing import Union, Set, Hashable
import networkx as nx
from graphcalc.core import SimpleGraph
from graphcalc.utils import get_default_solver, enforce_type, GraphLike


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
]

@enforce_type(0, (nx.Graph, SimpleGraph))
def maximum_independent_set(G: GraphLike) -> Set[Hashable]:
    r"""Return a largest independent set of nodes in *G*.

    This method uses integer programming to solve the following formulation:

    .. math::
        \max \sum_{v \in V} x_v

    subject to

    .. math::
        x_u + x_v \leq 1 \quad \text{for all } \{u, v\} \in E

    where *E* and *V* are the edge and vertex sets of *G*.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    set of hashable
        A set of nodes comprising a largest independent set in *G*.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.maximum_independent_set(G)
    {3}
    """
    # Initialize LP model
    prob = pulp.LpProblem("MaximumIndependentSet", pulp.LpMaximize)

    # Decision variables: x_v ∈ {0, 1} for each node
    variables = {
        v: pulp.LpVariable(f"x_{v}", cat="Binary")
        for v in G.nodes()
    }

    # Objective: maximize the number of selected nodes
    prob += pulp.lpSum(variables[v] for v in G.nodes())

    # Constraints: adjacent nodes cannot both be selected
    for u, v in G.edges():
        prob += variables[u] + variables[v] <= 1, f"edge_{u}_{v}"

    # Solve using default solver
    solver = get_default_solver()
    prob.solve(solver)

    # Raise value error if solution not found
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")

    # Extract solution
    return {v for v in G.nodes() if pulp.value(variables[v]) == 1}

@enforce_type(0, (nx.Graph, SimpleGraph))
def independence_number(G: GraphLike) -> int:
    r"""Return the size of a largest independent set in *G*.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    int
        The size of a largest independent set in *G*.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.independence_number(G)
    1

    """
    return len(maximum_independent_set(G))

@enforce_type(0, (nx.Graph, SimpleGraph))
def maximum_clique(G: GraphLike) -> Set[Hashable]:
    r"""Finds the maximum clique in a graph.

    This function computes the maximum clique of a graph `G` by finding the maximum independent set
    of the graph's complement.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    set
        A set of nodes representing the maximum clique in the graph `G`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.maximum_clique(G)
    {0, 1, 2, 3}
    """
    if hasattr(G, "complement"):
        return maximum_independent_set(G.complement())
    else:
        return maximum_independent_set(nx.complement(G))

@enforce_type(0, (nx.Graph, SimpleGraph))
def clique_number(G: GraphLike) -> int:
    r"""
    Compute the clique number of the graph.

    The clique number is the size of the largest clique in the graph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    int
        The clique number of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.clique_number(G)
    4
    """
    complement_graph = G.complement() if hasattr(G, "complement") else nx.complement(G)
    return independence_number(complement_graph)

@enforce_type(0, (nx.Graph, SimpleGraph))
def optimal_proper_coloring(G: GraphLike) -> Dict:
    r"""Finds the optimal proper coloring of a graph using linear programming.

    This function uses integer linear programming to find the optimal (minimum) number of colors
    required to color the graph `G` such that no two adjacent nodes have the same color. Each node
    is assigned a color represented by a binary variable.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.

    Returns
    -------
    dict:
        A dictionary where keys are color indices and values are lists of nodes in `G` assigned that color.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.optimal_proper_coloring(G)
    {0: [1], 1: [0], 2: [3], 3: [2]}
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
def chromatic_number(G):
    r"""Return the chromatic number of the graph G.

    The chromatic number of a graph is the smallest number of colors needed to color the vertices of G so that no two
    adjacent vertices share the same color.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    int
        The chromatic number of G.

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
def minimum_vertex_cover(G):
    r"""Return a smallest vertex cover of the graph G.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    set
        A smallest vertex cover of G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.minimum_vertex_cover(G)
    {0, 1, 2}
    """
    X = maximum_independent_set(G)
    return G.nodes() - X

@enforce_type(0, (nx.Graph, SimpleGraph))
def vertex_cover_number(G):
    r"""Return a the size of smallest vertex cover in the graph G.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    number
        The size of a smallest vertex cover of G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.vertex_cover_number(G)
    3
    """
    return G.order() - independence_number(G)

@enforce_type(0, (nx.Graph, SimpleGraph))
def minimum_edge_cover(G):
    r"""Return a smallest edge cover of the graph G.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    set
        A smallest edge cover of G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    >>> gc.minimum_edge_cover(G)
    {(2, 1), (3, 0)}
    """
    return nx.min_edge_cover(G)

@enforce_type(0, (nx.Graph, SimpleGraph))
def edge_cover_number(G):
    r"""Return the size of a smallest edge cover in the graph G.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    number
        The size of a smallest edge cover of G.

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
def maximum_matching(G):
    r"""Return a maximum matching in the graph G.

    A matching in a graph is a set of edges with no shared endpoint. This function uses
    integer programming to solve for a maximum matching in the graph G. It solves the following
    integer program:

    .. math::
        \max \sum_{e \in E} x_e \text{ where } x_e \in \{0, 1\} \text{ for all } e \in E

    subject to

    .. math::
        \sum_{e \in \delta(v)} x_e \leq 1 \text{ for all } v \in V

    where $\delta(v)$ is the set of edges incident to node v, and
    *E* and *V* are the set of edges and nodes of G, respectively.


    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    set
        A maximum matching of G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.maximum_matching(G)
    {(0, 1), (2, 3)}
    """
    prob = pulp.LpProblem("MaximumMatchingSet", pulp.LpMaximize)
    variables = {edge: pulp.LpVariable(f"x_{edge}", 0, 1, pulp.LpBinary) for edge in G.edges()}

    # Set the maximum matching objective function
    prob += pulp.lpSum(variables)

    # Set constraints
    for node in G.nodes():
        incident_edges = [variables[edge] for edge in variables if node in edge]
        prob += sum(incident_edges) <= 1

    solver = get_default_solver()
    prob.solve(solver)

    # Raise value error if solution not found
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")

    solution_set = {edge for edge in variables if variables[edge].value() == 1}
    return solution_set

@enforce_type(0, (nx.Graph, SimpleGraph))
def matching_number(G):
    r"""Return the size of a maximum matching in the graph G.

    Parameters
    ----------
    G : NetworkX Graph or GraphCalc SimpleGraph
        An undirected graph.

    Returns
    -------
    number
        The size of a maximum matching of G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.matching_number(G)
    2

    """
    return len(maximum_matching(G))
