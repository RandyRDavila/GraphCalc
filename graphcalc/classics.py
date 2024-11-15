import pulp
import itertools
import networkx as nx

__all__ = [
    "maximum_independent_set",
    "independence_number",
    "maximum_clique",
    "clique_number",
    "optimal_proper_coloring",
    "chromatic_number",
    "minimum_vertex_cover",
    "vertex_cover_number",
    "edge_cover_number",
    "maximum_matching",
    "matching_number",
]

def maximum_independent_set(G):
    r"""Return a largest independent set of nodes in *G*.

    This method uses integer programming to solve for a largest
    independent set. It solves the following integer program:


    .. math::
        \max \sum_{v \in V} x_v

    subject to

    .. math::
        \sum_{\{u, v\} \in E} x_u + x_v \leq 1 \text{ for all } e \in E

    where *E* and *V* are the set of edges and nodes of G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    set
        A set of nodes comprising a largest independent set in *G*.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.maximum_independent_set(G)
    {0}
    """

    # surpress output logs from pulp
    pulp.LpSolverDefault.msg = 0

    prob = pulp.LpProblem("MaximumIndependentSet", pulp.LpMaximize)
    variables = {node: pulp.LpVariable("x{}".format(i + 1), 0, 1, pulp.LpBinary) for i, node in enumerate(G.nodes())}

    # Set the domination number objective function
    prob += pulp.lpSum(variables)

    # Set constraints for independence
    for e in G.edges():
        prob += variables[e[0]] + variables[e[1]] <= 1

    prob.solve()
    solution_set = {node for node in variables if variables[node].value() == 1}
    return solution_set

def independence_number(G):
    r"""Return the size of a largest independent set in *G*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The size of a largest independent set in *G*.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.independence_number(G)
    1

    """
    return len(maximum_independent_set(G))

def maximum_clique(G):
    r"""Finds the maximum clique in a graph.

    This function computes the maximum clique of a graph `G` by finding the maximum independent set
    of the graph's complement.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        list: A list of nodes representing the maximum clique in the graph `G`.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.maximum_clique(G)
    {0, 1, 2, 3}
    """
    return maximum_independent_set(nx.complement(G))


def clique_number(G):
    r"""Calculates the clique number of a graph.

    The clique number of a graph `G` is the size of the largest clique. This function finds it by
    calculating the independence number of the complement of `G`.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        int: The clique number of the graph `G`.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.clique_number(G)
    4
    """
    return independence_number(nx.complement(G))


def optimal_proper_coloring(G):
    r"""Finds the optimal proper coloring of a graph using linear programming.

    This function uses integer linear programming to find the optimal (minimum) number of colors
    required to color the graph `G` such that no two adjacent nodes have the same color. Each node
    is assigned a color represented by a binary variable.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        dict: A dictionary where keys are color indices and values are lists of nodes in `G`
              assigned that color.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.optimal_proper_coloring(G)
    {0: [0], 1: [1], 2: [2], 3: [3]}
    """
    pulp.LpSolverDefault.msg = 0
    prob = pulp.LpProblem("OptimalProperColoring", pulp.LpMinimize)
    colors = {i: pulp.LpVariable("x_{}".format(i), 0, 1, pulp.LpBinary) for i in range(G.order())}
    node_colors = {
        node: [pulp.LpVariable("c_{}_{}".format(node, i), 0, 1, pulp.LpBinary) for i in range(G.order())] for node in G.nodes()
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

    prob.solve()
    solution_set = {color: [node for node in node_colors if node_colors[node][color].value() == 1] for color in colors}
    return solution_set


def chromatic_number(G):
    r"""Return the chromatic number of the graph G.

    The chromatic number of a graph is the smallest number of colors needed to color the vertices of G so that no two
    adjacent vertices share the same color.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The chromatic number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.chromatic_number(G)
    4

    """
    coloring = optimal_proper_coloring(G)
    colors = [color for color in coloring if len(coloring[color]) > 0]
    return len(colors)

def minimum_vertex_cover(G):
    r"""Return a smallest vertex cover of the graph G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    set
        A smallest vertex cover of G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.minimum_vertex_cover(G)
    {1, 2, 3}
    """
    X = maximum_independent_set(G)
    return G.nodes() - X

def vertex_cover_number(G):
    r"""Return a the size of smallest vertex cover in the graph G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    number
        The size of a smallest vertex cover of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.vertex_cover_number(G)
    3
    """
    return G.order() - independence_number(G)

def minimum_edge_cover(G):
    r"""Return a smallest edge cover of the graph G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    set
        A smallest edge cover of G.
    """
    return nx.min_edge_cover(G)

def edge_cover_number(G):
    r"""Return the size of a smallest edge cover in the graph G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    number
        The size of a smallest edge cover of G.
    """
    return len(nx.min_edge_cover(G))

def maximum_matching(G):
    r"""Return a maximum matching in the graph G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    set
        A maximum matching of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.maximum_matching(G)
    {(0, 1), (2, 3)}
    """
    pulp.LpSolverDefault.msg = 0
    prob = pulp.LpProblem("MaximumMatchingSet", pulp.LpMaximize)
    variables = {edge: pulp.LpVariable("x{}".format(i + 1), 0, 1, pulp.LpBinary) for i, edge in enumerate(G.edges())}

    # Set the maximum matching objective function
    prob += pulp.lpSum(variables)

    # Set constraints
    for node in G.nodes():
        incident_edges = [variables[edge] for edge in variables if node in edge]
        prob += sum(incident_edges) <= 1

    prob.solve()
    solution_set = {edge for edge in variables if variables[edge].value() == 1}
    return solution_set

def matching_number(G):
    r"""Return the size of a maximum matching in the graph G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    number
        The size of a maximum matching of G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.matching_number(G)
    2

    """
    pulp.LpSolverDefault.msg = 0
    prob = pulp.LpProblem("MaximumMatchingSet", pulp.LpMaximize)
    variables = {edge: pulp.LpVariable("x{}".format(i + 1), 0, 1, pulp.LpBinary) for i, edge in enumerate(G.edges())}

    # Set the maximum matching objective function
    prob += pulp.lpSum(variables)

    # Set constraints
    for node in G.nodes():
        incident_edges = [variables[edge] for edge in variables if node in edge]
        prob += sum(incident_edges) <= 1

    prob.solve()
    solution_set = {edge for edge in variables if variables[edge].value() == 1}
    return len(solution_set)
