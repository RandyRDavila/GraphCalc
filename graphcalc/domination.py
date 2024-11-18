import networkx as nx
from itertools import combinations
import pulp
from pulp import (
    value,
    PULP_CBC_CMD,
)

from .neighborhoods import neighborhood, closed_neighborhood

__all__ = [
    "is_dominating_set",
    "minimum_dominating_set",
    "domination_number",
    "minimum_total_domination_set",
    "total_domination_number",
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
    "two_rainbow_domination_number",
    "three_rainbow_domination_number",
    "min_maximal_matching_number",
]

def is_dominating_set(G, S):
    r"""
    Checks if a given set of nodes, S, is a dominating set in the graph G.

    A dominating set of a graph G = (V, E) is a subset of nodes S ⊆ V such that every node in V is either in S or
    adjacent to a node in S. In other words, every node in the graph is either part of the dominating set or is
    "dominated" by a node in the dominating set.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    S : set
        A subset of nodes in the graph to check for domination.

    Returns
    -------
    bool
        True if S is a dominating set, otherwise False.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> S = {0, 2}
    >>> print(gc.is_dominating_set(G, S))
    True

    >>> S = {0}
    >>> print(gc.is_dominating_set(G, S))
    False
    """
    return all(any(u in S for u in closed_neighborhood(G, v)) for v in G.nodes())


def minimum_dominating_set(G):
    r"""
    Finds a minimum dominating set for the input graph G.

    The minimum dominating set is the smallest subset of nodes S ⊆ V such that every node in the graph is either
    part of S or adjacent to at least one node in S. This function solves the problem using integer programming.

    Integer Programming Formulation:
    Let x_v ∈ {0, 1} for all v ∈ V, where x_v = 1 if v is in the dominating set, and x_v = 0 otherwise.

    Objective:

    .. math::
        \min \sum_{v \in V} x_v

    Constraints:

    .. math::
        x_v + \sum_{u \in N(v)} x_u \geq 1 \quad \forall v \in V

    Here, *V* is the set of vertices in the graph, and *N(v)* is the open neighborhood of vertex *v*.


    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    set
        A minimum dominating set of nodes in the graph.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.minimum_dominating_set(G))
    {1, 3}

    >>> G = nx.complete_graph(3)
    >>> print(gc.minimum_dominating_set(G))
    {0}
    """
    pulp.LpSolverDefault.msg = 0
    prob = pulp.LpProblem("MinDominatingSet", pulp.LpMinimize)
    variables = {node: pulp.LpVariable("x{}".format(i + 1), 0, 1, pulp.LpBinary) for i, node in enumerate(G.nodes())}

    # Set the domination number objective function.
    prob += pulp.lpSum([variables[n] for n in variables])

    # Set domination number constraints.
    for node in G.nodes():
        combination = [variables[n] for n in variables if n in closed_neighborhood(G, node)]
        prob += pulp.lpSum(combination) >= 1

    prob.solve()
    solution_set = {node for node in variables if variables[node].value() == 1}
    return solution_set

def domination_number(G):
    r"""
    Calculates the domination number of the graph G.

    The domination number is the size of the smallest dominating set in G. It represents the minimum number of nodes
    required such that every node in the graph is either in the dominating set or adjacent to a node in the set.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The domination number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.domination_number(G))
    2
    """
    return len(minimum_dominating_set(G))

def minimum_total_domination_set(G):
    r"""
    Finds a minimum total dominating set for the graph G.

    A total dominating set of a graph G = (V, E) is a subset of nodes S ⊆ V such that every node in V is adjacent
    to at least one node in S. This function solves the problem using integer programming.

    Integer Programming Formulation:
    Let x_v ∈ {0, 1} for all v ∈ V, where x_v = 1 if v is in the dominating set, and x_v = 0 otherwise.

    Objective:

    .. math::
        \min \sum_{v \in V} x_v

    Constraints:

    .. math::
        \sum_{u \in N(v)} x_u \geq 1 \quad \forall v \in V

    Here, *V* is the set of vertices in the graph, and *N(v)* is the open neighborhood of vertex *v*.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    set
        A minimum total dominating set of nodes in the graph.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> result = gc.minimum_total_domination_set(G)
    >>> print(result)
    {1, 2}
    """
    pulp.LpSolverDefault.msg = 0
    prob = pulp.LpProblem("MinTotalDominatingSet", pulp.LpMinimize)
    variables = {node: pulp.LpVariable("x{}".format(i + 1), 0, 1, pulp.LpBinary) for i, node in enumerate(G.nodes())}

    # Set the total domination number objective function.
    prob += pulp.lpSum([variables[n] for n in variables])

    # Set total domination constraints.
    for node in G.nodes():
        combination = [variables[n] for n in variables if n in neighborhood(G, node)]
        prob += pulp.lpSum(combination) >= 1

    prob.solve()
    solution_set = {node for node in variables if variables[node].value() == 1}
    return solution_set

def total_domination_number(G):
    r"""
    Calculates the total domination number of the graph G.

    The total domination number is the size of the smallest total dominating set in G. It represents the minimum
    number of nodes required such that every node in the graph is adjacent to at least one node in the dominating set.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The total domination number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.total_domination_number(G))
    2
    """
    return len(minimum_total_domination_set(G))

def minimum_independent_dominating_set(G):
    r"""
    Finds a minimum independent dominating set for the graph G using integer programming.

    Parameters:
    G (networkx.Graph): The graph to find the independent dominating set for.

    Returns:
    set: A minimum independent dominating set of nodes in G.
    """
    prob = pulp.LpProblem("MinIndependentDominatingSet", pulp.LpMinimize)
    variables = {node: pulp.LpVariable("x{}".format(i + 1), 0, 1, pulp.LpBinary) for i, node in enumerate(G.nodes())}

    # Set the objective function.
    prob += pulp.lpSum([variables[n] for n in variables])

    # Set constraints independent set constraint.
    for e in G.edges():
        prob += variables[e[0]] + variables[e[1]] <= 1

    # Set domination constraints.
    for node in G.nodes():
        combination = [variables[n] for n in variables if n in closed_neighborhood(G, node)]
        prob += pulp.lpSum(combination) >= 1

    prob.solve()
    solution_set = {node for node in variables if variables[node].value() == 1}
    return solution_set

def independent_domination_number(G):
    r"""
    Finds a minimum independent dominating set for the graph G.

    An independent dominating set of a graph G = (V, E) is a dominating set that is also an independent set,
    meaning no two nodes in the set are adjacent. This function uses integer programming to find the smallest such set.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    set
        A minimum independent dominating set of nodes in G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> result = gc.minimum_independent_dominating_set(G)
    >>> print(result)
    {1, 3}
    """
    return len(minimum_independent_dominating_set(G))

def complement_is_connected(G, S):
    r"""
    Checks if the complement of a set S in the graph G induces a connected subgraph.

    The complement of S is defined as the set of all nodes in G that are not in S. This function verifies
    whether the subgraph induced by the complement of S is connected.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    S : set
        A subset of nodes in the graph.

    Returns
    -------
    bool
        True if the subgraph induced by the complement of S is connected, otherwise False.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> S = {0}
    >>> print(gc.complement_is_connected(G, S))
    True

    >>> S = {0, 1}
    >>> print(gc.complement_is_connected(G, S))
    False
    """
    X = G.nodes() - S
    return nx.is_connected(G.subgraph(X))

def is_outer_connected_dominating_set(G, S):
    r"""
    Checks if a given set S is an outer-connected dominating set in the graph G.

    An outer-connected dominating set S ⊆ V of a graph G = (V, E) is a dominating set such that the subgraph
    induced by the complement of S is connected.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    S : set
        A subset of nodes in the graph.

    Returns
    -------
    bool
        True if S is an outer-connected dominating set, otherwise False.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> S = {0}
    >>> print(gc.is_outer_connected_dominating_set(G, S))
    True

    >>> S = {0, 1}
    >>> print(gc.is_outer_connected_dominating_set(G, S))
    False
    """
    return is_dominating_set(G, S) and complement_is_connected(G, S)

def minimum_outer_connected_dominating_set(G):
    r"""
    Finds a minimum outer-connected dominating set for the graph G by trying all subset sizes.

    Parameters:
    G (networkx.Graph): The graph to find the outer-connected dominating set for.

    Returns:
    set: A minimum outer-connected dominating set of nodes in G.
    """
    n = len(G.nodes())
    min_set = None

    for r in range(1, n + 1):  # Try all subset sizes
        for S in combinations(G.nodes(), r):
            S = set(S)
            if is_outer_connected_dominating_set(G, S):
                return S

def outer_connected_domination_number(G):
    r"""
    Finds a minimum outer-connected dominating set for the graph G.

    A minimum outer-connected dominating set is the smallest subset S ⊆ V of the graph G such that:
      1. S is a dominating set.
      2. The subgraph induced by the complement of S is connected.

    This function tries all subset sizes to find the smallest outer-connected dominating set.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    set
        A minimum outer-connected dominating set of nodes in G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> result = gc.minimum_outer_connected_dominating_set(G)
    >>> print(result)
    {0, 2}

    Notes
    -----
    This implementation is exponential in complexity (O(2^n)), as it tries all subsets of nodes in the graph.
    It is not suitable for large graphs.
    """
    return len(minimum_outer_connected_dominating_set(G))


def minimum_roman_dominating_function(graph):
    r"""
    Finds a Roman dominating function for the graph G using integer programming.

    A Roman dominating function (RDF) is an assignment of values 0, 1, or 2 to the vertices of G such that:
      1. Every vertex assigned 0 is adjacent to at least one vertex assigned 2.
      2. The objective is to minimize the sum of vertex values.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.

    Returns
    -------
    dict
        A dictionary containing the RDF values for each vertex and the total objective value.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> solution = gc.minimum_roman_dominating_function(G)
    >>> print(solution)
    {'x': {0: 1, 1: 0, 2: 1, 3: 0}, 'y': {0: 0, 1: 0, 2: 0, 3: 0}, 'objective': 2}
    """
    pulp.LpSolverDefault.msg = 0
    # Initialize the problem
    prob = pulp.LpProblem("RomanDomination", pulp.LpMinimize)

    # Define variables x_v, y_v for each vertex v
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in graph.nodes()}
    y = {v: pulp.LpVariable(f"y_{v}", cat=pulp.LpBinary) for v in graph.nodes()}

    # Objective function: min sum(x_v + 2*y_v)
    prob += pulp.lpSum(x[v] + 2 * y[v] for v in graph.nodes()), "MinimizeCost"

    # Dominance Constraint: x_v + y_v + sum(y_u for u in N(v)) >= 1 for all v
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        prob += x[v] + y[v] + pulp.lpSum(y[u] for u in neighbors) >= 1, f"DominanceConstraint_{v}"

    # Mutual Exclusivity: x_v + y_v <= 1 for all v
    for v in graph.nodes():
        prob += x[v] + y[v] <= 1, f"ExclusivityConstraint_{v}"

    # Solve the problem
    prob.solve()

    # Extract solution
    solution = {
        "x": {v: value(x[v]) for v in graph.nodes()},
        "y": {v: value(y[v]) for v in graph.nodes()},
        "objective": value(prob.objective)
    }

    return solution

def roman_domination_number(graph):
    r"""
    Calculates the Roman domination number of the graph G.

    The Roman domination number is the minimum cost of a Roman dominating function (RDF) on G.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The Roman domination number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.roman_domination_number(G))
    2
    """
    solution = minimum_roman_dominating_function(graph)
    return solution["objective"]

def minimum_double_roman_dominating_function(graph):
    r"""
    Finds a double Roman dominating function for the graph G using integer programming.

    A double Roman dominating function (DRDF) assigns values 0, 1, 2, or 3 to the vertices of G such that:
      1. Every vertex assigned 0 is adjacent to at least one vertex assigned 3 or two vertices assigned 2.
      2. The objective is to minimize the sum of vertex values.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.

    Returns
    -------
    dict
        A dictionary containing the DRDF values for each vertex and the total objective value.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> solution = gc.minimum_double_roman_dominating_function(G)
    >>> print(solution)
    {'x': {0: 0, 1: 1, 2: 0, 3: 0}, 'y': {0: 1, 1: 0, 2: 1, 3: 0}, 'z': {0: 0, 1: 0, 2: 0, 3: 0}, 'objective': 3}
    """
    pulp.LpSolverDefault.msg = 0
    # Initialize the problem
    prob = pulp.LpProblem("DoubleRomanDomination", pulp.LpMinimize)

    # Define variables x_v, y_v, z_v for each vertex v
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in graph.nodes()}
    y = {v: pulp.LpVariable(f"y_{v}", cat=pulp.LpBinary) for v in graph.nodes()}
    z = {v: pulp.LpVariable(f"z_{v}", cat=pulp.LpBinary) for v in graph.nodes()}

    # Objective function: min sum(x_v + 2*y_v + 3*z_v)
    prob += pulp.lpSum(x[v] + 2 * y[v] + 3 * z[v] for v in graph.nodes()), "MinimizeCost"

    # Constraint (1b): xv + yv + zv + 1/2 * sum(yu for u in N(v)) + sum(zu for u in N(v)) >= 1
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        prob += x[v] + y[v] + z[v] + 0.5 * pulp.lpSum(y[u] for u in neighbors) + pulp.lpSum(z[u] for u in neighbors) >= 1, f"Constraint_1b_{v}"

    # Constraint (1c): sum(yu + zu) >= xv for each vertex v
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        prob += pulp.lpSum(y[u] + z[u] for u in neighbors) >= x[v], f"Constraint_1c_{v}"

    # Constraint (1d): xv + yv + zv <= 1
    for v in graph.nodes():
        prob += x[v] + y[v] + z[v] <= 1, f"Constraint_1d_{v}"

    # Solve the problem
    prob.solve()

    # Extract solution
    solution = {
        "x": {v: value(x[v]) for v in graph.nodes()},
        "y": {v: value(y[v]) for v in graph.nodes()},
        "z": {v: value(z[v]) for v in graph.nodes()},
        "objective": value(prob.objective)
    }

    return solution

def double_roman_domination_number(graph):
    r"""
    Calculates the double Roman domination number of the graph G.

    The double Roman domination number is the minimum cost of a double Roman dominating function (DRDF) on G.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The double Roman domination number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.double_roman_domination_number(G))
    3
    """
    solution = minimum_double_roman_dominating_function(graph)
    return solution["objective"]

def minimum_rainbow_dominating_function(G, k):
    r"""
    Finds a rainbow dominating function for the graph G with k colors using integer programming.

    A rainbow dominating set is a set of nodes such that every uncolored node is adjacent to nodes
    of all k different colors.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    k : int
        The number of colors.

    Returns
    -------
    tuple
        A tuple containing a list of colored vertices and a list of uncolored vertices.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> colored, uncolored = gc.minimum_rainbow_dominating_function(G, 2)
    >>> print(colored)
    [(0, 1), (1, 2)]
    >>> print(uncolored)
    [2, 3]
    """
    pulp.LpSolverDefault.msg = 0
    # Create a PuLP problem instance
    prob = pulp.LpProblem("Rainbow_Domination", pulp.LpMinimize)

    # Create binary variables f_vi where f_vi = 1 if vertex v is colored with color i
    f = pulp.LpVariable.dicts("f", ((v, i) for v in G.nodes for i in range(1, k+1)), cat='Binary')

    # Create binary variables x_v where x_v = 1 if vertex v is uncolored
    x = pulp.LpVariable.dicts("x", G.nodes, cat='Binary')

    # Objective function: Minimize the total number of colored vertices
    prob += pulp.lpSum(f[v, i] for v in G.nodes for i in range(1, k+1)), "Minimize total colored vertices"

    # Constraint 1: Each vertex is either colored with one of the k colors or remains uncolored
    for v in G.nodes:
        prob += pulp.lpSum(f[v, i] for i in range(1, k+1)) + x[v] == 1, f"Color or Uncolored constraint for vertex {v}"

    # Constraint 2: If a vertex is uncolored (x_v = 1), it must be adjacent to vertices colored with all k colors
    for v in G.nodes:
        for i in range(1, k+1):
            # Ensure that uncolored vertex v is adjacent to a vertex colored with color i
            prob += pulp.lpSum(f[u, i] for u in G.neighbors(v)) >= x[v], f"Rainbow domination for vertex {v} color {i}"

    # Solve the problem using PuLP's default solver
    prob.solve()

    # Output results
    # print("Status:", pulp.LpStatus[prob.status])

    # Print which vertices are colored and with what color
    colored_vertices = [(v, i) for v in G.nodes for i in range(1, k+1) if value(f[v, i]) == 1]
    uncolored_vertices = [v for v in G.nodes if value(x[v]) == 1]

    print(f"Colored vertices: {colored_vertices}")
    print(f"Uncolored vertices: {uncolored_vertices}")

    return colored_vertices, uncolored_vertices

def rainbow_domination_number(G, k):
    r"""
    Calculates the rainbow domination number of the graph G with k colors.

    The rainbow domination number is the minimum number of colored vertices required to ensure every uncolored
    vertex is adjacent to vertices of all k different colors.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    k : int
        The number of colors.

    Returns
    -------
    int
        The rainbow domination number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.rainbow_domination_number(G, 2))
    2
    """
    colored_vertices, uncolored_vertices = minimum_rainbow_dominating_function(G, k)
    return len(colored_vertices)

def two_rainbow_domination_number(G):
    r"""
    Calculates the 2-rainbow domination number of the graph G.

    The 2-rainbow domination number is a special case of the rainbow domination number where k = 2.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The 2-rainbow domination number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.two_rainbow_domination_number(G))
    2
    """
    return rainbow_domination_number(G, 2)

def three_rainbow_domination_number(G):
    r"""
    Calculates the 3-rainbow domination number of the graph G.

    The 3-rainbow domination number is a special case of the rainbow domination number where k = 3.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The 3-rainbow domination number of G.

    Examples
    --------

    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.three_rainbow_domination_number(G))
    3
    """
    return rainbow_domination_number(G, 3)


def minimum_restrained_dominating_set(G):
    r"""
    Finds a minimum restrained dominating set for the graph G using integer programming.

    A restrained dominating set of a graph G = (V, E) is a subset S ⊆ V such that:
      1. Every vertex in V is either in S or adjacent to a vertex in S (domination condition).
      2. The subgraph induced by V \ S has no isolated vertices (restraint condition).

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    list
        A minimum restrained dominating set of nodes in G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(5)
    >>> restrained_dom_set = gc.minimum_restrained_dominating_set(G)
    >>> print(restrained_dom_set)
    [0, 2, 4]
    """
    pulp.LpSolverDefault.msg = 0
    # Initialize the linear programming problem
    prob = pulp.LpProblem("MinimumRestrainedDomination", pulp.LpMinimize)

    # Decision variables: x_v is 1 if vertex v is in the restrained dominating set, 0 otherwise
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}

    # Objective: Minimize the sum of x_v
    prob += pulp.lpSum(x[v] for v in G.nodes()), "Objective"

    # Constraint 1: Domination condition
    for v in G.nodes():
        prob += x[v] + pulp.lpSum(x[u] for u in G.neighbors(v)) >= 1, f"Domination_{v}"

    # Constraint 2: No isolated vertices in the complement of the dominating set
    for v in G.nodes():
        prob += pulp.lpSum(1 - x[u] for u in G.neighbors(v)) >= (1 - x[v]), f"NoIsolated_{v}"

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract the solution
    restrained_dom_set = [v for v in G.nodes() if value(x[v]) == 1]

    return restrained_dom_set

def restrained_domination_number(G):
    r"""
    Calculates the restrained domination number of the graph G.

    The restrained domination number is the size of a minimum restrained dominating set.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The restrained domination number of G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(5)
    >>> print(gc.restrained_domination_number(G))
    3
    """
    restrained_dom_set = minimum_restrained_dominating_set(G)
    return len(restrained_dom_set)

def min_maximal_matching_number(G):
    r"""
    Calculates the minimum maximal matching number of the graph G.

    The minimum maximal matching number of G is the size of a minimum maximal matching
    in G. This is equivalent to finding the domination number of the line graph of G.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    int
        The minimum maximal matching number of G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.min_maximal_matching_number(G))
    2
    """
    return domination_number(nx.line_graph(G))
