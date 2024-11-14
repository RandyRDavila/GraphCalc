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
    """
    Checks if a set of nodes, S, is a dominating set in the graph G.

    Parameters:
    G (networkx.Graph): The graph to check.
    S (set): A set of nodes in the graph.

    Returns:
    bool: True if S is a dominating set, otherwise False.
    """
    return all(any(u in S for u in closed_neighborhood(G, v)) for v in G.nodes())

def minimum_dominating_set(G):
    """
    Finds a minimum dominating set for the graph G using integer programming.

    Parameters:
    G (networkx.Graph): The graph to find the dominating set for.

    Returns:
    set: A minimum dominating set of nodes in G.
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
    """
    Calculates the domination number of the graph G, which is the size of a minimum dominating set.

    Parameters:
    G (networkx.Graph): The graph to calculate the domination number for.

    Returns:
    int: The domination number of G.
    """
    return len(minimum_dominating_set(G))

def minimum_total_domination_set(G):
    """
    Finds a minimum total dominating set for the graph G using integer programming.

    Parameters:
    G (networkx.Graph): The graph to find the total dominating set for.

    Returns:
    set: A minimum total dominating set of nodes in G.
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
    """
    Calculates the total domination number of the graph G, which is the size of a minimum total dominating set.

    Parameters:
    G (networkx.Graph): The graph to calculate the total domination number for.

    Returns:
    int: The total domination number of G.
    """
    return len(minimum_total_domination_set(G))

def minimum_independent_dominating_set(G):
    """
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
    """
    Calculates the independent domination number of the graph G, which is the size of a minimum independent dominating set.

    Parameters:
    G (networkx.Graph): The graph to calculate the independent domination number for.

    Returns:
    int: The independent domination number of G.
    """
    return len(minimum_independent_dominating_set(G))

def complement_is_connected(G, S):
    """
    Checks if the complement of set S in the graph G is connected.

    Parameters:
    G (networkx.Graph): The graph to check.
    S (set): A set of nodes in the graph.

    Returns:
    bool: True if the complement of S is connected, otherwise False.
    """
    X = G.nodes() - S
    return nx.is_connected(G.subgraph(X))

def is_outer_connected_dominating_set(G, S):
    """
    Checks if set S is an outer-connected dominating set in the graph G.

    Parameters:
    G (networkx.Graph): The graph to check.
    S (set): A set of nodes in the graph.

    Returns:
    bool: True if S is an outer-connected dominating set, otherwise False.
    """
    return is_dominating_set(G, S) and complement_is_connected(G, S)

def minimum_outer_connected_dominating_set(G):
    """
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
    """
    Calculates the outer-connected domination number of the graph G, which is the size of a minimum outer-connected dominating set.

    Parameters:
    G (networkx.Graph): The graph to calculate the outer-connected domination number for.

    Returns:
    int: The outer-connected domination number of G.
    """
    return len(minimum_outer_connected_dominating_set(G))


def minimum_roman_dominating_function(graph):
    """
    Finds a Roman dominating function for the graph G using integer programming.

    Parameters:
    G (networkx.Graph): The graph to find the Roman dominating function for.

    Returns:
    dict: A solution with the values for each vertex and the objective value.
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
    """
    Calculates the Roman domination number of the graph G.

    Parameters:
    G (networkx.Graph): The graph to calculate the Roman domination number for.

    Returns:
    int: The Roman domination number of G.
    """
    solution = minimum_roman_dominating_function(graph)
    return solution["objective"]

def minimum_double_roman_dominating_function(graph):
    """
    Finds a double Roman dominating function for the graph G using integer programming.

    Parameters:
    G (networkx.Graph): The graph to find the double Roman dominating function for.

    Returns:
    dict: A solution with the values for each vertex and the objective value.
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
    """
    Calculates the double Roman domination number of the graph G.

    Parameters:
    G (networkx.Graph): The graph to calculate the double Roman domination number for.

    Returns:
    int: The double Roman domination number of G.
    """
    solution = minimum_double_roman_dominating_function(graph)
    return solution["objective"]

def minimum_rainbow_dominating_function(G, k):
    """
    Finds a rainbow dominating set for the graph G with k colors using integer programming.

    Parameters:
    G (networkx.Graph): The graph to find the rainbow dominating set for.
    k (int): The number of colors.

    Returns:
    tuple: A list of colored vertices and a list of uncolored vertices.
    """
    pulp.LpSolverDefault.msg = 0
    # Create a PuLP problem instance
    prob = pulp.LpProblem("Rainbow_Domination", pulp.pulp.LpMinimize)

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
    """
    Calculates the rainbow domination number of the graph G with k colors.

    Parameters:
    G (networkx.Graph): The graph to calculate the rainbow domination number for.
    k (int): The number of colors.

    Returns:
    int: The rainbow domination number of G.
    """
    colored_vertices, uncolored_vertices = minimum_rainbow_dominating_function(G, k)
    return len(colored_vertices)

def two_rainbow_domination_number(G):
    """
    Calculates the 2-rainbow domination number of the graph G.

    Parameters:
    G (networkx.Graph): The graph to calculate the 2-rainbow domination number for.

    Returns:
    int: The 2-rainbow domination number of G.
    """
    return rainbow_domination_number(G, 2)

def three_rainbow_domination_number(G):
    """
    Calculates the 3-rainbow domination number of the graph G.

    Parameters:
    G (networkx.Graph): The graph to calculate the 3-rainbow domination number for.

    Returns:
    int: The 3-rainbow domination number of G.
    """
    return rainbow_domination_number(G, 3)


def minimum_restrained_dominating_set(G):
    """
    Finds a minimum restrained dominating set for the graph G using integer programming.

    Parameters:
    G (networkx.Graph): The graph to find the restrained dominating set for.

    Returns:
    list: A minimum restrained dominating set of nodes in G.
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
    """
    Calculates the restrained domination number of the graph G.

    Parameters:
    G (networkx.Graph): The graph to calculate the restrained domination number for.

    Returns:
    int: The restrained domination number of G.
    """
    restrained_dom_set = minimum_restrained_dominating_set(G)
    return len(restrained_dom_set)

def min_maximal_matching_number(G):
    """
    Calculates the minimum maximal matching number of the graph G by finding the domination number of its line graph.

    Parameters:
    G (networkx.Graph): The graph to calculate the minimum maximal matching number for.

    Returns:
    int: The minimum maximal matching number of G.
    """
    return domination_number(nx.line_graph(G))
