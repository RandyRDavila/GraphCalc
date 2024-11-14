import networkx as nx
from itertools import combinations
import pulp
from pulp import (
    value,
    PULP_CBC_CMD,
)

from .neighborhoods import neighborhood, closed_neighborhood

__all__ = [
    "domination_number",
    "total_domination_number",
    "independent_domination_number",
    "outer_connected_domination_number",
    "roman_domination_number",
    "double_roman_domination_number",
    "two_rainbow_domination_number",
    "three_rainbow_domination_number",
    "min_maximal_matching_number",
]

def is_dominating_set(G, S):
    return all(any(u in S for u in closed_neighborhood(G, v)) for v in G.nodes())

def minimum_dominating_set(G):
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
    return len(minimum_dominating_set(G))

def minimum_total_domination_set(G):
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
    return len(minimum_total_domination_set(G))

def minimum_independent_dominating_set(G):
    prob = pulp.LpProblem("MinIndependentDominatingSet", pulp.LpMinimize)
    variables = {node: pulp.LpVariable("x{}".format(i + 1), 0, 1, pulp.LpBinary) for i, node in enumerate(G.nodes())}

    # Set the objective function.
    prob += pulp.lpSum([variables[n] for n in variables])

    # Set constraints independent set constraint.
    for e in G.edges():
        prob += variables[e[0]] + variables[e[1]] <= 1

    # Set domination constraints.
    for node in G.nodes():
        combination = [variables[n] for n in variables if n in neighborhood(G, node)]
        prob += pulp.lpSum(combination) >= 1

    prob.solve()
    solution_set = {node for node in variables if variables[node].value() == 1}
    return solution_set

def independent_domination_number(G):
    return len(minimum_independent_dominating_set(G))

def complement_is_connected(G, S):
    X = G.nodes() - S
    return nx.is_connected(G.subgraph(X))

def is_outer_connected_dominating_set(G, S):
    return is_dominating_set(G, S) and complement_is_connected(G, S)

def min_outer_connected_dominating_set(G):
    n = len(G.nodes())
    min_set = None

    for r in range(1, n + 1):  # Try all subset sizes
        for S in combinations(G.nodes(), r):
            S = set(S)
            if is_outer_connected_dominating_set(G, S):
                return S

def outer_connected_domination_number(G):
    return len(min_outer_connected_dominating_set(G))


def roman_domination(graph):
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
    solution = roman_domination(graph)
    return solution["objective"]

def double_roman_domination(graph):
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
    solution = double_roman_domination(graph)
    return solution["objective"]

def solve_rainbow_domination(G, k):
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
    colored_vertices, uncolored_vertices = solve_rainbow_domination(G, k)
    return len(colored_vertices)

def two_rainbow_domination_number(G):
    return rainbow_domination_number(G, 2)

def three_rainbow_domination_number(G):
    return rainbow_domination_number(G, 3)


def minimum_restrained_dominating_set(G):
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
    restrained_dom_set = minimum_restrained_dominating_set(G)
    return len(restrained_dom_set)

def min_maximal_matching_number(G):
    return domination_number(nx.line_graph(G))
