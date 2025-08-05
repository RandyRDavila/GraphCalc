"""
General graph generators.

This module re-exports NetworkX graph generators and can optionally include
additional custom general-purpose generators.
"""

import networkx as nx
import graphcalc as gc

__all__ = [
    "complete_graph",
    "cycle_graph",
    "path_graph",
    "star_graph",
    "wheel_graph",
    "grid_2d_graph",
    "barbell_graph",
    "ladder_graph",
    "binomial_tree",
    "balanced_tree",
    "erdos_renyi_graph",
    "watts_strogatz_graph",
    "barabasi_albert_graph",
    "powerlaw_cluster_graph",
    "random_geometric_graph",
    "random_regular_graph",
    "petersen_graph",
    "diamond_necklace",
]


def complete_graph(n: int) -> gc.SimpleGraph:
    r"""
    Return the complete graph `K_n` with `n` nodes.

    The complete graph `K_n` is the simple undirected graph with `n` nodes
    and a single edge for every pair of distinct nodes.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.

    Returns
    -------
    gc.SimpleGraph
        The complete graph `K_n`.

    Examples
    --------
    >>> from graphcalc.generators import complete_graph
    >>> G = complete_graph(4)
    """
    return gc.SimpleGraph(nx.complete_graph(n).edges, name=f"Complete Graph K_{n}")

def cycle_graph(n: int) -> gc.SimpleGraph:
    r"""
    Return the cycle graph `C_n` with `n` nodes.

    The cycle graph `C_n` is the simple undirected graph with `n` nodes
    arranged in a cycle, where each node is connected to its two neighbors.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.

    Returns
    -------
    gc.SimpleGraph
        The cycle graph `C_n`.

    Examples
    --------
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    """
    return gc.SimpleGraph(nx.cycle_graph(n).edges, name=f"Cycle Graph C_{n}")

def path_graph(n: int) -> gc.SimpleGraph:
    r"""
    Return the path graph `P_n` with `n` nodes.

    The path graph `P_n` is the simple undirected graph with `n` nodes
    arranged in a line, where each node is connected to its two neighbors.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.

    Returns
    -------
    gc.SimpleGraph
        The path graph `P_n`.

    Examples
    --------
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    """
    return gc.SimpleGraph(nx.path_graph(n).edges, name=f"Path Graph P_{n}")

def star_graph(n: int) -> gc.SimpleGraph:
    r"""
    Return the star graph `S_n` with `n` nodes.

    The star graph `S_n` is the simple undirected graph with `n` nodes
    arranged in a star-like pattern, where one node is connected to all others.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.

    Returns
    -------
    gc.SimpleGraph
        The star graph `S_n`.

    Examples
    --------
    >>> from graphcalc.generators import star_graph
    >>> G = star_graph(4)
    """
    return gc.SimpleGraph(nx.star_graph(n).edges, name=f"Star Graph S_{n}")

def wheel_graph(n: int) -> gc.SimpleGraph:
    r"""
    Return the wheel graph `W_n` with `n` nodes.

    The wheel graph `W_n` is the simple undirected graph with `n` nodes
    arranged in a cycle, where one node is connected to all others.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.

    Returns
    -------
    gc.SimpleGraph
        The wheel graph `W_n`.

    Examples
    --------
    >>> from graphcalc.generators import wheel_graph
    >>> G = wheel_graph(4)
    """
    return gc.SimpleGraph(nx.wheel_graph(n).edges, name=f"Wheel Graph W_{n}")

def grid_2d_graph(m: int, n: int) -> gc.SimpleGraph:
    r"""
    Return the 2D grid graph `G_{m,n}` with `m` rows and `n` columns.

    The 2D grid graph `G_{m,n}` is the simple undirected graph with `m * n`
    nodes arranged in a 2D grid pattern, where each node is connected to its
    four neighbors (if they exist).

    Parameters
    ----------
    m : int
        The number of rows in the grid.
    n : int
        The number of columns in the grid.

    Returns
    -------
    gc.SimpleGraph
        The 2D grid graph `G_{m,n}`.

    Examples
    --------
    >>> from graphcalc.generators import grid_2d_graph
    >>> G = grid_2d_graph(2, 3)
    """
    return gc.SimpleGraph(nx.grid_2d_graph(m, n).edges, name=f"2D Grid Graph G_{{{m},{n}}}")

def barbell_graph(m: int, n: int) -> gc.SimpleGraph:
    r"""
    Return the barbell graph `B_{m,n}` with `m` nodes in each complete graph.

    The barbell graph `B_{m,n}` is the simple undirected graph with `2 * m + n`
    nodes arranged in a barbell-like pattern, where two complete graphs with `m`
    nodes are connected by a path graph with `n` nodes.

    Parameters
    ----------
    m : int
        The number of nodes in each complete graph.
    n : int
        The number of nodes in the path graph.

    Returns
    -------
    gc.SimpleGraph
        The barbell graph `B_{m,n}`.

    Examples
    --------
    >>> from graphcalc.generators import barbell_graph
    >>> G = barbell_graph(2, 3)
    """
    return gc.SimpleGraph(nx.barbell_graph(m, n).edges, name=f"Barbell Graph B_{{{m},{n}}}")

def ladder_graph(n: int) -> gc.SimpleGraph:
    r"""
    Return the ladder graph `L_n` with `2 * n` nodes.

    The ladder graph `L_n` is the simple undirected graph with `2 * n` nodes
    arranged in a ladder-like pattern, where two paths with `n` nodes are
    connected by `n` edges.

    Parameters
    ----------
    n : int
        The number of nodes in each path.

    Returns
    -------
    gc.SimpleGraph
        The ladder graph `L_n`.

    Examples
    --------
    >>> from graphcalc.generators import ladder_graph
    >>> G = ladder_graph(3)
    """
    return gc.SimpleGraph(nx.ladder_graph(n).edges, name=f"Ladder Graph L_{n}")

def binomial_tree(n: int) -> gc.SimpleGraph:
    r"""
    Return the binomial tree `BT_n` with `n` levels.

    The binomial tree `BT_n` is the simple undirected tree with `2^(n+1) - 1`
    nodes arranged in a binary tree pattern, where each node has either 0 or 2
    children.

    Parameters
    ----------
    n : int
        The number of levels in the tree.

    Returns
    -------
    gc.SimpleGraph
        The binomial tree `BT_n`.

    Examples
    --------
    >>> from graphcalc.generators import binomial_tree
    >>> G = binomial_tree(3)
    """
    return gc.SimpleGraph(nx.binomial_tree(n).edges, name=f"Binomial Tree BT_{n}")

def balanced_tree(r: int, h: int) -> gc.SimpleGraph:
    r"""
    Return the balanced tree `BT_{r,h}` with branching factor `r` and height `h`.

    The balanced tree `BT_{r,h}` is the simple undirected tree with `r^(h+1) - 1`
    nodes arranged in a balanced tree pattern, where each node has either 0 or `r`
    children.

    Parameters
    ----------
    r : int
        The branching factor of the tree.
    h : int
        The height of the tree.

    Returns
    -------
    gc.SimpleGraph
        The balanced tree `BT_{r,h}`.

    Examples
    --------
    >>> from graphcalc.generators import balanced_tree
    >>> G = balanced_tree(2, 3)
    """
    return gc.SimpleGraph(nx.balanced_tree(r, h).edges, name=f"Balanced Tree BT_{{{r},{h}}}")

def erdos_renyi_graph(n: int, p: float, seed=None) -> gc.SimpleGraph:
    r"""
    Return the Erdos-Renyi random graph `G_{n,p}` with `n` nodes and edge probability `p`.

    The Erdos-Renyi random graph `G_{n,p}` is the simple undirected graph with `n`
    nodes, where each pair of nodes is connected by an edge with probability `p`.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    p : float
        The probability of an edge between any pair of nodes.
    seed : int (optional)
        The randomization seed for constructing the Erdos-Renyi random graph.

    Returns
    -------
    gc.SimpleGraph
        The Erdos-Renyi random graph `G_{n,p}`.

    Examples
    --------
    >>> from graphcalc.generators import erdos_renyi_graph
    >>> G = erdos_renyi_graph(4, 0.5)
    """
    return gc.SimpleGraph(nx.erdos_renyi_graph(n, p, seed=seed).edges, name=f"Erdos-Renyi Graph G_{{{n},{p}}}")

def watts_strogatz_graph(n: int, k: int, p: float, seed=None) -> gc.SimpleGraph:
    r"""
    Return the Watts-Strogatz small-world graph `WS_{n,k,p}` with `n` nodes, degree `k`, and rewiring probability `p`.

    The Watts-Strogatz small-world graph `WS_{n,k,p}` is the simple undirected graph with `n`
    nodes, where each node is connected to its `k` nearest neighbors in a ring lattice
    pattern, and each edge is rewired with probability `p`.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    k : int
        The degree of each node in the ring lattice.
    p : float
        The probability of rewiring each edge.
    seed : int (optional)
        The randomization seed for constructing the Watts-Strogatz small-world graph.

    Returns
    -------
    gc.SimpleGraph
        The Watts-Strogatz small-world graph `WS_{n,k,p}`.

    Examples
    --------
    >>> from graphcalc.generators import watts_strogatz_graph
    >>> G = watts_strogatz_graph(4, 2, 0.5)
    """
    return gc.SimpleGraph(nx.watts_strogatz_graph(n, k, p, seed=seed).edges, name=f"Watts-Strogatz Graph WS_{{{n},{k},{p}}}")

def barabasi_albert_graph(n: int, m: int, seed=None) -> gc.SimpleGraph:
    r"""
    Return the Barabasi-Albert preferential attachment graph `BA_{n,m}` with `n` nodes and `m` edges per new node.

    The Barabasi-Albert preferential attachment graph `BA_{n,m}` is the simple undirected
    graph with `n` nodes, where new nodes are added one at a time and connected to `m`
    existing nodes with probability proportional to their degree.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    m : int
        The number of edges per new node.
    seed : int (optional)
        The randomization seed for constructing the Barabasi-Albert graph.

    Returns
    -------
    gc.SimpleGraph
        The Barabasi-Albert preferential attachment graph `BA_{n,m}`.

    Examples
    --------
    >>> from graphcalc.generators import barabasi_albert_graph
    >>> G = barabasi_albert_graph(4, 2)
    """
    return gc.SimpleGraph(nx.barabasi_albert_graph(n, m, seed=seed).edges, name=f"Barabasi-Albert Graph BA_{{{n},{m}}}")

def powerlaw_cluster_graph(n: int, m: int, p: float, seed=None) -> gc.SimpleGraph:
    r"""
    Return the powerlaw cluster graph `PLC_{n,m,p}` with `n` nodes, `m` edges per node, and rewiring probability `p`.

    The powerlaw cluster graph `PLC_{n,m,p}` is the simple undirected graph with `n`
    nodes, where each node is connected to its `m` nearest neighbors in a ring lattice
    pattern, and each edge is rewired with probability `p`.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    m : int
        The number of edges per node in the ring lattice.
    p : float
        The probability of rewiring each edge.
    seed : int (optional)
        The randomization seed for constructing the powerlaw cluster graph.

    Returns
    -------
    gc.SimpleGraph
        The powerlaw cluster graph `PLC_{n,m,p}`.

    Examples
    --------
    >>> from graphcalc.generators import powerlaw_cluster_graph

    >>> G = powerlaw_cluster_graph(4, 2, 0.5)
    """
    return gc.SimpleGraph(nx.powerlaw_cluster_graph(n, m, p, seed=seed).edges, name=f"Powerlaw Cluster Graph PLC_{{{n},{m},{p}}}")

def random_geometric_graph(n: int, radius: int, seed=None) -> gc.SimpleGraph:
    r"""
    Return the random geometric graph `RGG_{n,r}` with `n` nodes and radius `r`.

    The random geometric graph `RGG_{n,r}` is the simple undirected graph with `n`
    nodes, where each pair of nodes is connected by an edge if their Euclidean
    distance is less than `r`.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    radius : float
        The radius of the geometric graph.
    seed : int (optional)
        The randomization seed for constructing the random geometric graph.

    Returns
    -------
    gc.SimpleGraph
        The random geometric graph `RGG_{n,r}`.

    Examples
    --------
    >>> from graphcalc.generators import random_geometric_graph
    >>> G = random_geometric_graph(4, 0.5)
    """
    return gc.SimpleGraph(nx.random_geometric_graph(n, radius, seed=seed).edges, name=f"Random Geometric Graph RGG_{{{n},{radius}}}")

def random_regular_graph(d: int, n: int, seed=None) -> gc.SimpleGraph:
    r"""
    Return the random regular graph `RRG_{d,n}` with degree `d` and `n` nodes.

    The random regular graph `RRG_{d,n}` is the simple undirected graph with `n`
    nodes, where each node has degree `d` and edges are assigned randomly.

    Parameters
    ----------
    d : int
        The degree of each node in the graph.
    n : int
        The number of nodes in the graph.
    seed : int (optional)
        The randomization seed for constructing the random :math`d`-regular graph.

    Returns
    -------
    gc.SimpleGraph
        The random :math`d`-regular graph.

    Examples
    --------
    >>> from graphcalc.generators import random_regular_graph
    >>> G = random_regular_graph(3, 4)
    """
    return gc.SimpleGraph(nx.random_regular_graph(d, n, seed=seed).edges, name=f"Random Regular Graph RRG_{{{d},{n}}}")

def petersen_graph() -> gc.SimpleGraph:
    r"""
    Return the Petersen graph `P`.

    The Petersen graph `P` is the simple undirected graph with 10 nodes and 15 edges,
    where nodes are arranged in a pentagon with a star-like pattern inside.

    Returns
    -------
    graphcalc.SimpleGraph
        The Petersen graph `P`.

    Examples
    --------
    >>> from graphcalc.generators import petersen_graph
    >>> G = petersen_graph()
    """
    return gc.SimpleGraph(nx.petersen_graph().edges, name="Petersen Graph P")

def diamond_necklace(k: int) -> gc.SimpleGraph:
    r"""
    Build the diamond‐necklace graph :math:`N_k`:

    Parameters
    ----------
    k : int
        Number of diamonds to chain together. Must be ≥ 1.

    Returns
    -------
    graphcalc.SimpleGraph
        The diamond-necklace graph :math:`N_k`.

    Examples
    --------
    >>> from graphcalc.generators import diamond_necklace
    >>> G = diamond_necklace(2)
    >>> G.order(), G.size()
    (8, 12)
    """
    G = nx.Graph()
    deg2_verts = []

    # 1) Build k disjoint diamonds
    for i in range(k):
        base = 4 * i
        # add the 4 nodes of this diamond
        G.add_nodes_from(range(base, base + 4))
        # add all edges of K4 except the one between base and base+3
        for u in range(base, base+4):
            for v in range(u+1, base+4):
                if not (u == base and v == base+3):
                    G.add_edge(u, v)
        # record the two degree-2 vertices for linking
        deg2_verts.append((base, base+3))

    # 2) Link them in a cycle at those degree-2 vertices
    for i in range(k):
        j = (i + 1) % k
        # connect the "upper" deg-2 of diamond i to the "lower" deg-2 of diamond j
        _, high_i = deg2_verts[i]
        low_j, _ = deg2_verts[j]
        G.add_edge(high_i, low_j)

    return gc.SimpleGraph(G.edges, name=f"Diamond-Necklace-{k}")
