import networkx as nx
from .degree import *

__all__= [
    'order',
    'size',
    'connected',
    'diameter',
    'radius',
    'average_shortest_path_length',
    'connected_and_bipartite',
    'connected_and_chordal',
    'connected_and_cubic',
    'connected_and_eulerian',
    'connected_and_planar',
    'connected_and_regular',
    'connected_and_subcubic',
    'tree',
]

def order(G):
    r"""
    Returns the order of a graph, which is the number of vertices.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    int
        The order of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(4)
    >>> gc.order(G)
    4
    """
    return len(G.nodes)

def size(G):
    r"""
    Returns the size of a graph, which is the number of edges.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    int
        The size of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.size(G)
    3
    """
    return len(G.edges)

def connected(G):
    r"""
    Checks if the graph is connected.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.connected(G)
    True
    """
    return nx.is_connected(G)

def connected_and_bipartite(G):
    r"""
    Checks if the graph is both connected and bipartite.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and bipartite, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.connected_and_bipartite(G)
    True
    """
    return nx.is_connected(G) and nx.is_bipartite(G)

def tree(G):
    r"""
    Checks if the graph is a tree.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a tree, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.tree(G)
    True
    """
    return nx.is_tree(G)

def connected_and_regular(G):
    r"""
    Checks if the graph is both connected and regular.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and regular, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.cycle_graph(4)
    >>> gc.connected_and_regular(G)
    True
    """
    return nx.is_connected(G) and nx.is_regular(G)

def connected_and_eulerian(G):
    r"""
    Checks if the graph is both connected and Eulerian.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and Eulerian, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.cycle_graph(4)
    >>> gc.connected_and_eulerian(G)
    True
    """
    return nx.is_connected(G) and nx.is_eulerian(G)

def connected_and_planar(G):
    r"""
    Checks if the graph is both connected and planar.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and planar, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.connected_and_planar(G)
    True
    """
    return nx.is_connected(G) and nx.check_planarity(G)[0]

def connected_and_bipartite(G):
    r"""
    Checks if the graph is both connected and bipartite.

    A graph is connected if there is a path between every pair of vertices.
    A graph is bipartite if its vertices can be divided into two disjoint sets
    such that every edge connects a vertex in one set to a vertex in the other set.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and bipartite, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(4)
    >>> connected_and_bipartite(G)
    True

    >>> H = nx.cycle_graph(5)  # Odd-length cycle is not bipartite
    >>> connected_and_bipartite(H)
    False

    >>> I = nx.Graph()
    >>> I.add_edges_from([(1, 2), (3, 4)])  # Disconnected graph
    >>> connected_and_bipartite(I)
    False
    """
    return nx.is_connected(G) and nx.is_bipartite(G)

def connected_and_chordal(G):
    r"""
    Checks if the graph is both connected and chordal.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and chordal, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.complete_graph(4)
    >>> gc.connected_and_chordal(G)
    True
    """
    return nx.is_connected(G) and nx.is_chordal(G)

def connected_and_cubic(G):
    r"""
    Checks if the graph is both connected and cubic.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and cubic, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.petersen_graph()
    >>> gc.connected_and_cubic(G)
    True
    """
    return nx.is_connected(G) and maximum_degree(G) == minimum_degree(G) == 3

def connected_and_subcubic(G):
    r"""
    Checks if the graph is both connected and subcubic.

    A graph is subcubic if the degree of every vertex is at most 3.
    A graph is connected if there is a path between every pair of vertices.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and subcubic, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> from graphcalc import connected_and_subcubic

    >>> G = nx.cycle_graph(4)  # Degree of all nodes is 2, connected
    >>> connected_and_subcubic(G)
    True

    >>> H = nx.path_graph(5)  # Maximum degree is 2, connected
    >>> connected_and_subcubic(H)
    True

    >>> I = nx.star_graph(4)  # Maximum degree is 4, not subcubic
    >>> connected_and_subcubic(I)
    False

    >>> J = nx.Graph()
    >>> J.add_edges_from([(1, 2), (3, 4)])  # Disconnected graph
    >>> connected_and_subcubic(J)
    False
    """
    return nx.is_connected(G) and maximum_degree(G) <= 3

def diameter(G):
    r"""
    Returns the diameter of the graph.

    The diameter is the maximum shortest path length between any pair of nodes.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    int
        The diameter of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.diameter(G)
    3
    """
    return nx.diameter(G)

def radius(G):
    r"""
    Returns the radius of the graph.

    The radius is the minimum eccentricity among all vertices.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    int
        The radius of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.radius(G)
    2
    """
    return nx.radius(G)

def average_shortest_path_length(G):
    r"""
    Returns the average shortest path length of the graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    float
        The average shortest path length of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.average_shortest_path_length(G)
    1.5
    """
    return nx.average_shortest_path_length(G)
