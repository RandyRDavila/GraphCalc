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
    """
    Returns the order of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    int
        The order of the graph.
    """
    return len(G.nodes)

def size(G):
    """
    Returns the size of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    int
        The size of the graph.
    """
    return len(G.edges)

def connected(G):
    """
    Returns True if the graph is connected, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected, False otherwise.
    """
    return nx.is_connected(G)

def connected_and_bipartite(G):
    """
    Returns True if the graph is connected and bipartite, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and bipartite, False otherwise.
    """
    return nx.is_connected(G) and nx.is_bipartite(G)

def tree(G):
    """
    Returns True if the graph is a tree, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is a tree, False otherwise.
    """
    return nx.is_tree(G)

def connected_and_regular(G):
    """
    Returns True if the graph is connected and regular, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and regular, False otherwise.
    """
    return nx.is_connected(G) and nx.is_regular(G)

def connected_and_eulerian(G):
    """
    Returns True if the graph is connected and Eulerian, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and Eulerian, False otherwise.
    """
    return nx.is_connected(G) and nx.is_eulerian(G)

def connected_and_planar(G):
    """
    Returns True if the graph is connected and planar, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and planar, False otherwise.
    """
    return nx.is_connected(G) and nx.check_planarity(G)[0]

def connected_and_bipartite(G):
    """
    Returns True if the graph is connected and bipartite, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and bipartite, False otherwise.
    """
    return nx.is_connected(G) and nx.is_bipartite(G)

def connected_and_chordal(G):
    """
    Returns True if the graph is connected and chordal, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and chordal, False otherwise.
    """
    return nx.is_connected(G) and nx.is_chordal(G)

def connected_and_cubic(G):
    """
    Returns True if the graph is connected and cubic, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and cubic, False otherwise.
    """
    return nx.is_connected(G) and maximum_degree(G) == 3

def connected_and_subcubic(G):
    """
    Returns True if the graph is connected and subcubic, False otherwise.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    bool
        True if the graph is connected and subcubic, False otherwise.
    """
    return nx.is_connected(G) and maximum_degree(G) <= 3

def diameter(G):
    """
    Returns the diameter of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    int
        The diameter of the graph.
    """
    return nx.diameter(G)

def radius(G):
    """
    Returns the radius of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    int
        The radius of the graph.
    """
    return nx.radius(G)

def average_shortest_path_length(G):
    """
    Returns the average shortest path length of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    float
        The average shortest path length of the graph.
    """
    return nx.average_shortest_path_length(G)
