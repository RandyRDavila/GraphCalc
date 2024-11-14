import networkx as nx

__all__= [
    'neighborhood',
    'closed_neighborhood',
    'set_neighbors',
    'set_closed_neighbors',
]

def neighborhood(G, v):
    """
    Returns the neighborhood of a vertex in a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    v : int
        The vertex.

    Returns
    -------
    set
        The set of neighbors of the vertex.
    """
    return set(G.neighbors(v))

def closed_neighborhood(G, v):
    """
    Returns the closed neighborhood of a vertex in a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    v : int
        The vertex.

    Returns
    -------
    set
        The set of neighbors of the vertex, including the vertex itself.
    """
    return neighborhood(G, v) | {v}

def set_neighbors(G, S):
    """
    Returns the set of neighbors of a set of vertices in a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    S : set
        The set of vertices.

    Returns
    -------
    set
        The set of neighbors of the vertices in the set.
    """
    return set.union(*[neighborhood(G, v) for v in S])

def set_closed_neighbors(G, S):
    """
    Returns the set of closed neighbors of a set of vertices in a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    S : set
        The set of vertices.

    Returns
    -------
    set
        The set of closed neighbors of the vertices in the set.
    """
    return set.union(*[closed_neighborhood(G, v) for v in S])
