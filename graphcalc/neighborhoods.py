import networkx as nx

__all__= [
    'neighborhood',
    'closed_neighborhood',
    'set_neighbors',
    'set_closed_neighbors',
]

def neighborhood(G, v):
    r"""
    Returns the neighborhood of a vertex in a graph.

    The neighborhood of a vertex v consists of all vertices directly connected
    to v by an edge.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    v : int
        The vertex whose neighborhood is to be computed.

    Returns
    -------
    set
        A set of vertices adjacent to v.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.neighborhood(G, 1)
    {0, 2}
    """
    return set([u for u in G[v]])

def closed_neighborhood(G, v):
    r"""
    Returns the closed neighborhood of a vertex in a graph.

    The closed neighborhood of a vertex v consists of v and all vertices
    directly connected to v by an edge.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    v : int
        The vertex whose closed neighborhood is to be computed.

    Returns
    -------
    set
        A set of vertices including v and its neighbors.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.closed_neighborhood(G, 1)
    {0, 1, 2}
    """
    return neighborhood(G, v) | {v}

def set_neighbors(G, S):
    r"""
    Returns the set of neighbors of a set of vertices in a graph.

    The neighbors of a set of vertices S are all vertices adjacent to at least
    one vertex in S.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    S : set
        A set of vertices whose neighbors are to be computed.

    Returns
    -------
    set
        A set of vertices adjacent to at least one vertex in S.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.set_neighbors(G, {1, 2})
    {0, 3}
    """
    return set.union(*[neighborhood(G, v) for v in S])

def set_closed_neighbors(G, S):
    r"""
    Returns the set of closed neighbors of a set of vertices in a graph.

    The closed neighbors of a set of vertices S are all vertices in S along
    with all vertices adjacent to at least one vertex in S.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    S : set
        A set of vertices whose closed neighbors are to be computed.

    Returns
    -------
    set
        A set of vertices in S and their neighbors.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.set_closed_neighbors(G, {1, 2})
    {0, 1, 2, 3}
    """
    return set.union(*[closed_neighborhood(G, v) for v in S])
