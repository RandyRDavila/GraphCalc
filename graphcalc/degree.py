import networkx as nx

__all__= [
    'degree',
    'degree_sequence',
    'average_degree',
    'maximum_degree',
    'minimum_degree'
]

def degree(G, v):
    r"""
    Returns the degree of a vertex in a graph.

    The degree of a vertex is the number of edges connected to it.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    v : int
        The vertex whose degree is to be calculated.

    Returns
    -------
    int
        The degree of the vertex.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.degree(G, 1)
    2
    >>> gc.degree(G, 0)
    1
    """
    return G.degree(v)

def degree_sequence(G, nonincreasing=True):
    r"""
    Returns the degree sequence of a graph.

    The degree sequence is the list of vertex degrees in the graph, optionally
    sorted in nonincreasing order.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    nonincreasing : bool, optional (default=True)
        If True, the degree sequence is sorted in nonincreasing order.

    Returns
    -------
    list
        The degree sequence of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.degree_sequence(G)
    [2, 2, 1, 1]
    >>> gc.degree_sequence(G, nonincreasing=False)
    [1, 1, 2, 2]
    """
    degrees = [degree(G, v) for v in G.nodes]
    if nonincreasing:
        degrees.sort(reverse=True)
    return degrees

def average_degree(G):
    r"""
    Returns the average degree of a graph.

    The average degree of a graph is the sum of vertex degrees divided by the
    number of vertices.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    float
        The average degree of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.average_degree(G)
    1.5
    """
    degrees = degree_sequence(G)
    return sum(degrees) / len(degrees)

def maximum_degree(G):
    r"""
    Returns the maximum degree of a graph.

    The maximum degree of a graph is the highest vertex degree in the graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    int
        The maximum degree of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.maximum_degree(G)
    2
    """
    degrees = degree_sequence(G)
    return max(degrees)

def minimum_degree(G):
    r"""
    Returns the minimum degree of a graph.

    The minimum degree of a graph is the smallest vertex degree in the graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    int
        The minimum degree of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> gc.minimum_degree(G)
    1
    """
    degrees = degree_sequence(G)
    return min(degrees)
