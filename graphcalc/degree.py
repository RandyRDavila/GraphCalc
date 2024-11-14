import networkx as nx

__all__= [
    'degree',
    'degree_sequence',
    'average_degree',
    'maximum_degree',
    'minimum_degree'
]

def degree(G, v):
    """
    Returns the degree of a vertex in a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    v : int
        The vertex.

    Returns
    -------
    int
        The degree of the vertex.
    """
    return G.degree(v)

def degree_sequence(G, nonincreasing=True):
    """
    Returns the degree sequence of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    nonincreasing : bool
        If True, the sequence is sorted in nonincreasing order.

    Returns
    -------
    list
        The degree sequence of the graph.
    """
    degrees = [degree(G, v) for v in G.nodes]
    if nonincreasing:
        degrees.sort(reverse=True)
    return degrees

def average_degree(G):
    """
    Returns the average degree of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    float
        The average degree of the graph.
    """
    degrees = degree_sequence(G)
    return sum(degrees) / len(degrees)

def maximum_degree(G):
    """
    Returns the maximum degree of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    int
        The maximum degree of the graph.
    """
    degrees = degree_sequence(G)
    return max(degrees)

def minimum_degree(G):
    """
    Returns the minimum degree of a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    int
        The minimum degree of the graph.
    """
    degrees = degree_sequence(G)
    return min(degrees)
