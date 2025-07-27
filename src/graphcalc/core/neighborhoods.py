
from typing import Set, Hashable, List, Union
import networkx as nx

import graphcalc as gc
from graphcalc.core.basics import SimpleGraph
from graphcalc.utils import enforce_type, GraphLike

__all__= [
    'neighborhood',
    'closed_neighborhood',
    'set_neighbors',
    'set_closed_neighbors',
]

@enforce_type(0, (nx.Graph, SimpleGraph))
def neighborhood(G: GraphLike, v: Hashable) -> Set[Hashable]:
    r"""
    Returns the open neighborhood of a vertex in a graph.

    The neighborhood of a vertex v consists of all vertices directly connected
    to v by an edge.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    v : hashable
        The vertex whose neighborhood is to be computed.

    Returns
    -------
    set
        A set of vertices adjacent to v.

    Raises
    ------
    ValueError
        If the vertex v is not in the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.neighborhood(G, 1)
    {0, 2}
    """
    if v not in G:
        raise ValueError(f"Vertex {v!r} is not in the graph.")
    return set(G[v])

@enforce_type(0, (nx.Graph, SimpleGraph))
def closed_neighborhood(G: GraphLike, v: Hashable) -> Set[Hashable]:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.closed_neighborhood(G, 1)
    {0, 1, 2}
    """
    return neighborhood(G, v) | {v}

@enforce_type(0, (nx.Graph, SimpleGraph))
def set_neighbors(G: GraphLike, S: Union[Set[Hashable], List[Hashable]]) -> Set[Hashable]:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.set_neighbors(G, {1, 2})
    {0, 1, 2, 3}
    """
    return set.union(*[neighborhood(G, v) for v in S])

@enforce_type(0, (nx.Graph, SimpleGraph))
def set_closed_neighbors(G: GraphLike, S: Union[Set[Hashable], List[Hashable]]) -> Set[Hashable]:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = nx.path_graph(4)
    >>> gc.set_closed_neighbors(G, {1, 2})
    {0, 1, 2, 3}
    """
    return set.union(*[closed_neighborhood(G, v) for v in S])
