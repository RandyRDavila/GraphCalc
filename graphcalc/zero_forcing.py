import networkx as nx
from itertools import combinations

from .neighborhoods import neighborhood, closed_neighborhood
from .basics import connected
from .degree import minimum_degree

__all__ = [
    "minimum_k_forcing_set",
    "k_forcing_number",
    "is_zero_forcing_set",
    "minimum_zero_forcing_set",
    "zero_forcing_number",
    "two_forcing_number",
    "is_total_zero_forcing_set",
    "minimum_total_zero_forcing_set",
    "total_zero_forcing_number",
    "minimum_connected_k_forcing_set",
    "minimum_connected_zero_forcing_set",
    "connected_zero_forcing_number",
    "minimum_psd_zero_forcing_set",
    "positive_semidefinite_zero_forcing_number",
    "minimum_k_power_dominating_set",
    "k_power_domination_number",
    "minimum_power_dominating_set",
    "power_domination_number",
]


def is_k_forcing_vertex(G, v, nodes, k):
    """Return whether or not *v* can *k*-force relative to the set of nodes
    in `nodes`.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    v : node
        A single node in *G*.

    nodes : list, set
        An iterable container of nodes in G.

    k : int
        A positive integer.

    Returns
    -------
    boolean
        True if *v* can *k*-force relative to the nodes in `nodes`. False
        otherwise.
    """
    # check that k is a positive integer
    if not float(k).is_integer():
        raise TypeError("Expected k to be an integer.")
    k = int(k)
    if k < 1:
        raise ValueError("Expected k to be a positive integer.")
    S = set(n for n in nodes if n in G)
    n = len(neighborhood(G, v).difference(S))
    return v in S and n >= 1 and n <= k


def is_k_forcing_active_set(G, nodes, k):
    """Return whether or not at least one node in `nodes` can *k*-force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nbunch :
        A single node or iterable container or nodes.

    k : int
        A positive integer.

    Returns
    -------
    boolean
        True if at least one of the nodes in `nodes` can *k*-force. False
        otherwise.
    """
    S = set(n for n in nodes if n in G)
    for v in S:
        if is_k_forcing_vertex(G, v, S, k):
            return True
    return False


def is_k_forcing_set(G, nodes, k):
    """Return whether or not the nodes in `nodes` comprise a *k*-forcing set in
    *G*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, set
        An iterable container of nodes in G.

    k : int
        A positive integer.

    Returns
    -------
    boolean
        True if the nodes in `nodes` comprise a *k*-forcing set in *G*. False
        otherwise.
    """
    Z = set(n for n in nodes if n in G)
    while is_k_forcing_active_set(G, Z, k):
        Z_temp = Z.copy()
        for v in Z:
            if is_k_forcing_vertex(G, v, Z, k):
                Z_temp |= neighborhood(G, v)
        Z = Z_temp
    return Z == set(G.nodes())


def minimum_k_forcing_set(G, k):
    """Return a smallest *k*-forcing set in *G*.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    k : int
        A positive integer.

    Returns
    -------
    list
        A list of nodes in a smallest *k*-forcing set in *G*.
    """
    # use naive lower bound to compute a starting point for the search range
    rangeMin = minimum_degree(G) if k == 1 else 1
    # loop through subsets of nodes of G in increasing order of size until a zero forcing set is found
    for i in range(rangeMin, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_k_forcing_set(G, S, k):
                return set(S)


def k_forcing_number(G, k):
    """Return the *k*-forcing number of *G*.

    The *k*-forcing number of a graph is the cardinality of a smallest
    *k*-forcing set in the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    k : int
        A positive integer.

    Returns
    -------
    int
        The *k*-forcing number of *G*.
    """
    return len(minimum_k_forcing_set(G, k))


def is_zero_forcing_vertex(G, v, nbunch):
    """Return whether or not *v* can force relative to the set of nodes
    in nbunch.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    v : node
        A single node in *G*.

    nbunch :
        A single node or iterable container or nodes.

    Returns
    -------
    boolean
        True if *v* can force relative to the nodes in nbunch. False
        otherwise.
    """
    return is_k_forcing_vertex(G, v, nbunch, 1)


def is_zero_forcing_active_set(G, nbunch):
    """Return whether or not at least one node in nbunch can force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nbunch :
        A single node or iterable container or nodes.

    Returns
    -------
    boolean
        True if at least one of the nodes in nbunch can force. False
        otherwise.
    """
    return is_k_forcing_active_set(G, nbunch, 1)


def is_zero_forcing_set(G, nbunch):
    """Return whether or not the nodes in nbunch comprise a zero forcing set in
    *G*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nbunch :
        A single node or iterable container or nodes.

    Returns
    -------
    boolean
        True if the nodes in nbunch comprise a zero forcing set in *G*. False
        otherwise.
    """
    return is_k_forcing_set(G, nbunch, 1)


def minimum_zero_forcing_set(G):
    """Return a smallest zero forcing set in *G*.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    list
        A list of nodes in a smallest zero forcing set in *G*.
    """
    return minimum_k_forcing_set(G, 1)


def zero_forcing_number(G):
    """Return the zero forcing number of *G*.

    The zero forcing number of a graph is the cardinality of a smallest
    zero forcing set in the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The zero forcing number of *G*.
    """
    return len(minimum_zero_forcing_set(G))

def two_forcing_number(G):
    return k_forcing_number(G, 2)


def is_total_zero_forcing_set(G, nodes):
    """Return whether or not the nodes in `nodes` comprise a total zero forcing
    set in *G*.

    A *total zero forcing set* in a graph *G* is a zero forcing set that does
    not induce any isolated vertices.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, set
        An iterable container of nodes in G.

    Returns
    -------
    boolean
        True if the nodes in `nodes` comprise a total zero forcing set in *G*.
        False otherwise.
    """
    S = set(n for n in nodes if n in G)
    for v in S:
        if set(neighborhood(G, v)).intersection(S) == set():
            return False
    return is_zero_forcing_set(G, S)


def minimum_total_zero_forcing_set(G):
    """Return a smallest total zero forcing set in *G*.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    list
        A list of nodes in a smallest zero forcing set in *G*.
    """
    for i in range(2, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_total_zero_forcing_set(G, S):
                return set(S)
    # if the above loop completes, return None (should not occur)
    return None


def total_zero_forcing_number(G):
    """Return the total zero forcing number of *G*.

    The *total zero forcing number* of a graph is the cardinality of a smallest
    total zero forcing set in the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The total zero forcing number of *G*.
    """
    Z = minimum_total_zero_forcing_set(G)
    if Z is None:
        return None
    else:
        return len(Z)


def is_connected_k_forcing_set(G, nodes, k):
    """Return whether or not the nodes in `nodes` comprise a connected k-forcing
    set in *G*.

    A *connected k-forcing set* in a graph *G* is a k-forcing set that induces
    a connected subgraph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, set
        An iterable container of nodes in G.

    k : int
        A positive integer.

    Returns
    -------
    boolean
        True if the nodes in nbunch comprise a connected k-forcing set in *G*.
        False otherwise.
    """
    # check that k is a positive integer
    if not float(k).is_integer():
        raise TypeError("Expected k to be an integer.")
    k = int(k)
    if k < 1:
        raise ValueError("Expected k to be a positive integer.")
    S = set(n for n in nodes if n in G)
    H = G.subgraph(S)
    return connected(H) and is_k_forcing_set(G, S, k)


def is_connected_zero_forcing_set(G, nodes):
    """Return whether or not the nodes in `nodes` comprise a connected zero
    forcing set in *G*.

    A *connected zero forcing set* in a graph *G* is a zero forcing set that
    induces a connected subgraph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, set
        An iterable container of nodes in G.

    Returns
    -------
    boolean
        True if the nodes in `nodes` comprise a connected zero forcing set in
        *G*. False otherwise.
    """
    return is_connected_k_forcing_set(G, nodes, 1)


def minimum_connected_k_forcing_set(G, k):
    """Return a smallest connected k-forcing set in *G*.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    k : int
        A positive integer

    Returns
    -------
    list
        A list of nodes in a smallest connected k-forcing set in *G*.
    """
    # check that k is a positive integer
    if not float(k).is_integer():
        raise TypeError("Expected k to be an integer.")
    k = int(k)
    if k < 1:
        raise ValueError("Expected k to be a positive integer.")
    # only start search if graph is connected
    if not connected(G):
        return None
    for i in range(1, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_connected_k_forcing_set(G, S, k):
                return set(S)


def minimum_connected_zero_forcing_set(G):
    """Return a smallest connected zero forcing set in *G*.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    k : int
        A positive integer

    Returns
    -------
    list
        A list of nodes in a smallest connected zero forcing set in *G*.
    """
    return minimum_connected_k_forcing_set(G, 1)


def connected_k_forcing_number(G, k):
    """Return the connected k-forcing number of *G*.

    The *connected k-forcing number* of a graph is the cardinality of a smallest
    connected k-forcing set in the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The connected k-forcing number of *G*.
    """
    # check that k is a positive integer
    if not float(k).is_integer():
        raise TypeError("Expected k to be an integer.")
    k = int(k)
    if k < 1:
        raise ValueError("Expected k to be a positive integer.")
    Z = minimum_connected_k_forcing_set(G, k)
    if Z is None:
        return None
    else:
        return len(Z)


def connected_zero_forcing_number(G):
    """Return the connected zero forcing number of *G*.

    The *connected zero forcing number* of a graph is the cardinality of a
    smallest connected zero forcing set in the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The connected k-forcing number of *G*.
    """
    return connected_k_forcing_number(G, 1)

def is_psd_forcing_vertex(G, v, black_set, component):
    """
    Return whether or not v can force any white vertex in the component.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    v : node
        A single node in G.

    black_set : set
        A set of black vertices in G.

    component : set
        A set of vertices in a component of G - black_set.

    Returns
    -------
    tuple
        A tuple (True, w) if v can force the white vertex w in the component, (False, None) otherwise.
    """
    set_neighbors = set(neighborhood(G, v))
    white_neighbors_in_component = set_neighbors.intersection(component)

    if len(white_neighbors_in_component) == 1:
        w = white_neighbors_in_component.pop()
        return (True, w)
    return (False, None)


def psd_color_change(G, black_set):
    """
    Apply the PSD color change rule repeatedly until no more vertices can be forced.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    black_set : set
        A set of initial black vertices.

    Returns
    -------
    set
        The derived set of black vertices.
    """
    black_set = set(black_set)
    white_set = set(G.nodes()) - black_set

    while True:
        new_black = set()
        components = [set(c) for c in nx.connected_components(G.subgraph(white_set))]

        for component in components:
            for v in black_set:
                can_force, w = is_psd_forcing_vertex(G, v, black_set, component)
                if can_force:
                    new_black.add(w)

        if not new_black:
            break

        black_set.update(new_black)
        white_set -= new_black

    return black_set


def is_psd_zero_forcing_set(G, black_set):
    """
    Return whether or not the vertices in black_set comprise a PSD zero forcing set in G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    black_set : set
        A set of initial black vertices.

    Returns
    -------
    boolean
        True if the black_set is a PSD zero forcing set in G. False otherwise.
    """
    derived_set = psd_color_change(G, black_set)
    return len(derived_set) == G.order()


def minimum_psd_zero_forcing_set(G):
    """
    Return a smallest PSD zero forcing set in G.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    list
        A list of nodes in a smallest PSD zero forcing set in G.
    """
    for i in range(1, G.order() + 1):
        for black_set in combinations(G.nodes(), i):
            if is_psd_zero_forcing_set(G, black_set):
                return list(black_set)


def positive_semidefinite_zero_forcing_number(G):
    """
    Return the PSD zero forcing number of G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The PSD zero forcing number of G.
    """
    return len(minimum_psd_zero_forcing_set(G))


def is_k_power_dominating_set(G, nodes, k):
    """Return whether or not the nodes in `nodes` comprise a k-power dominating
    set.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, set
        An iterable container of nodes in G.

    k : int
        A positive integer.

    Returns
    -------
    boolean
        True if the nodes in `nodes` comprise a k-power dominating set, False
        otherwise.
    """
    return is_k_forcing_set(G, closed_neighborhood(G, nodes), k)


def minimum_k_power_dominating_set(G, k):
    """Return a smallest k-power dominating set of nodes in *G*.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    list
        A list of nodes in a smallest k-power dominating set in *G*.
    """
    for i in range(1, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_k_power_dominating_set(G, S, k):
                return set(S)


def k_power_domination_number(G, k):
    """Return the k-power domination number of *G*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The k-power domination number of *G*.
    """
    return len(minimum_k_power_dominating_set(G, k))


def is_power_dominating_set(G, nodes):
    """Return whether or not the nodes in `nodes` comprise a power dominating
    set.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, set
        An iterable container of nodes in G.

    Returns
    -------
    boolean
        True if the nodes in `nodes` comprise a power dominating set, False
        otherwise.
    """
    return is_k_power_dominating_set(G, nodes, 1)


def minimum_power_dominating_set(G):
    """Return a smallest power dominating set of nodes in *G*.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    list
        A list of nodes in a smallest power dominating set in *G*.
    """
    return minimum_k_power_dominating_set(G, 1)


def power_domination_number(G):
    """Return the power domination number of *G*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The power domination number of *G*.
    """
    return k_power_domination_number(G, 1)
