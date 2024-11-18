import networkx as nx
from itertools import combinations

from .neighborhoods import neighborhood, closed_neighborhood, set_closed_neighbors
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
    r"""
    Determines whether a node *v* can perform *k*-forcing with respect to a set of nodes.

    A node *v* is said to *k*-force if it is in the set `nodes` and has between 1 and *k* neighbors
    not in `nodes`.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    v : node
        The node to check for *k*-forcing.
    nodes : list or set
        The set of nodes under consideration.
    k : int
        The parameter for *k*-forcing, which must be a positive integer.

    Returns
    -------
    bool
        True if *v* can *k*-force relative to `nodes`. False otherwise.

    Raises
    ------
    TypeError
        If *k* is not an integer.
    ValueError
        If *k* is not a positive integer.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0, 1}
    >>> print(gc.is_k_forcing_vertex(G, 0, nodes, 1))
    True
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
    r"""
    Checks if at least one node in the given set can perform *k*-forcing.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        The set of nodes under consideration.
    k : int
        The parameter for *k*-forcing.

    Returns
    -------
    bool
        True if at least one node in `nodes` can *k*-force. False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0, 1}
    >>> print(gc.is_k_forcing_active_set(G, nodes, 1))
    True
    """
    S = set(n for n in nodes if n in G)
    for v in S:
        if is_k_forcing_vertex(G, v, S, k):
            return True
    return False


def is_k_forcing_set(G, nodes, k):
    r"""
    Determines whether the given set of nodes is a *k*-forcing set in the graph.

    A set of nodes is a *k*-forcing set if, starting from the set, all nodes in the graph
    can eventually be included by repeatedly applying the *k*-forcing rule.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        The set of nodes under consideration.
    k : int
        The parameter for *k*-forcing.

    Returns
    -------
    bool
        True if the nodes form a *k*-forcing set. False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0, 2}
    >>> print(gc.is_k_forcing_set(G, nodes, 1))
    True
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
    r"""
    Finds a smallest *k*-forcing set in the graph using brute force.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    k : int
        The parameter for *k*-forcing.

    Returns
    -------
    set
        A smallest *k*-forcing set.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.minimum_k_forcing_set(G, 1))
    {0}
    """
    # use naive lower bound to compute a starting point for the search range
    rangeMin = minimum_degree(G) if k == 1 else 1
    # loop through subsets of nodes of G in increasing order of size until a zero forcing set is found
    for i in range(rangeMin, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_k_forcing_set(G, S, k):
                return set(S)


def k_forcing_number(G, k):
    r"""
    Calculates the *k*-forcing number of the graph.

    The *k*-forcing number is the size of the smallest *k*-forcing set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    k : int
        The parameter for *k*-forcing.

    Returns
    -------
    int
        The *k*-forcing number.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.k_forcing_number(G, 1))
    1
    """
    return len(minimum_k_forcing_set(G, k))


def is_zero_forcing_vertex(G, v, S):
    r"""
    Determines whether a node *v* can force relative to a set of nodes.

    This is a special case of *k*-forcing where *k = 1*.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    v : node
        The node to check.
    S : list or set
        The set of nodes under consideration.

    Returns
    -------
    bool
        True if *v* can force. False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> S = {0, 1}
    >>> print(gc.is_zero_forcing_vertex(G, 0, S))
    True
    """
    return is_k_forcing_vertex(G, v, S, 1)


def is_zero_forcing_active_set(G, S):
    r"""
    Checks whether the given set of nodes forms a zero forcing set in the graph.

    A zero forcing set is a special case of a *k*-forcing set where *k = 1*.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    S : list or set
        The set of nodes under consideration.

    Returns
    -------
    bool
        True if the nodes form a zero forcing set. False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> S = {0, 3}
    >>> print(gc.is_zero_forcing_set(G, S))
    True
    """
    return is_k_forcing_active_set(G, S, 1)


def is_zero_forcing_set(G, S):
    r"""
    Checks whether the given set of nodes forms a zero forcing set in the graph.

    A zero forcing set is a special case of a *k*-forcing set where *k = 1*.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    S : list or set
        The set of nodes under consideration.

    Returns
    -------
    bool
        True if the nodes form a zero forcing set. False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> S = {0, 3}
    >>> print(gc.is_zero_forcing_set(G, S))
    True
    """
    return is_k_forcing_set(G, S, 1)


def minimum_zero_forcing_set(G):
    r"""
    Finds a smallest zero forcing set in the graph using brute force.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    set
        A smallest zero forcing set.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.minimum_zero_forcing_set(G, 1))
    {0}
    """
    return minimum_k_forcing_set(G, 1)


def zero_forcing_number(G):
    r"""
    Calculates the zero forcing number of the graph.

    The zero forcing number is the size of the smallest zero forcing set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    int
        The zero forcing number of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.zero_forcing_number(G))
    1
    """
    return len(minimum_zero_forcing_set(G))

def two_forcing_number(G):
    r"""
    Calculates the 2-forcing number of the graph.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    int
        The 2-forcing number of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.two_forcing_number(G))
    1
    """
    return k_forcing_number(G, 2)


def is_total_zero_forcing_set(G, nodes):
    r"""
    Checks if the given nodes form a total zero forcing set.

    A total zero forcing set is a zero forcing set that does not induce any isolated vertices.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        The set of nodes under consideration.

    Returns
    -------
    bool
        True if the nodes form a total zero forcing set. False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0, 1}
    >>> print(gc.is_total_zero_forcing_set(G, nodes))
    True
    """
    S = set(n for n in nodes if n in G)
    for v in S:
        if set(neighborhood(G, v)).intersection(S) == set():
            return False
    return is_zero_forcing_set(G, S)


def minimum_total_zero_forcing_set(G):
    r"""
    Finds a smallest total zero forcing set in the graph G.

    A total zero forcing set is a zero forcing set that does not induce any isolated vertices.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    set
        A smallest total zero forcing set in G, or None if none exists.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> result = gc.minimum_total_zero_forcing_set(G)
    >>> print(result)
    {0, 3}
    """
    for i in range(2, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_total_zero_forcing_set(G, S):
                return set(S)
    # if the above loop completes, return None (should not occur)
    return None


def total_zero_forcing_number(G):
    r"""
    Calculates the total zero forcing number of the graph G.

    The total zero forcing number is the size of the smallest total zero forcing set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    int
        The total zero forcing number of G, or None if no such set exists.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.total_zero_forcing_number(G))
    2
    """
    Z = minimum_total_zero_forcing_set(G)
    if Z is None:
        return None
    else:
        return len(Z)


def is_connected_k_forcing_set(G, nodes, k):
    r"""
    Determines whether the given nodes form a connected k-forcing set in the graph G.

    A connected k-forcing set is a k-forcing set that induces a connected subgraph.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        A set of nodes under consideration.
    k : int
        A positive integer representing the k-forcing parameter.

    Returns
    -------
    bool
        True if the nodes form a connected k-forcing set, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0, 1}
    >>> print(gc.is_connected_k_forcing_set(G, nodes, 1))
    True
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
    r"""
    Determines whether the given nodes form a connected zero forcing set in the graph G.

    A connected zero forcing set is a zero forcing set that induces a connected subgraph.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        A set of nodes under consideration.

    Returns
    -------
    bool
        True if the nodes form a connected k-forcing set, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0, 1}
    >>> print(gc.is_connected_zero_forcing_set(G, nodes))
    True
    """
    return is_connected_k_forcing_set(G, nodes, 1)


def minimum_connected_k_forcing_set(G, k):
    r"""
    Finds the smallest connected k-forcing set in the graph G using brute force.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    k : int
        A positive integer representing the k-forcing parameter.

    Returns
    -------
    set
        A smallest connected k-forcing set in G, or None if the graph is disconnected.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.minimum_connected_k_forcing_set(G, 1))
    {0}
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
    r"""
    Finds the smallest connected zero forcing set in the graph G using brute force.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    set
        A smallest connected zero forcing set in G, or None if the graph is disconnected.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.minimum_connected_zero_forcing_set(G))
    {0}
    """
    return minimum_connected_k_forcing_set(G, 1)


def connected_k_forcing_number(G, k):
    r"""
    Calculates the connected kforcing number of the graph G.

    The connected zero forcing number is the size of the smallest connected zero forcing set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    k : int
        A positive integer representing the k-forcing parameter.

    Returns
    -------
    int
        The connected k-forcing number of G, or None if no such set exists.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.connected_zero_forcing_number(G, 1))
    1
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
    r"""
    Calculates the connected zero forcing number of the graph G.

    The connected zero forcing number is the size of the smallest connected zero forcing set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    int
        The connected zero forcing number of G, or None if no such set exists.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.connected_zero_forcing_number(G))
    1
    """
    return connected_k_forcing_number(G, 1)

def is_psd_forcing_vertex(G, v, black_set, component):
    r"""
    Determines whether a node *v* can perform positive semidefinite (PSD) forcing in a specific component.

    A node *v* in the black set can force a single white vertex in a connected component
    of G - black_set if it has exactly one white neighbor in that component.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    v : node
        A single node in G.
    black_set : set
        A set of nodes considered as "black" in the forcing process.
    component : set
        A set of nodes representing a connected component of G - black_set.

    Returns
    -------
    tuple
        (True, w) if *v* can force the white vertex *w* in the component,
        (False, None) otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> black_set = {0, 1}
    >>> component = {2, 3}
    >>> print(gc.is_psd_forcing_vertex(G, 1, black_set, component))
    (True, 2)
    """
    set_neighbors = set(neighborhood(G, v))
    white_neighbors_in_component = set_neighbors.intersection(component)

    if len(white_neighbors_in_component) == 1:
        w = white_neighbors_in_component.pop()
        return (True, w)
    return (False, None)


def psd_color_change(G, black_set):
    r"""
    Applies the Positive Semidefinite (PSD) color change rule to a graph G.

    The PSD color change rule allows a black vertex *v* to force a white vertex *w*
    in a connected component of G - black_set if *v* has exactly one white neighbor
    in that component. This process is applied iteratively until no more vertices can
    be forced.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    black_set : set
        A set of initial black vertices.

    Returns
    -------
    set
        The derived set of black vertices after applying the PSD color change rule.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(5)
    >>> black_set = {0}
    >>> result = gc.psd_color_change(G, black_set)
    >>> print(result)
    {0, 1}
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
    r"""
    Determines whether the given set of black vertices is a PSD zero forcing set.

    A PSD zero forcing set is a set of vertices that, through iterative application
    of the PSD color change rule, results in all vertices of the graph being black.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    black_set : set
        A set of initial black vertices.

    Returns
    -------
    bool
        True if the given set is a PSD zero forcing set, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> black_set = {0, 3}
    >>> print(gc.is_psd_zero_forcing_set(G, black_set))
    True
    """
    derived_set = psd_color_change(G, black_set)
    return len(derived_set) == G.order()


def minimum_psd_zero_forcing_set(G):
    r"""
    Finds a smallest PSD zero forcing set in the graph G.

    The PSD zero forcing set is computed using brute force by iterating through
    all possible subsets of vertices until a valid set is found.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    list
        A list of nodes representing the smallest PSD zero forcing set in G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> result = gc.minimum_psd_zero_forcing_set(G)
    >>> print(result)
    [0, 3]
    """
    for i in range(1, G.order() + 1):
        for black_set in combinations(G.nodes(), i):
            if is_psd_zero_forcing_set(G, black_set):
                return set(black_set)


def positive_semidefinite_zero_forcing_number(G):
    r"""
    Calculates the Positive Semidefinite (PSD) zero forcing number of the graph G.

    The PSD zero forcing number is the size of the smallest PSD zero forcing set
    in the graph.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    int
        The PSD zero forcing number of G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.positive_semidefinite_zero_forcing_number(G))
    1
    """
    return len(minimum_psd_zero_forcing_set(G))


def is_k_power_dominating_set(G, nodes, k):
    r"""
    Checks if the given nodes comprise a k-power dominating set in the graph G.

    A k-power dominating set is a subset of nodes such that all nodes in the
    graph can be dominated through the k-forcing process. The k-forcing process
    begins with the closed neighborhood of the given nodes, and iteratively
    propagates domination by ensuring each dominated node dominates at least k
    additional nodes in its neighborhood.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        An iterable container of nodes in G to test as a k-power dominating set.
    k : int
        A positive integer representing the power domination threshold.

    Returns
    -------
    bool
        True if the nodes form a k-power dominating set, otherwise False.

    Notes
    -----
    - The closed neighborhood of a node v in a graph G includes v itself and
      all its neighbors.
    - A k-power dominating set is a generalization of the concept of domination
      in graphs.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(5)
    >>> nodes = {0}
    >>> print(gc.is_k_power_dominating_set(G, nodes, 2))
    True

    >>> nodes = {4}
    >>> print(gc.is_k_power_dominating_set(G, nodes, 3))
    False
    """
    return is_k_forcing_set(G, set_closed_neighbors(G, nodes), k)


def minimum_k_power_dominating_set(G, k):
    r"""
    Checks if the given nodes comprise a k-power dominating set in the graph G.

    A k-power dominating set is a subset of vertices such that all vertices
    of the graph can be dominated through the k-forcing process starting
    from the closed neighborhood of the given nodes.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        An iterable container of nodes in G.
    k : int
        A positive integer representing the power domination threshold.

    Returns
    -------
    bool
        True if the given nodes form a k-power dominating set, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0}
    >>> print(gc.is_k_power_dominating_set(G, nodes, 2))
    True
    """
    for i in range(1, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_k_power_dominating_set(G, S, k):
                return set(S)


def k_power_domination_number(G, k):
    r"""
    Finds the smallest k-power dominating set in the graph G.

    This function uses a brute-force approach to identify the minimum subset
    of vertices that form a k-power dominating set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    k : int
        A positive integer representing the power domination threshold.

    Returns
    -------
    set
        A set of nodes representing the smallest k-power dominating set.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> result = gc.minimum_k_power_dominating_set(G, 2)
    >>> print(result)
    {0}
    """
    for i in range(1, G.order() + 1):
        for S in combinations(G.nodes(), i):
            if is_k_power_dominating_set(G, S, k):
                return i


def is_power_dominating_set(G, nodes):
    r"""
    Checks if the given nodes form a power dominating set in the graph G.

    A power dominating set is a 1-power dominating set, meaning the domination
    process follows the 1-forcing rule starting from the closed neighborhood of the nodes.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.
    nodes : list or set
        An iterable container of nodes in G.

    Returns
    -------
    bool
        True if the nodes form a power dominating set, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> nodes = {0}
    >>> print(gc.is_power_dominating_set(G, nodes))
    True
    """
    return is_k_power_dominating_set(G, nodes, 1)


def minimum_power_dominating_set(G):
    r"""
    Finds the smallest power dominating set in the graph G.

    This function uses a brute-force approach to identify the minimum subset
    of vertices that form a power dominating set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    set
        A set of nodes representing the smallest power dominating set.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> result = gc.minimum_power_dominating_set(G)
    >>> print(result)
    {0}
    """
    return minimum_k_power_dominating_set(G, 1)


def power_domination_number(G):
    r"""
    Calculates the power domination number of the graph G.

    The power domination number is the size of the smallest power dominating set.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph.

    Returns
    -------
    int
        The power domination number of G.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)
    >>> print(gc.power_domination_number(G))
    1
    """
    return k_power_domination_number(G, 1)
