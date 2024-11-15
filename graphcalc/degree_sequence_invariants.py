import networkx as nx

from .degree import degree, degree_sequence
from .basics import size

__all__ = [
    "sub_k_domination_number",
    "slater",
    "sub_total_domination_number",
    "annihilation_number",
    "residue",
    "harmonic_index",
]


def sub_k_domination_number(G, k):
    r"""Return the sub-k-domination number of the graph.

    The *sub-k-domination number* of a graph G with *n* nodes is defined as the
    smallest positive integer t such that the following relation holds:

    .. math::
        t + \frac{1}{k}\sum_{i=0}^t d_i \geq n

    where

    .. math::
        {d_1 \geq d_2 \geq \cdots \geq d_n}

    is the degree sequence of the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    k : int
        A positive integer.

    Returns
    -------
    int
        The sub-k-domination number of a graph.

    See Also
    --------
    slater

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.cycle_graph(4)
    >>> gc.sub_k_domination_number(G, 1)
    True

    References
    ----------
    D. Amos, J. Asplund, B. Brimkov and R. Davila, The sub-k-domination number
    of a graph with applications to k-domination, *arXiv preprint
    arXiv:1611.02379*, (2016)
    """
    # check that k is a positive integer
    if not float(k).is_integer():
        raise TypeError("Expected k to be an integer.")
    k = int(k)
    if k < 1:
        raise ValueError("Expected k to be a positive integer.")
    D = degree_sequence(G)
    D.sort(reverse=True)
    n = len(D)
    for i in range(n + 1):
        if i + (sum(D[:i]) / k) >= n:
            return i
    # if above loop completes, return None
    return None


def slater(G):
    r"""
    Returns the Slater invariant for the graph.

    The Slater invariant of a graph G is a lower bound for the domination
    number of a graph defined by:

    .. math::
        sl(G) = \min\{t : t + \sum_{i=0}^t d_i \geq n\}

    where

    .. math::
        {d_1 \geq d_2 \geq \cdots \geq d_n}

    is the degree sequence of the graph ordered in non-increasing order and *n*
    is the order of G.

    Amos et al. rediscovered this invariant and generalized it into what is
    now known as the sub-*k*-domination number.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The Slater invariant for the graph.

    See Also
    --------
    sub_k_domination_number : A generalized version of the Slater invariant.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.cycle_graph(4)  # A 4-cycle
    >>> gc.slater(G)
    2

    >>> H = nx.path_graph(5)  # A path graph with 5 vertices
    >>> gc.slater(H)
    2

    >>> K = nx.complete_graph(5)  # A complete graph with 5 vertices
    >>> gc.slater(K)
    1

    References
    ----------
    D. Amos, J. Asplund, B. Brimkov, and R. Davila, The sub-k-domination number
    of a graph with applications to k-domination, *arXiv preprint
    arXiv:1611.02379*, (2016)

    P.J. Slater, Locating dominating sets and locating-dominating set, *Graph
    Theory, Combinatorics and Applications: Proceedings of the 7th Quadrennial
    International Conference on the Theory and Applications of Graphs*,
    2: 2073-1079 (1995)
    """
    return sub_k_domination_number(G, 1)


def sub_total_domination_number(G):
    r"""
    Returns the sub-total domination number of the graph.

    The sub-total domination number is defined as:

    .. math::
        sub_{t}(G) = \min\{t : \sum_{i=0}^t d_i \geq n\}

    where

    .. math::
        d_1 \geq d_2 \geq \cdots \geq d_n

    is the degree sequence of the graph ordered in non-increasing order, and
    *n* is the order of the graph (number of vertices).

    This invariant was defined and investigated by Randy Davila.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The sub-total domination number of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.cycle_graph(6)  # A cycle graph with 6 vertices
    >>> gc.sub_total_domination_number(G)
    3

    >>> H = nx.path_graph(4)  # A path graph with 4 vertices
    >>> gc.sub_total_domination_number(H)
    2

    >>> K = nx.complete_graph(5)  # A complete graph with 5 vertices
    >>> gc.sub_total_domination_number(K)
    1

    References
    ----------
    R. Davila, A note on sub-total domination in graphs. *arXiv preprint
    arXiv:1701.07811*, (2017)
    """
    D = degree_sequence(G)
    D.sort(reverse=True)
    n = len(D)
    for i in range(n + 1):
        if sum(D[:i]) >= n:
            return i
    # if above loop completes, return None
    return None


def annihilation_number(G):
    r"""
    Returns the annihilation number of the graph.

    The annihilation number of a graph *G* is defined as:

    .. math::
        a(G) = \max\{t : \sum_{i=1}^t d_i \leq m \}

    where

    .. math::
        d_1 \leq d_2 \leq \cdots \leq d_n

    is the degree sequence of the graph ordered in non-decreasing order, and
    *m* is the number of edges in *G*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The annihilation number of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.cycle_graph(6)  # A cycle graph with 6 vertices
    >>> gc.annihilation_number(G)
    3

    >>> H = nx.path_graph(5)  # A path graph with 5 vertices
    >>> gc.annihilation_number(H)
    2

    >>> K = nx.complete_graph(5)  # A complete graph with 5 vertices
    >>> gc.annihilation_number(K)
    1

    References
    ----------
    - P. Dankelmann, S. Mukwembi, and H.C. Swart, The annihilation number of a
      graph, *Utilitas Mathematica*, 72:91–108, (2007).
    """
    D = degree_sequence(G)
    D.sort()  # sort in non-decreasing order
    n = len(D)
    m = size(G)
    # sum over degrees in the sequence until the sum is larger than the number of edges in the graph
    for i in reversed(range(n + 1)):
        if sum(D[:i]) <= m:
            return i

def residue(G):
    r"""
    Returns the residue of a graph.

    The residue of a graph is defined as the number of zeros obtained at the
    end of the Havel-Hakimi process. This process determines whether a given
    degree sequence corresponds to a graphical sequence, which is a sequence
    of integers that can be realized as the degree sequence of a simple graph.

    **Havel-Hakimi Algorithm**:
    - Sort the degree sequence in non-increasing order.
    - Remove the largest degree (say, `d`) from the sequence.
    - Reduce the next `d` degrees by 1.
    - Repeat until all degrees are zero (graphical) or a negative degree is encountered (non-graphical).

    The residue is the count of zeros in the sequence when the algorithm terminates.

    Parameters
    ----------
    G : nx.Graph
        The graph whose residue is to be calculated.

    Returns
    -------
    int
        The residue of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)  # Path graph with 4 vertices
    >>> gc.residue(G)
    4

    >>> H = nx.complete_graph(4)  # Complete graph with 4 vertices
    >>> gc.residue(H)
    0

    >>> K = nx.cycle_graph(5)  # Cycle graph with 5 vertices
    >>> gc.residue(K)
    5

    References
    ----------
    Havel, Václav, and Hakimi, Seymour L. "A Method of Constructing Graphs from
    their Degree Sequence." *Journal of Mathematical Analysis and Applications*,
    1963.

    Notes
    -----
    The Havel-Hakimi process ensures the degree sequence remains graphical
    throughout the steps, making it a key concept in graph theory.
    """
    degrees = degree_sequence(G)
    degrees.sort(reverse=True)
    while degrees[0] > 0:
        max_degree = degrees.pop(0)
        for i in range(max_degree):
            degrees[i] -= 1
        degrees.sort(reverse=True)

    return len(degrees)


def harmonic_index(G):
    r"""
    Returns the harmonic index of a graph.

    The harmonic index of a graph is defined as:

    .. math::
        H(G) = \sum_{uv \in E(G)} \frac{2}{d(u) + d(v)}

    where:
    - :math:`E(G)` is the edge set of the graph :math:`G`.
    - :math:`d(u)` is the degree of vertex :math:`u`.

    The harmonic index is commonly used in mathematical chemistry and network science
    to measure structural properties of molecular and network graphs.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    float
        The harmonic index of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc

    >>> G = nx.path_graph(4)  # Path graph with 4 vertices
    >>> gc.harmonic_index(G)
    1.8333333333333333

    >>> H = nx.complete_graph(3)  # Complete graph with 3 vertices
    >>> gc.harmonic_index(H)
    2.0

    Notes
    -----
    - The harmonic index assumes all edge weights are equal to 1. If you want
      to consider weighted graphs, modify the function to account for edge weights.
    - The harmonic index is symmetric with respect to the graph's structure, making
      it invariant under graph isomorphism.

    References
    ----------
    S. Klavžar and I. Gutman, A comparison of the Schultz molecular topological
    index and the Wiener index. *Journal of Chemical Information and Computer Sciences*,
    33(6), 1006-1009 (1993).
    """
    return 2*sum((1/(degree(G, v) + degree(G, u)) for u, v in G.edges()))
