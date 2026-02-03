
from typing import Hashable, List
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import graphcalc as gc
from graphcalc import SimpleGraph
from graphcalc.utils import enforce_type, GraphLike


__all__= [
    'degree',
    'degree_sequence',
    'average_degree',
    'maximum_degree',
    'minimum_degree',
    "sub_k_domination_number",
    "slater",
    "sub_total_domination_number",
    "annihilation_number",
    "residue_from_degrees",
    "k_residue_from_degrees",
    "residue",
    "k_residue",
    "harmonic_index",
]

@enforce_type(0, (nx.Graph, SimpleGraph))
def degree(G: GraphLike, v: Hashable) -> int:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.degree(G, 1)
    2
    >>> gc.degree(G, 0)
    1
    """
    return G.degree(v)

@enforce_type(0, (nx.Graph, SimpleGraph))
def degree_sequence(G: GraphLike, nonincreasing=True) -> List[int]:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.degree_sequence(G)
    [2, 2, 1, 1]
    >>> gc.degree_sequence(G, nonincreasing=False)
    [1, 1, 2, 2]
    """
    degrees = [degree(G, v) for v in G.nodes]
    if nonincreasing:
        degrees.sort(reverse=True)
    else:
        degrees.sort()
    return degrees

@enforce_type(0, (nx.Graph, SimpleGraph))
def average_degree(G: GraphLike) -> float:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.average_degree(G)
    1.5
    """
    degrees = degree_sequence(G)
    return sum(degrees) / len(degrees)

@enforce_type(0, (nx.Graph, SimpleGraph))
def maximum_degree(G: GraphLike) -> int:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.maximum_degree(G)
    2
    """
    degrees = degree_sequence(G)
    return max(degrees)

@enforce_type(0, (nx.Graph, SimpleGraph))
def minimum_degree(G: GraphLike) -> int:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.minimum_degree(G)
    1
    """
    degrees = degree_sequence(G)
    return min(degrees)

@enforce_type(0, (nx.Graph, SimpleGraph))
def sub_k_domination_number(G: GraphLike, k: int) -> float:
    r"""Return the sub-k-domination number of the graph.

    The *sub-k-domination number* of a graph :math:`G` with *n* nodes is defined as the
    smallest positive integer :math:`t` such that the following relation holds:

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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> gc.sub_k_domination_number(G, 1)
    2

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

@enforce_type(0, (nx.Graph, SimpleGraph))
def slater(G: GraphLike) -> int:
    r"""
    Returns the Slater invariant for the graph.

    The Slater invariant of a graph :math:`G` is a lower bound for the domination
    number of a graph defined by:

    .. math::
        sl(G) = \min\{t : t + \sum_{i=0}^t d_i \geq n\}

    where

    .. math::
        {d_1 \geq d_2 \geq \cdots \geq d_n}

    is the degree sequence of the graph ordered in non-increasing order and *n*
    is the order of :math:`G`.

    Amos et al. rediscovered this invariant and generalized it into what is
    now known as the sub-:math:`k`-domination number.

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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, cycle_graph, complete_graph

    >>> G = cycle_graph(4)  # A 4-cycle
    >>> gc.slater(G)
    2

    >>> H = path_graph(5)  # A path graph with 5 vertices
    >>> gc.slater(H)
    2

    >>> K = complete_graph(5)  # A complete graph with 5 vertices
    >>> gc.slater(K)
    1

    References
    ----------
    D. Amos, J. Asplund, B. Brimkov, and R. Davila, The sub-:math:`k`-domination number
    of a graph with applications to :math:`k`-domination, *arXiv preprint
    arXiv:1611.02379*, (2016)

    P.J. Slater, Locating dominating sets and locating-dominating set, *Graph
    Theory, Combinatorics and Applications: Proceedings of the 7th Quadrennial
    International Conference on the Theory and Applications of Graphs*,
    2: 2073-1079 (1995)
    """
    return sub_k_domination_number(G, 1)

@enforce_type(0, (nx.Graph, SimpleGraph))
def sub_total_domination_number(G: GraphLike) -> int:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, cycle_graph, complete_graph

    >>> G = cycle_graph(6)  # A cycle graph with 6 vertices
    >>> gc.sub_total_domination_number(G)
    3

    >>> H = path_graph(4)  # A path graph with 4 vertices
    >>> gc.sub_total_domination_number(H)
    2

    >>> K = complete_graph(5)  # A complete graph with 5 vertices
    >>> gc.sub_total_domination_number(K)
    2

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

@enforce_type(0, (nx.Graph, SimpleGraph))
def annihilation_number(G: GraphLike) -> int:
    r"""
    Returns the annihilation number of the graph.

    The annihilation number of a graph :math:`G` is defined as:

    .. math::
        a(G) = \max\{t : \sum_{i=1}^t d_i \leq m \}

    where

    .. math::
        d_1 \leq d_2 \leq \cdots \leq d_n

    is the degree sequence of the graph ordered in non-decreasing order, and
    :math:`m` is the number of edges in :math:`G`.

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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, cycle_graph, complete_graph

    >>> G = cycle_graph(6)  # A cycle graph with 6 vertices
    >>> gc.annihilation_number(G)
    3

    >>> H = path_graph(5)  # A path graph with 5 vertices
    >>> gc.annihilation_number(H)
    3

    >>> K = complete_graph(5)  # A complete graph with 5 vertices
    >>> gc.annihilation_number(K)
    2

    References
    ----------
    - P. Dankelmann, S. Mukwembi, and H.C. Swart, The annihilation number of a
      graph, *Utilitas Mathematica*, 72:91–108, (2007).
    """

    D = degree_sequence(G)
    D.sort()  # sort in non-decreasing order
    n = len(D)
    m = gc.size(G)
    # sum over degrees in the sequence until the sum is larger than the number of edges in the graph
    for i in reversed(range(n + 1)):
        if sum(D[:i]) <= m:
            return i


# If you have these in your package:
# from .simplegraph import SimpleGraph
# from .typing import GraphLike
# from .decorators import enforce_type
# import graphcalc as gc
 # replace with your union (nx.Graph | SimpleGraph)

# ──────────────────────────────────────────────────────────────────────────────
# Core: elimination sequence from a degree list
# ──────────────────────────────────────────────────────────────────────────────

def elimination_sequence_from_degrees(degrees: Sequence[int]) -> List[int]:
    """
    Compute the Havel–Hakimi elimination sequence E(D) from a degree sequence.

    The elimination sequence records *every* removed entry (including trailing 0's)
    as the HH process proceeds on a copy of the degree list.

    Parameters
    ----------
    degrees : Sequence[int]
        Nonnegative integer degree sequence (assumed graphical when coming from a graph).

    Returns
    -------
    List[int]
        The elimination sequence E(D), i.e., the list of popped values at each HH step.

    Notes
    -----
    - For a genuine graph degree sequence, no negativity or length violations occur.
    - We sort in non-increasing order at each step (standard HH).
    """
    seq = list(degrees)
    if not seq:
        return []

    # Ensure non-increasing
    seq.sort(reverse=True)
    E: List[int] = []

    while seq:
        d = seq.pop(0)
        E.append(d)

        if d < 0:
            raise ValueError("Negative entry encountered during Havel–Hakimi.")
        if d > len(seq):
            # This would signal non-graphical input if not from a graph.
            raise ValueError("Degree too large for remaining sequence in Havel–Hakimi.")

        # Decrement next d entries
        for i in range(d):
            seq[i] -= 1

        # Keep sorted for next iteration
        seq.sort(reverse=True)

    return E


# ──────────────────────────────────────────────────────────────────────────────
# k-residue from degrees
# ──────────────────────────────────────────────────────────────────────────────

def k_residue_from_degrees(degrees: Sequence[int], k: int) -> int:
    """
    Compute the k-residue R_k from a degree sequence via its elimination sequence.

    Definition
    ----------
    For elimination sequence E = E(D) and k >= 1,
        R_k(E) = sum_{i=0}^{k-1} (k - i) * f_i(E),
    where f_i(E) is the frequency of i in E.

    Parameters
    ----------
    degrees : Sequence[int]
        Nonnegative integer degree sequence (assumed graphical when from a graph).
    k : int
        Parameter k >= 1.

    Returns
    -------
    int
        The k-residue R_k(D).

    See Also
    --------
    elimination_sequence_from_degrees
    """
    if k < 1:
        raise ValueError("k must be an integer >= 1.")

    E = elimination_sequence_from_degrees(degrees)
    # Count only 0..k-1
    freq = [0] * k
    for x in E:
        if 0 <= x < k:
            freq[x] += 1
    return int(sum((k - i) * freq[i] for i in range(k)))


# ──────────────────────────────────────────────────────────────────────────────
# Graph wrappers that reuse the degree-sequence core
# ──────────────────────────────────────────────────────────────────────────────

# Optional: a degree-sequence version of your classical residue for symmetry
def residue_from_degrees(degrees: Sequence[int]) -> int:
    """
    Classical residue R(D): number of zeros in the elimination sequence E(D).

    Parameters
    ----------
    degrees : Sequence[int]
        Degree sequence.

    Returns
    -------
    int
        R(D) = f_0(E(D)).
    """
    E = elimination_sequence_from_degrees(degrees)
    return E.count(0)


@enforce_type(0, (nx.Graph, SimpleGraph))
def residue(G: GraphLike) -> int:
    r"""
    Returns the residue of a graph.

    The residue of a graph is defined as the number of zeros obtained at the
    end of the Havel-Hakimi process. This process determines whether a given
    degree sequence corresponds to a graphical sequence, which is a sequence
    of integers that can be realized as the degree sequence of a simple graph.

    **Havel-Hakimi Algorithm**:
    - Sort the degree sequence in non-increasing order.
    - Remove the largest degree (say, :math:`d`) from the sequence.
    - Reduce the next :math:`d` degrees by 1.
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, complete_graph, cycle_graph

    >>> G = path_graph(4)  # Path graph with 4 vertices
    >>> gc.residue(G)
    2

    >>> H = complete_graph(4)  # Complete graph with 4 vertices
    >>> gc.residue(H)
    1

    >>> K = cycle_graph(5)  # Cycle graph with 5 vertices
    >>> gc.residue(K)
    2

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
    degrees = gc.degree_sequence(G)
    degrees.sort(reverse=True)
    while degrees[0] > 0:
        max_degree = degrees.pop(0)
        for i in range(max_degree):
            degrees[i] -= 1
        degrees.sort(reverse=True)

    return len(degrees)

@enforce_type(0, (nx.Graph, SimpleGraph))
@enforce_type(1, int)
def k_residue(G: GraphLike, k: int) -> int:
    r"""
    Compute the k-residue R_k(G) from the Havel–Hakimi elimination sequence.

    Definition (Jelen generalizing Favaron–Mahéo–Saclé, Griggs–Kleitman, Triesch)
    ---------------------------------------------------------------------------
    Let D be a graphic degree sequence and E = E(D) its Havel–Hakimi elimination
    sequence (the list of values removed at each step, including trailing zeros).
    For k ≥ 1,
        R_k(E) = sum_{i=0}^{k-1} (k - i) * f_i(E),
    where f_i(E) is the frequency of i in E. Since E is determined by D(G), we
    write R_k(G).

    Special case: k = 1 gives R_1(G) = f_0(E) = R(G) (the usual residue).

    Parameters
    ----------
    G : nx.Graph or SimpleGraph
        Input graph.
    k : int
        Parameter k ≥ 1.

    Returns
    -------
    int
        The k-residue R_k(G).

    Notes
    -----
    We explicitly build the elimination sequence E by performing the Havel–Hakimi
    process and recording the removed value at each step, including zeros at the
    end. This ensures f_0(E) equals the classical residue.

    Examples
    --------
    >>> from graphcalc.generators import path_graph, complete_graph
    >>> G = path_graph(4)
    >>> k_residue(G, 1) == residue(G)  # should match your residue()
    True
    >>> H = complete_graph(4)
    >>> k_residue(H, 2)  # weighted count of 0s and 1s in E(H)
    3
    """
    degrees = gc.degree_sequence(G)  # or list(dict(G.degree()).values())
    return k_residue_from_degrees(degrees, k)

@enforce_type(0, (nx.Graph, SimpleGraph))
def harmonic_index(G: GraphLike) -> float:
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, complete_graph

    >>> G = path_graph(4)  # Path graph with 4 vertices
    >>> gc.harmonic_index(G)
    1.8333333333333333

    >>> H = complete_graph(3)  # Complete graph with 3 vertices
    >>> gc.harmonic_index(H)
    1.5

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

def irregularity(G):
    r"""
    Compute the **(Albertson) irregularity** of a graph :math:`G`.

    The (Albertson) irregularity is the edge-sum of absolute degree differences:

    .. math::

        \operatorname{irr}(G) \;=\; \sum_{uv \in E(G)} \bigl| \deg(u) - \deg(v) \bigr|,

    where :math:`\deg(u)` denotes the (undirected) degree of vertex :math:`u`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. For standard usage in graph invariants, :math:`G` is typically a
        simple undirected graph. Degrees are read from ``G.degree()`` and the sum ranges over
        the edges returned by ``G.edges()``.

        - If ``G`` is a MultiGraph, parallel edges are iterated with multiplicity and degrees
          count multiplicity, so this computes the natural multigraph extension.
        - If ``G`` is directed, ``G.degree()`` is the total degree (in-degree + out-degree),
          so the result is the Albertson irregularity with respect to total degree unless you
          replace it by ``G.in_degree()`` or ``G.out_degree()`` by convention.

    Returns
    -------
    int
        The value :math:`\operatorname{irr}(G)`. In particular, if :math:`G` has no edges,
        the sum is empty and the function returns 0.

    Notes
    -----
    - This invariant is commonly attributed to **Albertson** and is often called the
      *Albertson irregularity*.
    - :math:`\operatorname{irr}(G)=0` if and only if :math:`G` is **regular** on every edge,
      i.e., every edge joins two vertices of equal degree. (For simple graphs, this holds
      in particular for regular graphs.)
    - The quantity is additive over components in the sense that it is a sum over edges; there
      are no cross-component contributions.

    Complexity
    ----------
    Let :math:`n=|V(G)|` and :math:`m=|E(G)|`. Constructing the degree dictionary takes
    :math:`O(n+m)` time. The subsequent edge-sum takes :math:`O(m)` time. Memory usage is
    :math:`O(n)` for the cached degrees.

    Examples
    --------
    >>> import networkx as nx
    >>> # Path P4 has degrees [1,2,2,1]; edge differences are 1,0,1 so irr=2
    >>> G = nx.path_graph(4)
    >>> irregularity(G)
    2

    >>> # Any regular graph has irr=0
    >>> H = nx.cycle_graph(6)
    >>> irregularity(H)
    0
    """
    deg = dict(G.degree())
    return sum(abs(deg[u] - deg[v]) for u, v in G.edges())

def n1_degree_count(G):
    r"""
    Compute :math:`n_1(G)`, the number of degree-1 vertices of a graph :math:`G`.

    This invariant is the multiplicity of 1 in the degree multiset (degree sequence) of :math:`G`:

    .. math::

        n_1(G) \;=\; \bigl|\{\, v \in V(G) : \deg(v) = 1 \,\}\bigr|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Degrees are taken from ``G.degree()``, following NetworkX conventions:

        - ``Graph``: undirected degree.
        - ``DiGraph``: total degree (in-degree + out-degree).
        - ``MultiGraph`` / ``MultiDiGraph``: degree counts edge multiplicity.

    Returns
    -------
    int
        The number of vertices :math:`v` with :math:`\deg(v)=1`. If :math:`G` has no vertices,
        returns 0.

    Notes
    -----
    - For simple undirected graphs, :math:`n_1(G)` is the number of **leaves**.
    - Isolated vertices (degree 0) do not contribute.
    - This quantity depends on the degree convention for directed/multi graphs as described above.

    Complexity
    ----------
    :math:`O(|V(G)|)`, since it scans the degree view once.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)  # degrees: 1,2,2,2,1
    >>> n1_degree_count(G)
    2
    """
    return sum(1 for _, d in G.degree() if d == 1)


def distinct_degree_count(G):
    r"""
    Return the number of distinct vertex degrees attained in a graph :math:`G`.

    Formally, this computes the cardinality of the set of degrees appearing among vertices:

    .. math::

        \bigl|\{\, \deg(v) : v \in V(G) \,\}\bigr|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Degrees are taken from ``G.degree()``, following NetworkX conventions:

        - ``Graph``: undirected degree.
        - ``DiGraph``: total degree (in-degree + out-degree).
        - ``MultiGraph`` / ``MultiDiGraph``: degree counts edge multiplicity.

    Returns
    -------
    int
        The number of distinct degree values occurring in :math:`G`. For the empty graph
        (no vertices), this returns 0.

    Notes
    -----
    - For simple undirected graphs, this is the number of distinct entries in the degree sequence.
    - This is sometimes used as a coarse measure of “degree heterogeneity”.
    - If you want distinct *in-degrees* or *out-degrees* for a digraph, use ``G.in_degree()``
      or ``G.out_degree()`` instead of ``G.degree()``.

    Complexity
    ----------
    :math:`O(|V(G)|)` time and :math:`O(k)` additional space, where :math:`k` is the number of
    distinct degrees (at most :math:`|V(G)|`).

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)  # degrees: {1,2}
    >>> distinct_degree_count(G)
    2
    >>> H = nx.empty_graph(4)  # degrees: {0}
    >>> distinct_degree_count(H)
    1
    >>> distinct_degree_count(nx.empty_graph(0))
    0
    """
    return len({d for _, d in G.degree()})


def count_of_maximum_degree_vertices(G):
    r"""
    Count the vertices attaining the maximum degree in a graph :math:`G`.

    Let :math:`\Delta(G) = \max\{\deg(v) : v \in V(G)\}` be the maximum degree. This function returns

    .. math::

        \bigl|\{\, v \in V(G) : \deg(v) = \Delta(G) \,\}\bigr|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Degrees are taken from ``G.degree()`` using NetworkX conventions for the
        graph type (undirected degree for ``Graph``, total degree for ``DiGraph``, multiplicity for
        ``MultiGraph``).

    Returns
    -------
    int
        The number of vertices of degree :math:`\Delta(G)`. If :math:`G` has no vertices,
        returns 0.

    Notes
    -----
    - For simple undirected graphs, this is the number of vertices with maximum degree.
    - For directed graphs, this uses **total degree** unless you substitute ``G.in_degree()``
      or ``G.out_degree()`` by convention.

    Complexity
    ----------
    :math:`O(|V(G)|)` time to scan degrees (and :math:`O(|V(G)|)` auxiliary space in this
    particular implementation due to materializing the degree list).

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.star_graph(5)  # center degree 5, leaves degree 1
    >>> count_of_maximum_degree_vertices(G)
    1
    """
    degs = [d for _, d in G.degree()]
    if not degs:
        return 0
    dmax = max(degs)
    return sum(1 for d in degs if d == dmax)


def count_of_minimum_degree_vertices(G):
    r"""
    Count the vertices attaining the minimum degree in a graph :math:`G`.

    Let :math:`\delta(G) = \min\{\deg(v) : v \in V(G)\}` be the minimum degree. This function returns

    .. math::

        \bigl|\{\, v \in V(G) : \deg(v) = \delta(G) \,\}\bigr|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Degrees are taken from ``G.degree()`` using NetworkX conventions for the
        graph type (undirected degree for ``Graph``, total degree for ``DiGraph``, multiplicity for
        ``MultiGraph``).

    Returns
    -------
    int
        The number of vertices of degree :math:`\delta(G)`. If :math:`G` has no vertices,
        returns 0.

    Notes
    -----
    - For simple undirected graphs, isolated vertices (degree 0) determine :math:`\delta(G)=0`
      whenever they exist.
    - For directed graphs, this uses **total degree** unless you substitute ``G.in_degree()``
      or ``G.out_degree()`` by convention.

    Complexity
    ----------
    :math:`O(|V(G)|)` time to scan degrees (and :math:`O(|V(G)|)` auxiliary space in this
    particular implementation due to materializing the degree list).

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)  # min degree is 1, achieved by 2 endpoints
    >>> count_of_minimum_degree_vertices(G)
    2
    >>> H = nx.empty_graph(4)  # all degrees 0
    >>> count_of_minimum_degree_vertices(H)
    4
    """
    degs = [d for _, d in G.degree()]
    if not degs:
        return 0
    dmin = min(degs)
    return sum(1 for d in degs if d == dmin)

