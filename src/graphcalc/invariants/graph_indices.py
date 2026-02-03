
import math
import networkx as nx
import graphcalc as gc
from graphcalc import SimpleGraph
from graphcalc.utils import enforce_type, GraphLike

__all__ = [
    "randic_index",
    "zagreb_1",
    "zagreb_2",
    "reciprocal_zagreb_1",
    "reciprocal_zagreb_2",
    "abc_index",
    "ga_index",
    "reciprocal_ga_index",
    "sum_connectivity_index",
    "sombor_index",
    "reciprocal_sombor_index",
    "hyper_zagreb_index",
    "reciprocal_hyper_zagreb_index",
    "augmented_zagreb_index",
    "reciprocal_augmented_zagreb_index",
    "harmonic_index",
]

def randic_index(G, alpha=-0.5):
    r"""
    Compute the generalized Randić index :math:`R_\alpha(G)` of a graph :math:`G`.

    For a real parameter :math:`\alpha`, the **generalized Randić index**
    (also called the generalized connectivity index) is

    .. math::
        R_\alpha(G) \;=\; \sum_{\{u,v\}\in E(G)} \bigl(d(u)\,d(v)\bigr)^{\alpha},

    where :math:`d(u)` denotes the degree of :math:`u`.

    The **classical Randić index** is the special case :math:`\alpha=-\tfrac12`:

    .. math::
        R(G) \;=\; \sum_{\{u,v\}\in E(G)} \frac{1}{\sqrt{d(u)\,d(v)}}.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.
    alpha : float, default=-0.5
        The exponent :math:`\alpha`.

    Notes
    -----
    - Only edges contribute, so isolated vertices do not affect the value.
    - In a simple graph, every edge has endpoints of degree at least 1, so
      :math:`\alpha<0` causes no division-by-zero issues.
    - For multigraphs, degrees count multiplicity and parallel edges are summed
      repeatedly; this matches the literal formula over the multiset of edges, which
      may differ from some chemistry conventions.

    Returns
    -------
    float
        The value :math:`R_\alpha(G)`.

    Examples
    --------
    A path on 4 vertices has degrees :math:`(1,2,2,1)`, so

    .. math::
        R(G)=\tfrac{1}{\sqrt{2}}+\tfrac{1}{2}+\tfrac{1}{\sqrt{2}}.

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(4)
    >>> round(gc.randic_index(G), 6)
    1.914214

    A complete graph :math:`K_n` has all degrees :math:`n-1`, so
    :math:`R_{-1/2}(K_n) = |E|/(n-1) = n/2`.

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.randic_index(nx.complete_graph(6))
    3.0

    Changing :math:`\alpha` changes the weighting:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(4)
    >>> gc.randic_index(G, alpha=1)  # sum of degree products over edges
    8
    """
    deg = dict(G.degree())
    return sum((deg[u] * deg[v]) ** alpha for u, v in G.edges())

def zagreb_1(G):
    r"""
    Compute the first Zagreb index :math:`M_1(G)`.

    The **first Zagreb index** is

    .. math::
        M_1(G) = \sum_{v \in V(G)} d(v)^2,

    where :math:`d(v)` is the degree of :math:`v`.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    int
        The first Zagreb index :math:`M_1(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.zagreb_1(nx.path_graph(4))  # degrees 1,2,2,1
    10
    >>> gc.zagreb_1(nx.complete_graph(4))  # degrees all 3
    36
    """
    return sum(d * d for _, d in G.degree())


def zagreb_2(G):
    r"""
    Compute the second Zagreb index :math:`M_2(G)`.

    The **second Zagreb index** is

    .. math::
        M_2(G) = \sum_{\{u,v\} \in E(G)} d(u)\,d(v),

    where :math:`d(\cdot)` denotes vertex degree.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    int
        The second Zagreb index :math:`M_2(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.zagreb_2(nx.path_graph(4))  # edge degree-products: 1*2 + 2*2 + 2*1
    8
    >>> gc.zagreb_2(nx.complete_graph(4))  # 6 edges, each contributes 3*3
    54
    """
    deg = dict(G.degree())
    return sum(deg[u] * deg[v] for u, v in G.edges())


def reciprocal_zagreb_1(G):
    r"""
    Compute the reciprocal first Zagreb index :math:`RM_1(G)`.

    A common “reciprocal” variant replaces :math:`d(v)^2` by its reciprocal:

    .. math::
        RM_1(G) = \sum_{v \in V(G)} \frac{1}{d(v)^2}.

    Isolated vertices (degree 0) contribute nothing in this implementation.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    float
        The reciprocal first Zagreb index :math:`RM_1(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(4)  # degrees 1,2,2,1
    >>> gc.reciprocal_zagreb_1(G)
    2.5
    >>> H = nx.empty_graph(3)  # all isolated -> contributes 0 by convention here
    >>> gc.reciprocal_zagreb_1(H)
    0.0
    """
    s = 0.0
    for _, d in G.degree():
        if d > 0:
            s += 1.0 / (d * d)
    return s


def reciprocal_zagreb_2(G):
    r"""
    Compute the reciprocal second Zagreb index :math:`RM_2(G)`.

    A common “reciprocal” variant replaces :math:`d(u)d(v)` by its reciprocal:

    .. math::
        RM_2(G) = \sum_{\{u,v\} \in E(G)} \frac{1}{d(u)\,d(v)}.

    For simple graphs, every edge has endpoints of degree at least 1, so no
    division-by-zero occurs.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    float
        The reciprocal second Zagreb index :math:`RM_2(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(4)  # edge degree-products: 2,4,2 -> reciprocals: 1/2+1/4+1/2
    >>> gc.reciprocal_zagreb_2(G)
    1.25
    >>> gc.reciprocal_zagreb_2(nx.complete_graph(4))  # 6 edges, each 1/(3*3)
    0.6666666666666666
    """
    deg = dict(G.degree())
    return sum(1.0 / (deg[u] * deg[v]) for u, v in G.edges())

def abc_index(G):
    r"""
    Compute the Atom–Bond Connectivity (ABC) index of a graph :math:`G`.

    The **ABC index** is

    .. math::
        \mathrm{ABC}(G) \;=\; \sum_{\{u,v\}\in E(G)}
        \sqrt{\frac{d(u)+d(v)-2}{d(u)\,d(v)}}\,,

    where :math:`d(\cdot)` denotes vertex degree.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Notes
    -----
    - For a pendant edge with :math:`d(u)=d(v)=1` (which can only occur in :math:`K_2`),
      the summand is :math:`\sqrt{0/1}=0`.
    - In a simple graph, every edge has endpoints of degree at least 1, so the denominator
      is positive.

    Returns
    -------
    float
        The Atom–Bond Connectivity index :math:`\mathrm{ABC}(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.abc_index(nx.complete_graph(2))  # single edge with degrees 1,1
    0.0

    For :math:`P_3` (degrees 1-2-1), each edge contributes :math:`\sqrt{1/2}`, so the
    total is :math:`\sqrt{2}`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.abc_index(nx.path_graph(3)), 6)
    1.414214

    A 4-cycle has 4 edges with degree-pair (2,2), so each contributes :math:`\sqrt{(2)/(4)}=\sqrt{1/2}`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.abc_index(nx.cycle_graph(4)), 6)
    2.828427
    """
    deg = dict(G.degree())
    s = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        s += math.sqrt((du + dv - 2) / (du * dv))
    return s

def ga_index(G):
    r"""
    Compute the Geometric–Arithmetic (GA) index of a graph :math:`G`.

    The **GA index** is the degree-based topological index

    .. math::
        \mathrm{GA}(G) \;=\; \sum_{\{u,v\}\in E(G)}
        \frac{2\sqrt{d(u)\,d(v)}}{d(u)+d(v)},

    where :math:`d(\cdot)` denotes vertex degree.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Notes
    -----
    - For a simple graph, :math:`d(u),d(v)\ge 1` on every edge, and :math:`d(u)+d(v)\ge 2`,
      so the expression is well-defined.
    - Each summand lies in :math:`(0,1]` by AM–GM, with equality 1 iff :math:`d(u)=d(v)`.

    Returns
    -------
    float
        The GA index :math:`\mathrm{GA}(G)`.

    Examples
    --------
    For :math:`K_2`, the single edge has degrees (1,1), so GA = 1:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.ga_index(nx.complete_graph(2))
    1.0

    For :math:`P_3`, each edge has degrees (1,2), contributing :math:`2\sqrt{2}/3`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.ga_index(nx.path_graph(3)), 6)
    1.885618

    For :math:`C_4`, all edges have degrees (2,2), so each term is 1 and GA = 4:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.ga_index(nx.cycle_graph(4))
    4.0
    """
    deg = dict(G.degree())
    s = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        s += (2.0 * math.sqrt(du * dv)) / (du + dv)
    return s


def reciprocal_ga_index(G):
    r"""
    Compute the reciprocal Geometric–Arithmetic index of a graph :math:`G`.

    This “reciprocal” variant sums the reciprocals of the GA edge terms:

    .. math::
        \mathrm{RGA}(G) \;=\; \sum_{\{u,v\}\in E(G)}
        \frac{d(u)+d(v)}{2\sqrt{d(u)\,d(v)}}.

    For simple graphs, degrees on an edge are at least 1, so this is well-defined.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    float
        The reciprocal GA index :math:`\mathrm{RGA}(G)`.

    Examples
    --------
    For :math:`K_2`, RGA = 1:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.reciprocal_ga_index(nx.complete_graph(2))
    1.0

    For :math:`C_4`, every edge has degrees (2,2), so each term is 1 and RGA = 4:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.reciprocal_ga_index(nx.cycle_graph(4))
    4.0
    """
    deg = dict(G.degree())
    s = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        s += (du + dv) / (2.0 * math.sqrt(du * dv))
    return s

def sum_connectivity_index(G, alpha=-0.5):
    r"""
    Compute the generalized sum-connectivity index :math:`SC_\alpha(G)` of a graph :math:`G`.

    For a real parameter :math:`\alpha`, the **generalized sum-connectivity index** is

    .. math::
        SC_\alpha(G) \;=\; \sum_{\{u,v\}\in E(G)} \bigl(d(u)+d(v)\bigr)^{\alpha},

    where :math:`d(\cdot)` denotes vertex degree.

    The classical sum-connectivity index is the special case :math:`\alpha=-\tfrac12`.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.
    alpha : float, default=-0.5
        The exponent :math:`\alpha`.

    Notes
    -----
    - Only edges contribute, so isolated vertices do not affect the value.
    - In a simple graph, every edge satisfies :math:`d(u)+d(v)\ge 2`, so negative
      :math:`\alpha` causes no division-by-zero issues.

    Returns
    -------
    float
        The value :math:`SC_\alpha(G)`.

    Examples
    --------
    For :math:`K_2`, the single edge has degrees (1,1), so :math:`SC_{-1/2} = 2^{-1/2}`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.sum_connectivity_index(nx.complete_graph(2)), 6)
    0.707107

    For :math:`P_4`, edge degree-sums are 3, 4, 3, so
    :math:`SC_{-1/2} = 3^{-1/2} + 4^{-1/2} + 3^{-1/2}`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.sum_connectivity_index(nx.path_graph(4)), 6)
    1.654701

    With :math:`\alpha=1`, this is the sum of degree-sums over edges:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.sum_connectivity_index(nx.path_graph(4), alpha=1)
    10
    """
    deg = dict(G.degree())
    return sum((deg[u] + deg[v]) ** alpha for u, v in G.edges())

def sombor_index(G):
    r"""
    Compute the Sombor index :math:`SO(G)` of a graph :math:`G`.

    The **Sombor index** is the degree-based topological index

    .. math::
        SO(G) \;=\; \sum_{\{u,v\}\in E(G)} \sqrt{d(u)^2 + d(v)^2},

    where :math:`d(\cdot)` denotes vertex degree.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    float
        The Sombor index :math:`SO(G)`.

    Examples
    --------
    For :math:`K_2`, degrees are (1,1), so :math:`SO(K_2)=\sqrt{2}`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.sombor_index(nx.complete_graph(2)), 6)
    1.414214

    For :math:`P_3`, degrees are 1-2-1, so each edge contributes :math:`\sqrt{1^2+2^2}=\sqrt{5}`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.sombor_index(nx.path_graph(3)), 6)
    4.472136

    For :math:`C_4`, all degrees are 2, so each edge contributes :math:`\sqrt{8}`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.sombor_index(nx.cycle_graph(4)), 6)
    11.313708
    """
    deg = dict(G.degree())
    s = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        s += math.sqrt(du * du + dv * dv)
    return s


def reciprocal_sombor_index(G):
    r"""
    Compute the reciprocal Sombor index :math:`RSO(G)` of a graph :math:`G`.

    This “reciprocal” variant sums the reciprocals of the per-edge Sombor terms:

    .. math::
        RSO(G) \;=\; \sum_{\{u,v\}\in E(G)} \frac{1}{\sqrt{d(u)^2 + d(v)^2}}.

    For simple graphs, every edge has endpoints of degree at least 1, so each denominator
    is at least :math:`\sqrt{2}` and the expression is well-defined.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    float
        The reciprocal Sombor index :math:`RSO(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.reciprocal_sombor_index(nx.complete_graph(2)), 6)
    0.707107

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.reciprocal_sombor_index(nx.path_graph(3)), 6)  # 2 edges, each 1/sqrt(5)
    0.894427
    """
    deg = dict(G.degree())
    s = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        s += 1.0 / math.sqrt(du * du + dv * dv)
    return s

def hyper_zagreb_index(G):
    r"""
    Compute the Hyper Zagreb index :math:`HM(G)` of a graph :math:`G`.

    The **Hyper Zagreb index** is

    .. math::
        HM(G) \;=\; \sum_{\{u,v\}\in E(G)} \bigl(d(u)+d(v)\bigr)^2,

    where :math:`d(\cdot)` denotes vertex degree.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    int
        The Hyper Zagreb index :math:`HM(G)` (integer-valued for simple graphs).

    Examples
    --------
    For :math:`P_4`, edge degree-sums are 3, 4, 3, so
    :math:`HM(P_4) = 3^2 + 4^2 + 3^2 = 34`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.hyper_zagreb_index(nx.path_graph(4))
    34

    For :math:`K_4`, all degrees are 3 and each of the 6 edges has sum 6, so
    :math:`HM(K_4) = 6 \cdot 6^2 = 216`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.hyper_zagreb_index(nx.complete_graph(4))
    216

    A graph with no edges has Hyper Zagreb index 0:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.hyper_zagreb_index(nx.empty_graph(5))
    0
    """
    deg = dict(G.degree())
    return sum((deg[u] + deg[v]) ** 2 for u, v in G.edges())


def reciprocal_hyper_zagreb_index(G):
    r"""
    Compute the reciprocal Hyper Zagreb index :math:`RHM(G)` of a graph :math:`G`.

    This “reciprocal” variant sums the reciprocals of the Hyper Zagreb edge terms:

    .. math::
        RHM(G) \;=\; \sum_{\{u,v\}\in E(G)} \frac{1}{\bigl(d(u)+d(v)\bigr)^2}.

    For simple graphs, every edge satisfies :math:`d(u)+d(v)\ge 2`, so the expression is
    well-defined.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Returns
    -------
    float
        The reciprocal Hyper Zagreb index :math:`RHM(G)`.

    Examples
    --------
    For :math:`K_2`, the single edge has degree-sum 2, so :math:`RHM=1/4`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.reciprocal_hyper_zagreb_index(nx.complete_graph(2))
    0.25

    For :math:`P_4`, edge degree-sums are 3, 4, 3, so
    :math:`RHM(P_4)=1/3^2 + 1/4^2 + 1/3^2`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> round(gc.reciprocal_hyper_zagreb_index(nx.path_graph(4)), 6)
    0.284722
    """
    deg = dict(G.degree())
    return sum(1.0 / ((deg[u] + deg[v]) ** 2) for u, v in G.edges())

def augmented_zagreb_index(G):
    r"""
    Compute the Augmented Zagreb index :math:`AZI(G)` of a graph :math:`G`
    (with a total-function convention).

    The usual definition is

    .. math::
        AZI(G) \;=\; \sum_{\{u,v\}\in E(G)}
        \left(\frac{d(u)\,d(v)}{d(u)+d(v)-2}\right)^3,

    where :math:`d(\cdot)` denotes vertex degree.

    Convention / edge cases
    -----------------------
    The denominator :math:`d(u)+d(v)-2` is zero exactly when :math:`d(u)=d(v)=1`,
    which in a simple graph occurs only for :math:`G=K_2`. Different sources either
    treat :math:`AZI(K_2)` as undefined or define a special value.

    This implementation uses a **total-function** convention:
    any edge with :math:`d(u)+d(v)-2 \le 0` contributes 0 to the sum.
    In particular, this yields :math:`AZI(K_2)=0`.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Notes
    -----
    For all other edges in a simple graph, :math:`d(u)+d(v)-2 \ge 1`, so the summand is
    well-defined.

    Returns
    -------
    float
        The Augmented Zagreb index :math:`AZI(G)` under the convention above.

    Examples
    --------
    By convention, :math:`AZI(K_2)=0`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.augmented_zagreb_index(nx.complete_graph(2))
    0.0

    For :math:`P_3`, degrees are 1-2-1, and each edge contributes
    :math:`((2)/(1))^3 = 8`, so :math:`AZI(P_3)=16`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.augmented_zagreb_index(nx.path_graph(3))
    16.0

    For :math:`C_4`, every edge has degrees (2,2), so each contributes
    :math:`((4)/(2))^3 = 8`, hence :math:`AZI(C_4)=32`:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.augmented_zagreb_index(nx.cycle_graph(4))
    32.0
    """
    deg = dict(G.degree())
    s = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        denom = du + dv - 2
        if denom <= 0:
            continue
        s += ((du * dv) / denom) ** 3
    return s

def reciprocal_augmented_zagreb_index(G):
    r"""
    Compute a reciprocal variant of the Augmented Zagreb index.

    This per-edge reciprocal variant is

    .. math::
        RAZI(G) \;=\; \sum_{\{u,v\}\in E(G)}
        \left(\frac{d(u)+d(v)-2}{d(u)\,d(v)}\right)^3,

    which is the reciprocal of the usual AZI edge fraction (inside the cube).

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Intended for finite simple undirected graphs.

    Notes
    -----
    - For :math:`K_2`, the single edge has :math:`d(u)=d(v)=1`, so the summand is 0 and
      :math:`RAZI(K_2)=0`.
    - For simple graphs, :math:`d(u),d(v)\ge 1` on every edge, so the expression is
      well-defined.

    Returns
    -------
    float
        The reciprocal Augmented Zagreb index :math:`RAZI(G)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.reciprocal_augmented_zagreb_index(nx.complete_graph(2))
    0.0

    For :math:`P_3`, each edge has degrees (1,2), so each contributes
    :math:`((1)/(2))^3 = 1/8` and the total is 1/4:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> gc.reciprocal_augmented_zagreb_index(nx.path_graph(3))
    0.25
    """
    deg = dict(G.degree())
    s = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        # denom (of the reciprocal fraction) is du*dv >= 1 on any simple-graph edge
        s += ((du + dv - 2) / (du * dv)) ** 3
    return s

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
    return 2*sum((1/(gc.degree(G, v) + gc.degree(G, u)) for u, v in G.edges()))
