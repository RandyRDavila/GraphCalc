from dataclasses import dataclass
from typing import Optional
import numpy as np
import networkx as nx

import graphcalc.graphs as gc
from ..core import SimpleGraph
from graphcalc.utils import enforce_type, GraphLike
from graphcalc.metadata import invariant_metadata

__all__ = [
    'adjacency_matrix',
    'laplacian_matrix',
    'adjacency_eigenvalues',
    'laplacian_eigenvalues',
    'algebraic_connectivity',
    'spectral_radius',
    'largest_laplacian_eigenvalue',
    'zero_adjacency_eigenvalues_count',
    'second_largest_adjacency_eigenvalue',
    'smallest_adjacency_eigenvalue',
    'adjacency_positive_inertia_index',
    'adjacency_negative_inertia_index',
    'adjacency_zero_inertia_index',
    'adjacency_nullity',
    'adjacency_inertia_triple',
    'adjacency_signature',
    'adjacency_rank',
    'adjacency_smallest_positive_eigenvalue',
    'adjacency_graph_energy',
    'AdjacencyInertia',
]

@enforce_type(0, (nx.Graph, SimpleGraph))
def adjacency_matrix(G: GraphLike) -> np.ndarray:
    r"""
    Compute the adjacency matrix of a graph.

    For a simple graph :math:`G = (V,E)` with vertex set
    :math:`V = \{0,1,\dots,n-1\}`, the **adjacency matrix**
    :math:`A(G)` is the :math:`n \times n` matrix defined by:

    .. math::
       A_{ij} =
       \begin{cases}
           1 & \text{if } \{i,j\} \in E, \\
           0 & \text{otherwise}.
       \end{cases}

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph. Vertex labels will be relabeled
        to consecutive integers :math:`0,1,\dots,n-1`
        for the matrix representation.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix :math:`A(G)` as a dense NumPy array.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> gc.adjacency_matrix(G)
    array([[0, 1, 1, 0],
           [1, 0, 0, 1],
           [1, 0, 0, 1],
           [0, 1, 1, 0]])
    """
    G = nx.convert_node_labels_to_integers(G)
    return nx.to_numpy_array(G, dtype=int)


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def laplacian_matrix(G: GraphLike) -> np.array:
    r"""
    Compute the Laplacian matrix of a graph.

    For a graph :math:`G = (V,E)` with adjacency matrix :math:`A(G)`
    and degree matrix :math:`D(G) = \mathrm{diag}(\deg(v_0), \dots, \deg(v_{n-1}))`,
    the **combinatorial Laplacian matrix** is defined as:

    .. math::
       L(G) = D(G) - A(G).

    This symmetric positive semidefinite matrix encodes important structural
    properties of the graph, including connectivity, spanning trees,
    and spectral invariants.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph. Vertex labels will be relabeled
        to consecutive integers :math:`0,1,\dots,n-1`
        for the matrix representation.

    Returns
    -------
    numpy.ndarray
        The Laplacian matrix :math:`L(G)` as a dense NumPy array.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> gc.laplacian_matrix(G)
    array([[ 2, -1, -1,  0],
           [-1,  2,  0, -1],
           [-1,  0,  2, -1],
           [ 0, -1, -1,  2]])
    """
    G = nx.convert_node_labels_to_integers(G)  # Ensure node labels are integers
    A = nx.to_numpy_array(G, dtype=int)  # Adjacency matrix
    Degree = np.diag(np.sum(A, axis=1))  # Degree matrix
    return Degree - A  # Laplacian matrix

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def adjacency_eigenvalues(G: GraphLike) -> float:
    r"""
    Compute the eigenvalues of the adjacency matrix of a graph.

    For a graph :math:`G=(V,E)` with adjacency matrix :math:`A(G)`,
    the **adjacency eigenvalues** are the roots of the characteristic
    polynomial

    .. math::
        \det(\lambda I - A(G)) = 0.

    These eigenvalues (the **spectrum** of the graph) encode rich
    structural information, including connectivity, regularity,
    and expansion properties.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    numpy.ndarray
        The sorted list of real eigenvalues of :math:`A(G)`.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> eigenvals = gc.adjacency_eigenvalues(G)
    >>> np.allclose(eigenvals, [-2.0, 0.0, 0.0, 2.0], atol=1e-6)
    True
    """
    A = nx.to_numpy_array(G, dtype=int)  # Adjacency matrix
    eigenvals = np.linalg.eigvalsh(A)
    return np.sort(eigenvals)

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def laplacian_eigenvalues(G: GraphLike):  # ideally -> np.ndarray
    r"""
    Compute the eigenvalues of the (combinatorial) Laplacian matrix of a graph.

    For a graph :math:`G=(V,E)`, the (combinatorial) Laplacian is

    .. math::
        L(G) \;=\; D(G) - A(G),

    where :math:`A(G)` is the adjacency matrix and :math:`D(G)` is the diagonal matrix of
    vertex degrees. The **Laplacian eigenvalues** are the eigenvalues of :math:`L(G)`,
    equivalently the values :math:`\lambda` satisfying

    .. math::
        \det(\lambda I - L(G)) = 0.

    Laplacian eigenvalues are real and nonnegative and play a central role in spectral
    graph theory. In particular:

    - The multiplicity of 0 equals the number of connected components of :math:`G`.
    - The second-smallest eigenvalue is the **algebraic connectivity**.
    - The largest eigenvalue provides bounds for various graph invariants.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    numpy.ndarray
        The Laplacian eigenvalues of :math:`L(G)`, sorted in nondecreasing order.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.laplacian_eigenvalues(G), np.array([0., 2., 2., 4.]))
    True

    The number of zero eigenvalues equals the number of connected components:

    >>> import numpy as np
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> H = nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))  # 2 components
    >>> eigs = gc.laplacian_eigenvalues(H)
    >>> int(np.sum(np.isclose(eigs, 0.0)))
    2
    """

    L = laplacian_matrix(G)
    eigenvals = np.linalg.eigvalsh(L)
    return np.sort(eigenvals)

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Algebraic connectivity",
    notation=r"a(G)",
    category="spectral",
    aliases=("Fiedler value", "second-smallest Laplacian eigenvalue"),
    definition=(
        "The algebraic connectivity a(G) is the second-smallest eigenvalue of "
        "the Laplacian matrix of G."
    ),
)
def algebraic_connectivity(G: GraphLike) -> float:
    r"""
    Compute the algebraic connectivity of a graph.

    For a graph :math:`G = (V,E)` with Laplacian matrix :math:`L(G)`,
    the **algebraic connectivity** is defined as the second-smallest
    Laplacian eigenvalue:

    .. math::
        a(G) = \lambda_2(L(G)),

    where the eigenvalues are ordered

    .. math::
        0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n.

    Properties
    ----------
    * :math:`a(G) > 0` if and only if :math:`G` is connected.
    * Larger values of :math:`a(G)` indicate greater graph connectivity
      and expansion.
    * The corresponding eigenvector is known as the **Fiedler vector**,
      used in spectral clustering and partitioning.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The algebraic connectivity :math:`a(G)` of the graph.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.algebraic_connectivity(G), 2.0)
    True
    """
    eigenvals = laplacian_eigenvalues(G)
    return eigenvals[1]  # Second smallest eigenvalue

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Spectral radius",
    notation=r"\rho(G)",
    category="spectral",
    aliases=("adjacency spectral radius",),
    definition=(
        "The spectral radius rho(G) is the largest eigenvalue in absolute "
        "value of the adjacency matrix of G."
    ),
)
def spectral_radius(G: GraphLike) -> float:
    r"""
    Compute the spectral radius of a graph.

    For a graph :math:`G = (V,E)` with adjacency matrix :math:`A(G)`,
    the **spectral radius** is the largest eigenvalue in absolute value:

    .. math::
        \rho(G) = \max_i |\lambda_i(A(G))|.

    Properties
    ----------
    * For nonnegative, symmetric adjacency matrices (as in simple graphs),
      the spectral radius equals the largest eigenvalue :math:`\lambda_{\max}`.
    * :math:`\rho(G)` provides bounds on many invariants, such as maximum
      degree and average degree:

      .. math::
          \bar{d}(G) \leq \rho(G) \leq \Delta(G).
    * The eigenvector associated with :math:`\rho(G)` is nonnegative
      by the Perron–Frobenius theorem and often called the **Perron vector**.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The spectral radius :math:`\rho(G)` of the adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.spectral_radius(G), 2.0)
    True
    """
    eigenvals = adjacency_eigenvalues(G)
    return max(abs(eigenvals))

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Largest Laplacian eigenvalue",
    notation=r"\lambda_{\max}(L(G))",
    category="spectral",
    aliases=("largest Laplacian root",),
    definition=(
        "The largest Laplacian eigenvalue is the maximum eigenvalue of the "
        "Laplacian matrix of G."
    ),
)
def largest_laplacian_eigenvalue(G: GraphLike) -> np.float64:
    r"""
    Compute the largest Laplacian eigenvalue of a graph.

    For a graph :math:`G = (V,E)` with Laplacian matrix :math:`L(G)`,
    the **largest Laplacian eigenvalue** is

    .. math::
        \lambda_{\max}(G) = \max_i \lambda_i(L(G)),

    where :math:`0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n`
    are the eigenvalues of :math:`L(G)`.

    Properties
    ----------
    * Always satisfies :math:`\lambda_{\max}(G) \leq 2\Delta(G)`,
      where :math:`\Delta(G)` is the maximum degree.
    * Provides information about expansion, connectivity,
      and can be used in spectral partitioning.
    * Together with the algebraic connectivity (second-smallest Laplacian eigenvalue),
      it bounds the **Laplacian spectrum**.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The largest Laplacian eigenvalue :math:`\lambda_{\max}(G)`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.largest_laplacian_eigenvalue(G), 4.0)
    True
    """
    eigenvals = laplacian_eigenvalues(G)
    return max(abs(eigenvals))

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def zero_adjacency_eigenvalues_count(G: GraphLike) -> int:
    r"""
    Count the number of zero eigenvalues of the adjacency matrix.

    For a graph :math:`G=(V,E)` with adjacency matrix :math:`A(G)`, this function returns
    the multiplicity of the eigenvalue :math:`0` in the spectrum of :math:`A(G)`:

    .. math::
        m_0(G) \;=\; \bigl|\{\, i : \lambda_i(A(G)) = 0 \,\}\bigr|.

    Properties
    ----------
    - :math:`m_0(G)` is the **nullity** of the adjacency matrix :math:`A(G)`.
    - It is related to rank by :math:`\mathrm{rank}(A(G)) = |V(G)| - m_0(G)`.
    - In many families of graphs, the nullity reflects structural redundancy.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The multiplicity of the zero eigenvalue of :math:`A(G)`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> gc.zero_adjacency_eigenvalues_count(G)
    2
    """
    eigenvals = adjacency_eigenvalues(G)
    return sum(1 for e in eigenvals if np.isclose(e, 0))

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Second largest adjacency eigenvalue",
    notation=r"\lambda_{n-1}(A(G))",
    category="spectral",
    aliases=("subleading adjacency eigenvalue",),
    definition=(
        "The second largest adjacency eigenvalue is the second largest "
        "eigenvalue of the adjacency matrix of G."
    ),
)
def second_largest_adjacency_eigenvalue(G: GraphLike) -> np.float64:
    r"""
    Compute the second largest eigenvalue of the adjacency matrix.

    For a graph :math:`G=(V,E)` with adjacency matrix :math:`A(G)`,
    let the eigenvalues of :math:`A(G)` be ordered as

    .. math::
        \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_{|V|}.

    This function returns :math:`\lambda_{|V|-1}`, the second largest
    eigenvalue of :math:`A(G)`.

    Notes
    -----
    * The second largest adjacency eigenvalue is important in the study of
      graph expansion, mixing rates of random walks, and spectral gaps.
    * For a *d*-regular graph, the gap :math:`d - \lambda_{|V|-1}` measures
      how well-connected (expander-like) the graph is.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The second largest eigenvalue of the adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.second_largest_adjacency_eigenvalue(G), 0.0)
    True
    """
    eigenvals = adjacency_eigenvalues(G)
    return eigenvals[-2]  # Second largest in sorted eigenvalues

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Smallest adjacency eigenvalue",
    notation=r"\lambda_{\min}(A(G))",
    category="spectral",
    aliases=("least adjacency eigenvalue",),
    definition=(
        "The smallest adjacency eigenvalue is the minimum eigenvalue of the "
        "adjacency matrix of G."
    ),
)
def smallest_adjacency_eigenvalue(G: GraphLike) -> np.float64:
    r"""
    Compute the smallest eigenvalue of the adjacency matrix.

    For a graph :math:`G=(V,E)` with adjacency matrix :math:`A(G)`,
    let the eigenvalues of :math:`A(G)` be ordered as

    .. math::
        \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_{|V|}.

    This function returns :math:`\lambda_1`, the smallest adjacency
    eigenvalue of :math:`G`.

    Notes
    -----
    * The smallest adjacency eigenvalue is often negative unless the graph is complete multipartite.
    * It appears in Hoffman's bound for the chromatic number:

      .. math::
          \chi(G) \geq 1 - \frac{\lambda_{\max}}{\lambda_{\min}},

      where :math:`\lambda_{\max}` is the largest adjacency eigenvalue
      and :math:`\lambda_{\min}` is the smallest.
    * Also useful in spectral extremal graph theory and characterizations
      of special graph classes (e.g., line graphs).

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The smallest eigenvalue of the adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.smallest_adjacency_eigenvalue(G), -2.0)
    True
    """
    eigenvals = adjacency_eigenvalues(G)
    return eigenvals[0]  # Smallest eigenvalue


@dataclass(frozen=True)
class AdjacencyInertia:
    r"""
    Inertia triple of the adjacency matrix of a graph.

    For a graph :math:`G`, let :math:`A(G)` be its adjacency matrix. The
    **adjacency inertia** is the triple

    .. math::
        (p(G), n(G), z(G)),

    where:

    - :math:`p(G)` is the number of positive eigenvalues of :math:`A(G)`,
    - :math:`n(G)` is the number of negative eigenvalues of :math:`A(G)`,
    - :math:`z(G)` is the number of zero eigenvalues of :math:`A(G)`.

    Since :math:`A(G)` is real symmetric for an undirected graph, these counts
    are well-defined over the real numbers and satisfy

    .. math::
        p(G) + n(G) + z(G) = |V(G)|.

    Parameters
    ----------
    positive : int
        The number of positive adjacency eigenvalues.
    negative : int
        The number of negative adjacency eigenvalues.
    zero : int
        The number of zero adjacency eigenvalues.
    """
    positive: int
    negative: int
    zero: int


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Positive adjacency inertia index",
    notation=r"p(G)",
    category="spectral",
    aliases=("positive inertia index",),
    definition=(
        "The positive adjacency inertia index p(G) is the number of positive "
        "eigenvalues of the adjacency matrix of G."
    ),
)
def adjacency_positive_inertia_index(
    G: GraphLike,
    tol: float = 1e-10,
) -> int:
    r"""
    Compute the positive inertia index of the adjacency matrix of a graph.

    For a graph :math:`G` with adjacency eigenvalues
    :math:`\lambda_1,\dots,\lambda_n`, the **positive inertia index** is

    .. math::
        p(G) = |\{i : \lambda_i > 0\}|.

    Numerically, eigenvalues greater than ``tol`` are counted as positive.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used to decide positivity. Eigenvalues greater
        than ``tol`` are counted as positive. Default is ``1e-10``.

    Returns
    -------
    int
        The number of positive adjacency eigenvalues of :math:`G`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.adjacency_positive_inertia_index(G)
    1
    """
    ev = adjacency_eigenvalues(G)
    return int(np.sum(ev > tol))


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Negative adjacency inertia index",
    notation=r"n(G)",
    category="spectral",
    aliases=("negative inertia index",),
    definition=(
        "The negative adjacency inertia index n(G) is the number of negative "
        "eigenvalues of the adjacency matrix of G."
    ),
)
def adjacency_negative_inertia_index(
    G: GraphLike,
    tol: float = 1e-10,
) -> int:
    r"""
    Compute the negative inertia index of the adjacency matrix of a graph.

    For a graph :math:`G` with adjacency eigenvalues
    :math:`\lambda_1,\dots,\lambda_n`, the **negative inertia index** is

    .. math::
        n(G) = |\{i : \lambda_i < 0\}|.

    Numerically, eigenvalues less than ``-tol`` are counted as negative.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used to decide negativity. Eigenvalues less
        than ``-tol`` are counted as negative. Default is ``1e-10``.

    Returns
    -------
    int
        The number of negative adjacency eigenvalues of :math:`G`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.adjacency_negative_inertia_index(G)
    1
    """
    ev = adjacency_eigenvalues(G)
    return int(np.sum(ev < -tol))


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Adjacency nullity",
    notation=r"\eta(G)",
    category="spectral",
    aliases=("nullity of the adjacency matrix",),
    definition=(
        "The adjacency nullity eta(G) is the multiplicity of the eigenvalue 0 "
        "in the spectrum of the adjacency matrix of G."
    ),
)
def adjacency_nullity(
    G: GraphLike,
    tol: float = 1e-10,
) -> int:
    r"""
    Compute the adjacency nullity of a graph.

    For a graph :math:`G` with adjacency matrix :math:`A(G)`, the
    **adjacency nullity** is the multiplicity of the eigenvalue :math:`0`
    in the spectrum of :math:`A(G)`. Equivalently,

    .. math::
        \eta(G) = \dim(\ker(A(G))).

    Numerically, eigenvalues whose absolute value is at most ``tol`` are
    treated as zero.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used to decide whether an eigenvalue is zero.
        Default is ``1e-10``.

    Returns
    -------
    int
        The adjacency nullity of :math:`G`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.adjacency_nullity(G)
    1
    """
    ev = adjacency_eigenvalues(G)
    return int(np.sum(np.abs(ev) <= tol))


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Zero adjacency inertia index",
    notation=r"z(G)",
    category="spectral",
    aliases=("zero inertia index",),
    definition=(
        "The zero adjacency inertia index z(G) is the number of zero "
        "eigenvalues of the adjacency matrix of G."
    ),
)
def adjacency_zero_inertia_index(
    G: GraphLike,
    tol: float = 1e-10,
) -> int:
    r"""
    Compute the zero inertia index of the adjacency matrix of a graph.

    The **zero inertia index** is the number of zero adjacency eigenvalues.
    For an undirected graph, this is exactly the adjacency nullity:

    .. math::
        z(G) = \eta(G).

    Numerically, eigenvalues whose absolute value is at most ``tol`` are
    treated as zero.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used to decide whether an eigenvalue is zero.
        Default is ``1e-10``.

    Returns
    -------
    int
        The zero inertia index of :math:`G`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.adjacency_zero_inertia_index(G)
    1
    """
    return adjacency_nullity(G, tol=tol)


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def adjacency_inertia_triple(
    G: GraphLike,
    tol: float = 1e-10,
) -> AdjacencyInertia:
    r"""
    Compute the inertia triple of the adjacency matrix of a graph.

    For a graph :math:`G`, the **adjacency inertia triple** is

    .. math::
        (p(G), n(G), z(G)),

    where :math:`p(G)`, :math:`n(G)`, and :math:`z(G)` denote the numbers
    of positive, negative, and zero eigenvalues of :math:`A(G)`,
    respectively.

    Since the adjacency matrix of an undirected graph is real symmetric,
    these quantities satisfy

    .. math::
        p(G) + n(G) + z(G) = |V(G)|.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used to classify eigenvalues as positive,
        negative, or zero. Default is ``1e-10``.

    Returns
    -------
    AdjacencyInertia
        The adjacency inertia triple of :math:`G`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.adjacency_inertia_triple(G)
    AdjacencyInertia(positive=1, negative=1, zero=1)
    """
    ev = adjacency_eigenvalues(G)
    p = int(np.sum(ev > tol))
    n = int(np.sum(ev < -tol))
    z = int(np.sum(np.abs(ev) <= tol))
    return AdjacencyInertia(positive=p, negative=n, zero=z)


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Adjacency signature",
    notation=r"s(G)",
    category="spectral",
    aliases=("signature of the adjacency matrix",),
    definition=(
        "The adjacency signature s(G) is the difference p(G) - n(G), where "
        "p(G) and n(G) are the positive and negative adjacency inertia indices."
    ),
)
def adjacency_signature(
    G: GraphLike,
    tol: float = 1e-10,
) -> int:
    r"""
    Compute the signature of the adjacency matrix of a graph.

    For a graph :math:`G`, the **adjacency signature** is defined by

    .. math::
        s(G) = p(G) - n(G),

    where :math:`p(G)` and :math:`n(G)` are the positive and negative
    inertia indices of :math:`A(G)`.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used when classifying eigenvalues. Default is
        ``1e-10``.

    Returns
    -------
    int
        The adjacency signature of :math:`G`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> gc.adjacency_signature(G)
    0
    """
    p = adjacency_positive_inertia_index(G, tol=tol)
    n = adjacency_negative_inertia_index(G, tol=tol)
    return p - n


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Adjacency rank",
    notation=r"\operatorname{rank}(A(G))",
    category="spectral",
    aliases=("rank of the adjacency matrix",),
    definition=(
        "The adjacency rank is the rank of the adjacency matrix A(G), "
        "equivalently the number of nonzero adjacency eigenvalues counted "
        "with multiplicity."
    ),
)
def adjacency_rank(
    G: GraphLike,
    tol: float = 1e-10,
) -> int:
    r"""
    Compute the rank of the adjacency matrix of a graph.

    For a graph :math:`G` with adjacency matrix :math:`A(G)`, the
    **adjacency rank** is

    .. math::
        \operatorname{rank}(A(G)).

    In terms of the adjacency inertia indices, one has

    .. math::
        \operatorname{rank}(A(G)) = p(G) + n(G),

    since zero eigenvalues do not contribute to the rank.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used when classifying eigenvalues. Default is
        ``1e-10``.

    Returns
    -------
    int
        The rank of :math:`A(G)`.

    Examples
    --------
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> gc.adjacency_rank(G)
    2
    """
    p = adjacency_positive_inertia_index(G, tol=tol)
    n = adjacency_negative_inertia_index(G, tol=tol)
    return p + n


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Smallest positive adjacency eigenvalue",
    notation=r"\lambda_{\min}^{+}(A(G))",
    category="spectral",
    aliases=("least positive adjacency eigenvalue",),
    definition=(
        "The smallest positive adjacency eigenvalue is the least strictly "
        "positive eigenvalue of the adjacency matrix of G, if one exists."
    ),
)
def adjacency_smallest_positive_eigenvalue(
    G: GraphLike,
    tol: float = 1e-12,
) -> Optional[float]:
    r"""
    Compute the smallest strictly positive adjacency eigenvalue of a graph.

    If the adjacency spectrum of :math:`G` is

    .. math::
        \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n,

    then this function returns the smallest eigenvalue satisfying
    :math:`\lambda_i > 0`.

    Numerically, eigenvalues greater than ``tol`` are treated as positive.
    If no such eigenvalue exists, the function returns ``None``.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.
    tol : float, optional
        Numerical tolerance used to test positivity. Default is ``1e-12``.

    Returns
    -------
    float or None
        The smallest strictly positive adjacency eigenvalue of :math:`G`,
        or ``None`` if the adjacency matrix has no positive eigenvalues.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(3)
    >>> np.allclose(gc.adjacency_smallest_positive_eigenvalue(G), np.sqrt(2))
    True
    """
    ev = adjacency_eigenvalues(G)
    pos = ev[ev > tol]
    return None if pos.size == 0 else float(pos[0])


@enforce_type(0, (nx.Graph, gc.SimpleGraph))
@invariant_metadata(
    display_name="Graph energy",
    notation=r"E(G)",
    category="spectral",
    aliases=("adjacency energy",),
    definition=(
        "The graph energy E(G) is the sum of the absolute values of the "
        "adjacency eigenvalues of G."
    ),
)
def adjacency_graph_energy(G: GraphLike) -> float:
    r"""
    Compute the graph energy of a graph.

    For a graph :math:`G` with adjacency eigenvalues
    :math:`\lambda_1,\dots,\lambda_n`, the **graph energy** is defined as

    .. math::
        E(G) = \sum_{i=1}^n |\lambda_i|.

    This invariant was introduced by Gutman and plays an important role in
    spectral graph theory and chemical graph theory.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The graph energy of :math:`G`.

    Examples
    --------
    >>> import numpy as np
    >>> import graphcalc.graphs as gc
    >>> from graphcalc.graphs.generators import path_graph
    >>> G = path_graph(2)
    >>> np.allclose(gc.adjacency_graph_energy(G), 2.0)
    True
    """
    ev = adjacency_eigenvalues(G)
    return float(np.sum(np.abs(ev)))
