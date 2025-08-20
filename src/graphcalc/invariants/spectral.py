import numpy as np
import networkx as nx

import graphcalc as gc
from graphcalc import SimpleGraph
from graphcalc.utils import enforce_type, GraphLike

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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> eigenvals = gc.adjacency_eigenvalues(G)
    >>> np.allclose(eigenvals, [-2.0, 0.0, 0.0, 2.0], atol=1e-6)
    True
    """
    A = nx.to_numpy_array(G, dtype=int)  # Adjacency matrix
    eigenvals = np.linalg.eigvals(A)
    return np.sort(eigenvals)

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def laplacian_eigenvalues(G: GraphLike) -> float:
    r"""
    Compute the eigenvalues of the Laplacian matrix of a graph.

    For a graph :math:`G=(V,E)` with Laplacian matrix
    :math:`L(G) = D(G) - A(G)`, the **Laplacian eigenvalues**
    are the roots of the characteristic polynomial

    .. math::
        \det(\lambda I - L(G)) = 0.

    These eigenvalues are always nonnegative and play a central role
    in spectral graph theory. In particular:
      * The multiplicity of 0 equals the number of connected components.
      * The second-smallest eigenvalue (the **algebraic connectivity**)
        measures how well the graph is connected.
      * The largest eigenvalue provides bounds on graph invariants
        such as the diameter.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    numpy.ndarray
        The sorted eigenvalues of the Laplacian matrix :math:`L(G)`.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> eigenvals = gc.laplacian_eigenvalues(G)
    >>> eigenvals
    array([0., 2., 2., 4.])
    """
    L = laplacian_matrix(G)
    eigenvals = np.linalg.eigvals(L)
    return np.sort(eigenvals)

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.algebraic_connectivity(G), 2.0)
    True
    """
    eigenvals = laplacian_eigenvalues(G)
    return eigenvals[1]  # Second smallest eigenvalue

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.spectral_radius(G), 2.0)
    True
    """
    eigenvals = adjacency_eigenvalues(G)
    return max(abs(eigenvals))

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
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

    For a graph :math:`G = (V,E)` with adjacency matrix :math:`A(G)`,
    this function returns the multiplicity of the eigenvalue :math:`0` in the spectrum
    of :math:`A(G)`:

    .. math::
        m_0(G) = |\{ i : \lambda_i(A(G)) = 0 \}|.

    Properties
    ----------
    * :math:`m_0(G)` is the **nullity** of the adjacency matrix.
    * Closely related to the **rank**:
      .. math:: \mathrm{rank}(A(G)) = |V| - m_0(G).
    * In some cases, reflects structural redundancy and graph symmetry.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The multiplicity of the zero eigenvalue of the adjacency matrix.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> gc.zero_adjacency_eigenvalues_count(G)
    2
    """
    eigenvals = adjacency_eigenvalues(G)
    return sum(1 for e in eigenvals if np.isclose(e, 0))

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.second_largest_adjacency_eigenvalue(G), 0.0)
    True
    """
    eigenvals = adjacency_eigenvalues(G)
    return eigenvals[-2]  # Second largest in sorted eigenvalues

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(4)
    >>> np.allclose(gc.smallest_adjacency_eigenvalue(G), -2.0)
    True
    """
    eigenvals = adjacency_eigenvalues(G)
    return eigenvals[0]  # Smallest eigenvalue
