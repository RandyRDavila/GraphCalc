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

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the graph.

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

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    numpy.ndarray
        The Laplacian matrix of the graph.

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

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    numpy.ndarray
        Sorted eigenvalues of the adjacency matrix.

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

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    numpy.ndarray
        Sorted eigenvalues of the Laplacian matrix.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> solution = gc.laplacian_eigenvalues(G)
    """
    L = laplacian_matrix(G)
    eigenvals = np.linalg.eigvals(L)
    return np.sort(eigenvals)

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def algebraic_connectivity(G: GraphLike) -> float:
    r"""
    Compute the algebraic connectivity of a graph.

    The algebraic connectivity is the second smallest eigenvalue of the Laplacian matrix.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The algebraic connectivity of the graph.

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
    Compute the spectral radius (largest eigenvalue by absolute value) of the adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The spectral radius of the adjacency matrix.

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
    Compute the largest eigenvalue of the Laplacian matrix.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The largest eigenvalue of the Laplacian matrix.

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
    Compute the number of zero eigenvalues of the adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The number of zero eigenvalues.

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
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> solution = gc.second_largest_adjacency_eigenvalue(G)
    """
    eigenvals = adjacency_eigenvalues(G)
    return eigenvals[-2]  # Second largest in sorted eigenvalues

@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def smallest_adjacency_eigenvalue(G: GraphLike) -> np.float64:
    r"""
    Compute the smallest eigenvalue of the adjacency matrix.

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
