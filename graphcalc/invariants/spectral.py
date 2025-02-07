import numpy as np
import networkx as nx

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

def adjacency_matrix(G):
    r"""
    Compute the adjacency matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
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
    G = nx.convert_node_labels_to_integers(G)  # Ensure node labels are integers
    return nx.to_numpy_array(G, dtype=int)  # Adjacency matrix

def laplacian_matrix(G):
    r"""
    Compute the Laplacian matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
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


def adjacency_eigenvalues(G):
    r"""
    Compute the eigenvalues of the adjacency matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    numpy.ndarray
        Sorted eigenvalues of the adjacency matrix.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.adjacency_eigenvalues(G)
    array([-2.00000000e+00,  0.00000000e+00,  6.59737022e-17,  2.00000000e+00])
    """
    A = nx.to_numpy_array(G, dtype=int)  # Adjacency matrix
    eigenvals = np.linalg.eigvals(A)
    return np.sort(eigenvals)


def laplacian_eigenvalues(G):
    r"""
    Compute the eigenvalues of the Laplacian matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
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
    >>> gc.laplacian_eigenvalues(G)
    array([-2.22044605e-16,  2.00000000e+00,  2.00000000e+00,  4.00000000e+00])
    """
    L = laplacian_matrix(G)
    eigenvals = np.linalg.eigvals(L)
    return np.sort(eigenvals)


def algebraic_connectivity(G):
    r"""
    Compute the algebraic connectivity of a graph.

    The algebraic connectivity is the second smallest eigenvalue of the Laplacian matrix.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    float
        The algebraic connectivity of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.algebraic_connectivity(G)
    1.9999999999999996
    """
    eigenvals = laplacian_eigenvalues(G)
    return eigenvals[1]  # Second smallest eigenvalue


def spectral_radius(G):
    r"""
    Compute the spectral radius (largest eigenvalue by absolute value) of the adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    float
        The spectral radius of the adjacency matrix.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.spectral_radius(G)
    2.0
    """
    eigenvals = adjacency_eigenvalues(G)
    return max(abs(eigenvals))


def largest_laplacian_eigenvalue(G):
    r"""
    Compute the largest eigenvalue of the Laplacian matrix.

    Parameters
    ----------
    G : networkx.Graph
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
    >>> gc.largest_laplacian_eigenvalue(G)
    3.9999999999999996
    """
    eigenvals = laplacian_eigenvalues(G)
    return max(abs(eigenvals))


def zero_adjacency_eigenvalues_count(G):
    r"""
    Compute the number of zero eigenvalues of the adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph
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


def second_largest_adjacency_eigenvalue(G):
    r"""
    Compute the second largest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph
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
    >>> gc.second_largest_adjacency_eigenvalue(G)
    6.597370221478735e-17
    """
    eigenvals = adjacency_eigenvalues(G)
    return eigenvals[-2]  # Second largest in sorted eigenvalues


def smallest_adjacency_eigenvalue(G):
    r"""
    Compute the smallest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    float
        The smallest eigenvalue of the adjacency matrix.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.smallest_adjacency_eigenvalue(G)
    -1.9999999999999996
    """
    eigenvals = adjacency_eigenvalues(G)
    return eigenvals[0]  # Smallest eigenvalue
