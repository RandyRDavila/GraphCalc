import networkx as nx
import graphcalc as gc
# from graphcalc.utils import enforce_type

import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt

__all__= [
    'order',
    'size',
    'connected',
    'diameter',
    'radius',
    'average_shortest_path_length',
    'bipartite',
    'connected_and_bipartite',
    'chordal',
    'connected_and_chordal',
    'cubic',
    'connected_and_cubic',
    'eulerian',
    'connected_and_eulerian',
    'planar',
    'connected_and_planar',
    'regular',
    'connected_and_regular',
    'subcubic',
    'connected_and_subcubic',
    'tree',
    'SimpleGraph',
    'K_4_free',
    'connected_and_K_4_free',
    'triangle_free',
    'connected_and_triangle_free',
    'claw_free',
    'connected_and_claw_free',
    'planar',
    'connected_and_planar',
    'cograph',
    'connected_and_cograph',
]

class SimpleGraph(nx.Graph):
    r"""
    A subclass of networkx.Graph with additional functionality.

    Features:
    - Optional `name` and `info` attributes for metadata.
    - Default integer labels for nodes.
    - Methods to read and write edge lists to/from CSV files.
    - Method to draw the graph using Matplotlib.

    Parameters
    ----------
    edges : list of tuple, optional
        A list of edges to initialize the graph.
    nodes : list, optional
        A list of nodes to initialize the graph.
    name : str, optional
        An optional name for the graph.
    info : str, optional
        Additional information about the graph.
    *args, **kwargs : arguments
        Arguments passed to the base `networkx.Graph` class.
    """

    def __init__(self, edges=None, nodes=None, name=None, info=None, *args, **kwargs):
        """
        Initialize a SimpleGraph instance with optional edges and nodes.

        Parameters
        ----------
        edges : list of tuple, optional
            A list of edges to initialize the graph.
        nodes : list, optional
            A list of nodes to initialize the graph.
        name : str, optional
            An optional name for the graph.
        info : str, optional
            Additional information about the graph.
        *args, **kwargs : arguments
            Arguments passed to the base `networkx.Graph` class.

        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.info = info

        # Add nodes and edges if provided
        if nodes:
            self.add_nodes_from(nodes)

        if edges:
            self.add_edges_from(edges)

    def write_edgelist_to_csv(self, filepath):
        r"""
        Write the edge list of the graph to a CSV file.

        Parameters
        ----------
        filepath : str
            The path to the CSV file where the edge list will be written.
        """
        with open(filepath, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Source", "Target"])
            for edge in self.edges:
                writer.writerow(edge)

    def read_edge_list(self, filepath, delimiter=None):
        r"""
        Read an edge list from a file (CSV or TXT) and add edges to the graph.

        Parameters
        ----------
        filepath : str
            The path to the file containing the edge list.
        delimiter : str, optional
            The delimiter used in the file (default is ',' for CSV and whitespace for TXT).

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is invalid.

        Notes
        -----
        - For CSV files, the file must have a header with "Source" and "Target".
        - For TXT files, the file should contain one edge per line with node pairs separated by whitespace.
        """
        import os

        # Determine the file type
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext == ".csv":
            # Set default delimiter for CSV
            delimiter = delimiter or ","
            self._read_edge_list_csv(filepath, delimiter)
        elif ext == ".txt":
            # Set default delimiter for TXT
            delimiter = delimiter or None  # Default for whitespace-separated files
            self._read_edge_list_txt(filepath, delimiter)
        else:
            raise ValueError("Unsupported file format. Only .csv and .txt files are supported.")

    def _read_edge_list_csv(self, filepath, delimiter):
        """Internal method to read edge lists from a CSV file."""
        import csv

        try:
            with open(filepath, mode="r") as csvfile:
                reader = csv.reader(csvfile, delimiter=delimiter)
                header = next(reader)  # Read the header
                if header != ["Source", "Target"]:
                    raise ValueError("CSV file must have 'Source' and 'Target' as headers.")
                for row in reader:
                    if len(row) != 2:
                        raise ValueError(f"Invalid row in CSV file: {row}")
                    u, v = map(int, row)  # Convert nodes to integers
                    self.add_edge(u, v)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' does not exist.")
        except Exception as e:
            raise Exception(f"Error reading edge list from '{filepath}': {e}")

    def _read_edge_list_txt(self, filepath, delimiter):
        """Internal method to read edge lists from a TXT file."""
        try:
            with open(filepath, mode="r") as txtfile:
                for line in txtfile:
                    if line.strip():  # Skip empty lines
                        nodes = line.strip().split(delimiter)
                        if len(nodes) != 2:
                            raise ValueError(f"Invalid line in TXT file: {line.strip()}")
                        u, v = map(int, nodes)  # Convert nodes to integers
                        self.add_edge(u, v)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' does not exist.")
        except Exception as e:
            raise Exception(f"Error reading edge list from '{filepath}': {e}")

    def read_adjacency_matrix(self, filepath, delimiter=None):
        r"""
        Read an adjacency matrix from a file (CSV or TXT) and create the graph.

        Parameters
        ----------
        filepath : str
            The path to the file containing the adjacency matrix.
        delimiter : str, optional
            The delimiter used in the file (default is ',' for CSV and whitespace for TXT).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is invalid or the adjacency matrix is not square.
        """
        import os

        # Determine the file type
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        # Set default delimiter
        if ext == ".csv":
            delimiter = delimiter or ","
        elif ext == ".txt":
            delimiter = delimiter or None  # Default for whitespace-separated files
        else:
            raise ValueError("Unsupported file format. Only .csv and .txt files are supported.")

        try:
            # Load the adjacency matrix
            adjacency_matrix = np.loadtxt(filepath, delimiter=delimiter)

            # Validate that the adjacency matrix is square
            if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
                raise ValueError("The adjacency matrix must be square.")

            # Create the graph from the adjacency matrix
            G = nx.from_numpy_array(adjacency_matrix, create_using=type(self))
            self.clear()  # Clear any existing edges/nodes in the current graph
            self.add_edges_from(G.edges)

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' does not exist.")
        except Exception as e:
            raise Exception(f"Error reading adjacency matrix from '{filepath}': {e}")

    def get_adjacency_matrix(self, as_numpy_array=True):
        r"""
        Returns the adjacency matrix of the graph.

        Parameters
        ----------
        as_numpy_array : bool, optional
            If True (default), returns the adjacency matrix as a NumPy array.
            If False, returns the adjacency matrix as a SciPy sparse matrix.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            The adjacency matrix of the graph.

        Examples
        --------
        >>> import graphcalc as gc
        >>> G = gc.SimpleGraph()
        >>> G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        >>> adjacency_matrix = G.get_adjacency_matrix()
        >>> print(adjacency_matrix)
        [[0. 1. 0. 0.]
         [1. 0. 1. 0.]
         [0. 1. 0. 1.]
         [0. 0. 1. 0.]]
        """
        if as_numpy_array:
            return nx.to_numpy_array(self)
        else:
            return nx.to_scipy_sparse_matrix(self)

    def draw(self, with_labels=True, node_color="lightblue", node_size=500, font_size=10):
        r"""
        Draw the graph using Matplotlib.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display node labels (default is True).
        node_color : str or list, optional
            The color of the nodes (default is "lightblue").
        node_size : int, optional
            The size of the nodes (default is 500).
        font_size : int, optional
            The font size of the labels (default is 10).
        """
        plt.figure(figsize=(8, 6))
        nx.draw(
            self,
            with_labels=with_labels,
            node_color=node_color,
            node_size=node_size,
            font_size=font_size,
            edge_color="gray"
        )
        if self.name:
            plt.title(self.name, fontsize=14)
        plt.show()

    def __repr__(self):
        """
        String representation of the SimpleGraph.

        Returns
        -------
        str
            A string summarizing the graph's name, information, and basic properties.
        """
        description = super().__repr__()
        metadata = f"Name: {self.name}" if self.name else "No Name"
        info = f"Info: {self.info}" if self.info else "No Additional Information"
        return f"{description}\n{metadata}\n{info}"

    def complement(self):
        r"""
        Returns the complement of the graph as a GraphCalc SimpleGraph.

        This ensures that constraints specific to SimpleGraph or its subclasses
        are not applied to the complement graph.

        Returns
        -------
        graphcalc.core.basics.SimpleGraph
            The complement of the graph.

        Examples
        --------
        >>> import graphcalc as gc
        >>> G = gc.SimpleGraph()
        >>> G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        >>> H = G.complement()
        """
        H = nx.complement(nx.Graph(self))
        return SimpleGraph(edges=H.edges, nodes=H.nodes, name=f"{self.name} Complement")


from typing import Union
GraphLike = Union[nx.Graph, SimpleGraph]

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def order(G: GraphLike) -> int:
    r"""
    Returns the order of a graph, which is the number of vertices.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The order of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph
    >>> G = path_graph(4)
    >>> gc.order(G)
    4
    """
    return len(G.nodes)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def size(G: GraphLike) -> int:
    r"""
    Returns the size of a graph, which is the number of edges.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The size of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.size(G)
    3
    """
    return len(G.edges)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected(G: GraphLike) -> bool:
    r"""
    Checks if the graph is connected.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.connected(G)
    True
    """
    return nx.is_connected(G)

def bipartite(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and bipartite.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is bipartite, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = nx.path_graph(4)
    >>> gc.bipartite(G)
    True
    """
    return nx.is_bipartite(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected_and_bipartite(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and bipartite.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and bipartite, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = nx.path_graph(4)
    >>> gc.connected_and_bipartite(G)
    True
    """
    return connected(G) and bipartite(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def tree(G: GraphLike) -> bool:
    r"""
    Checks if the graph is a tree.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a tree, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = nx.path_graph(4)
    >>> gc.tree(G)
    True
    """
    return nx.is_tree(G)

def regular(G: GraphLike) -> bool:
    r"""
    Checks if the graph is regular.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is regular, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.regular(G)
    True
    """
    return nx.is_regular(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected_and_regular(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and regular.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and regular, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.connected_and_regular(G)
    True
    """
    return connected(G) and regular(G)

def eulerian(G: GraphLike) -> bool:
    r"""
    Checks if the graph is Eulerian.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is Eulerian, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.eulerian(G)
    True
    """
    return nx.is_eulerian(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected_and_eulerian(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and Eulerian.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and Eulerian, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.connected_and_eulerian(G)
    True
    """
    return connected(G) and eulerian(G)

def planar(G: GraphLike) -> bool:
    """
    Determine whether a graph is planar.

    A graph is planar if it can be drawn in the plane without
    any edges crossing. This function uses the Boyer–Myrvold
    planarity test implemented in NetworkX.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph.

    Returns
    -------
    bool
        True if the graph is planar, False otherwise.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph, complete_graph

    >>> G = cycle_graph(5)
    >>> gc.planar(G)
    True

    >>> H = complete_graph(5)
    >>> gc.planar(H)
    False
    """
    is_planar, _ = nx.check_planarity(G)
    return is_planar

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected_and_planar(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and planar.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and planar, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = cycle_graph(4)
    >>> gc.connected_and_planar(G)
    True
    """
    return connected(G) and planar(G)

def chordal(G: GraphLike) -> bool:
    r"""
    Checks if the graph is chordal.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is chordal, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.chordal(G)
    True
    """
    return nx.is_chordal(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected_and_chordal(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and chordal.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and chordal, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.connected_and_chordal(G)
    True
    """
    return connected(G) and chordal(G)

def cubic(G: GraphLike) -> bool:
    r"""
    Checks if the graph is cubic.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is cubic, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import petersen_graph

    >>> G = petersen_graph()
    >>> gc.cubic(G)
    True
    """
    return gc.maximum_degree(G) == gc.minimum_degree(G) == 3

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected_and_cubic(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and cubic.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and cubic, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import petersen_graph

    >>> G = petersen_graph()
    >>> gc.connected_and_cubic(G)
    True
    """
    return connected(G) and cubic(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def subcubic(G: GraphLike) -> bool:
    r"""
    Checks if the graph is subcubic.

    A graph is subcubic if the degree of every vertex is at most 3.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is subcubic, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph

    >>> G = nx.cycle_graph(4)  # Degree of all nodes is 2
    >>> gc.subcubic(G)
    True
    """
    return gc.maximum_degree(G) <= 3

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def connected_and_subcubic(G: GraphLike) -> bool:
    r"""
    Checks if the graph is both connected and subcubic.

    A graph is subcubic if the degree of every vertex is at most 3.
    A graph is connected if there is a path between every pair of vertices.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is connected and subcubic, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph, path_graph, star_graph

    >>> G = cycle_graph(4)  # Degree of all nodes is 2, connected
    >>> gc.connected_and_subcubic(G)
    True

    >>> H = path_graph(5)  # Maximum degree is 2, connected
    >>> gc.connected_and_subcubic(H)
    True

    >>> I = star_graph(4)  # Maximum degree is 4, not subcubic
    >>> gc.connected_and_subcubic(I)
    False

    >>> J = gc.SimpleGraph()
    >>> J.add_edges_from([(1, 2), (3, 4)])  # Disconnected graph
    >>> connected_and_subcubic(J)
    False
    """
    return connected(G) and subcubic(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def claw_free(G: GraphLike) -> bool:
    r"""
    Checks if a graph is claw-free. A claw is a tree with three leaves adjacent to a single vertex.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is claw-free, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.claw_free(G)
    True
    """
    claw = nx.star_graph(3)
    for S in set(itertools.combinations(G.nodes(), 3)):
        H = G.subgraph(list(S))
        if nx.is_isomorphic(H, claw):
            return False
    # if the above loop completes, the graph is claw-free
    return True

def connected_and_claw_free(G: GraphLike) -> bool:
    r"""
    Checks if a graph is claw-free. A claw is a tree with three leaves adjacent to a single vertex.
    A graph is connected if there is a path between every pair of vertices.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    bool
        True if the graph is claw-free, otherwise False.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.connected_and_claw_free(G)
    True
    """
    return connected(G) and claw_free(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def K_4_free(G: GraphLike) -> bool:
    r"""
    Returns True if *G* does not contain an induced subgraph isomorphic to the complete graph on 4 vertices, and False otherwise.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    boolean
        True if G does not contain the complete graph K_4 as a subgraph, False otherwise.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.K_4_free(G)
    False
    """
    K_4 = nx.complete_graph(4)
    for S in set(itertools.combinations(G.nodes(), 4)):
        H = G.subgraph(list(S))
        if nx.is_isomorphic(H, K_4):
            return False
    return True

def connected_and_K_4_free(G: GraphLike) -> bool:
    r"""
    Returns True if *G* is connected and does not contain an induced subgraph isomorphic to the complete graph on 4 vertices, and False otherwise.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    boolean
        True if G is connected and does not contain the complete graph K_4 as a subgraph, False otherwise.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.connected_and_K_4_free(G)
    False
    """
    return connected(G) and K_4_free(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def triangle_free(G: GraphLike) -> bool:
    r"""Returns True if *G* is triangle-free, and False otherwise.

    A graph is *triangle-free* if it contains no induced subgraph isomorphic to
    the complete graph on 3 vertices.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    boolean
        True if G is triangle-free, False otherwise.

    Examples
    --------
    >>> import graphcalc as gp
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.triangle_free(G)
    False
    """
    # define a triangle graph, also known as the complete graph K_3
    triangle = nx.complete_graph(3)

    # enumerate over all possible combinations of 3 vertices contained in G
    for S in set(itertools.combinations(G.nodes(), 3)):
        H = G.subgraph(list(S))
        if nx.is_isomorphic(H, triangle):
            return False
    # if the above loop completes, the graph is triangle free
    return True

def connected_and_triangle_free(G: GraphLike) -> bool:
    r"""
    Returns True if *G* is connected and triangle-free, and False otherwise.

    A graph is *triangle-free* if it contains no induced subgraph isomorphic to
    the complete graph on 3 vertices.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    boolean
        True if G is connected and triangle-free, False otherwise.

    Examples
    --------
    >>> import graphcalc as gp
    >>> from graphcalc.generators import complete_graph

    >>> G = complete_graph(4)
    >>> gc.connected_and_triangle_free(G)
    False
    """
    return connected(G) and triangle_free(G)

#@enforce_type(0, (networkx.Graph or graphcalc.SimpleGraph, gc.SimpleGraph))
def diameter(G: GraphLike) -> int:
    r"""
    Returns the diameter of the graph.

    The diameter is the maximum shortest path length between any pair of nodes.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The diameter of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.diameter(G)
    3
    """
    return nx.diameter(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def radius(G: GraphLike) -> int:
    r"""
    Returns the radius of the graph.

    The radius is the minimum eccentricity among all vertices.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    int
        The radius of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.radius(G)
    2
    """
    return nx.radius(G)

#@enforce_type(0, (nx.Graph, gc.SimpleGraph))
def average_shortest_path_length(G: GraphLike) -> float:
    r"""
    Returns the average shortest path length of the graph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        The input graph.

    Returns
    -------
    float
        The average shortest path length of the graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph

    >>> G = path_graph(4)
    >>> gc.average_shortest_path_length(G)
    1.6666666666666667
    """
    return nx.average_shortest_path_length(G)

def cograph(G: nx.Graph) -> bool:
    """
    Determine whether a graph is a cograph (P4-free).

    A cograph is any graph that contains no induced path on four vertices (P4).
    Equivalently, the class is generated from K1 by repeatedly taking disjoint
    unions and joins; or, recursively: a graph is a cograph iff it has at most
    one vertex, or it is disconnected and each connected component is a cograph,
    or its complement is disconnected and each complement-component induces
    a cograph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph (not necessarily connected).

    Returns
    -------
    bool
        True if the graph is a cograph (P4-free), False otherwise.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, complete_graph, cycle_graph

    # P4 is not a cograph
    >>> gc.cograph(path_graph(4))
    False

    # Complete graphs are cographs
    >>> gc.cograph(complete_graph(5))
    True

    # C4 is P4-free, hence a cograph
    >>> gc.cograph(cycle_graph(4))
    True
    """
    n = G.number_of_nodes()
    if n <= 1:
        return True

    # Case 1: G is disconnected -> all components must be cographs
    try:
        connected = nx.is_connected(G)
    except nx.NetworkXPointlessConcept:
        # empty graph: treat as cograph by convention
        return True

    if not connected:
        for comp in nx.connected_components(G):
            if not cograph(G.subgraph(comp).copy()):
                return False
        return True

    # Case 2: complement is disconnected -> all complement-components (as
    # node sets) must induce cographs in the ORIGINAL graph
    H = nx.complement(G)
    if not nx.is_connected(H):
        for comp in nx.connected_components(H):
            if not cograph(G.subgraph(comp).copy()):
                return False
        return True

    # Otherwise, both G and its complement are connected and |V|>1 -> not a cograph
    return False

def connected_and_cograph(G: nx.Graph) -> bool:
    """
    Determine whether a graph is connacter and a cograph (P4-free).

    A cograph is any graph that contains no induced path on four vertices (P4).
    Equivalently, the class is generated from K1 by repeatedly taking disjoint
    unions and joins; or, recursively: a graph is a cograph iff it has at most
    one vertex, or it is disconnected and each connected component is a cograph,
    or its complement is disconnected and each complement-component induces
    a cograph.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected graph (not necessarily connected).

    Returns
    -------
    bool
        True if the graph is connected and a cograph (P4-free), False otherwise.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, complete_graph, cycle_graph

    # P4 is not a cograph
    >>> gc.connected_and_cograph(path_graph(4))
    False

    # Complete graphs are cographs
    >>> gc.connected_and_cograph(complete_graph(5))
    True

    # C4 is P4-free, hence a cograph
    >>> gc.connected_and_cograph(cycle_graph(4))
    True
    """
    return connected(G) and cograph(G)
