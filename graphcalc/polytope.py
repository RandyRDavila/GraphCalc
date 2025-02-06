import networkx as nx
import graphcalc as gc
import matplotlib.pyplot as plt
# import numpy as np

__all__ = [
    'p_vector',
    'p_gons',
    'fullerene',
    'simple_graph',
    'polytope_graph',
    'simple_polytope_graph',
    'polytope_graph_with_p6_zero',
    'simple_polytope_graph_with_p6_zero',
    'polytope_graph_with_p6_greater_than_zero',
    'simple_polytope_graph_with_p6_greater_than_zero',
    'PolytopeGraph',
    'SimplePolytopeGraph',
]

def p_vector(G_nx):
    r"""
    Compute the p-vector of a planar graph.

    The p-vector of a graph is a list where the i-th entry represents the count of i-sided faces
    (e.g., triangles, quadrilaterals, pentagons) in a planar embedding of the graph. The function
    assumes the input graph is planar and connected.

    Parameters
    ----------
    G_nx : networkx.Graph
        A planar graph for which the p-vector is computed.

    Returns
    -------
    list of int
        The p-vector, where the value at index `k-3` corresponds to the number of k-sided faces in the graph.

    Notes
    -----
    - This function first checks the planarity of the input graph using NetworkX's `check_planarity`.
    - If the graph is not planar, a `ValueError` is raised.

    Examples
    --------
    Compute the p-vector of a simple planar graph:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(6)  # Hexagon
    >>> gc.p_vector(G)
    [0, 1]  # One hexagonal face and no smaller faces

    Compute the p-vector of a graph with multiple face sizes:

    >>> G = nx.Graph()
    >>> G.add_edges_from([
    ...     (0, 1), (1, 2), (2, 3), (3, 0),  # Quadrilateral face
    ...     (0, 4), (4, 1),  # Two triangular faces
    ...     (1, 5), (5, 2)
    ... ])
    >>> gc.p_vector(G)
    [2, 1]  # Two triangles and one quadrilateral
    """
    # Ensure the graph is labeled with consecutive integers
    G_nx = nx.convert_node_labels_to_integers(G_nx)
    graph = nx.to_numpy_array(G_nx, dtype=int)


    # Dictionary to store the count of faces by their number of sides
    num_i_sides = {}

    # Check if the graph is planar and obtain its planar embedding
    is_planar, embedding_nx = nx.check_planarity(G_nx)
    if not is_planar:
        raise ValueError("The input graph is not planar.")

    # Initialize vertex elements list
    vert_elms = list(range(1, len(graph[0]) + 1))

    # Initialize edge elements and relations
    edge_elms = []
    edge_dict = {}
    relations = []

    # Construct edges and their relationships
    for vert in vert_elms:
        vert_mat_index = vert - 1
        neighbors = [j + 1 for j in range(len(graph[0])) if graph[vert_mat_index][j] == 1]

        for buddy in neighbors:
            if vert < buddy:
                new_edge = edge_elms[-1] + 1 if edge_elms else vert_elms[-1] + 1
                edge_elms.append(new_edge)
                edge_dict[new_edge] = [vert, buddy]
                relations.extend([[vert, new_edge], [buddy, new_edge]])

    # Initialize face elements and relations
    face_elms = []
    face_dict = {}

    # Construct faces using planar embedding
    for edge, (v1, v2) in edge_dict.items():
        for face_vertices in [embedding_nx.traverse_face(v=v1-1, w=v2-1), embedding_nx.traverse_face(v=v2-1, w=v1-1)]:
            face_vertices = list(face_vertices)
            if not any(sorted(face_vertices) == sorted(existing) for existing in face_dict.values()):
                new_face = face_elms[-1] + 1 if face_elms else edge_elms[-1] + 1
                face_elms.append(new_face)
                face_dict[new_face] = face_vertices
                relations.append([edge, new_face])

    # Count faces by size
    for face_vertices in face_dict.values():
        num_i_sides[len(face_vertices)] = num_i_sides.get(len(face_vertices), 0) + 1

    # Construct p-vector
    max_face_size = max(num_i_sides.keys(), default=2)
    p_k_vec = [num_i_sides.get(j, 0) for j in range(3, max_face_size + 1)]

    return p_k_vec


def p_gons(G, p=3):
    r"""
    Compute the number of p-sided faces in a planar graph.

    This function determines the count of faces with exactly `p` sides in a given planar graph
    by leveraging the p-vector. The graph must be planar and connected.

    Parameters
    ----------
    G : networkx.Graph
        A planar graph for which the count of p-sided faces is computed.
    p : int, optional
        The number of sides of the faces to count. Defaults to 3 (triangular faces).

    Returns
    -------
    int
        The number of p-sided faces in the graph. Returns 0 if no such faces exist.

    Notes
    -----
    - This function assumes the input graph is planar.
    - It internally calls the `p_vector` function to calculate the p-vector of the graph.

    Examples
    --------
    Count the number of triangular faces in a hexagonal graph:

    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.cycle_graph(6)  # Hexagon
    >>> gc.p_gons(G, p=3)
    0  # The hexagon has no triangular faces

    Count the number of hexagonal faces in the same graph:

    >>> gc.p_gons(G, p=6)
    1  # The hexagon has exactly one 6-sided face

    Count the number of pentagonal faces in a graph with multiple face types:

    >>> G = nx.Graph()
    >>> G.add_edges_from([
    ...     (0, 1), (1, 2), (2, 3), (3, 0),  # Quadrilateral face
    ...     (0, 4), (4, 1),  # Two triangular faces
    ...     (1, 5), (5, 2)
    ... ])
    >>> gc.p_gons(G, p=5)
    0  # The graph has no pentagonal faces
    """
    vector = p_vector(G)
    return vector[p - 3] if p - 3 < len(vector) else 0

def fullerene(G):
    r"""
    Determine if a graph is a fullerene.

    Parameters
    ----------
    G : networkx.Graph
        The graph to be checked for fullerene properties.

    Returns
    -------
    bool
        True if the graph is a fullerene, False otherwise.

    Notes
    -----
    This function assumes the graph is simple and connected. It uses the
    `p_vector` function to compute the face structure of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    >>> gc.fullerene(G)
    False
    """
    # Check if the graph is 3-regular
    if not all(degree == 3 for _, degree in G.degree):
        return False

    # Check if the graph is planar
    is_planar, _ = nx.check_planarity(G)
    if not is_planar:
        return False

    # Use the p_vector_graph function to count faces of different sizes
    vector = p_vector(G)

    # Ensure there are exactly 12 pentagonal faces
    if len(vector) < 1 or vector[0] != 12:
        return False

    # Ensure all other faces are hexagonal
    if any(vector[i] != 0 for i in range(1, len(vector) - 1)):
        return False

    return True

def simple_graph(G):
    r"""
    Check if a graph is simple.

    A graph is simple if:
    1. It has no self-loops.
    2. It has no multiple edges.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is simple, False otherwise.
    """
    # Check for self-loops
    if any(G.has_edge(u, u) for u in G.nodes):
        return False

    # Check for multiple edges (only relevant for MultiGraph)
    if isinstance(G, nx.MultiGraph):
        for u, v, count in G.edges(keys=True):
            if G.number_of_edges(u, v) > 1:
                return False

    return True


def polytope_graph(G):
    r"""
    Check if a graph is the graph of a polyhedron.

    A graph is the graph of a polyhedron (or a polytope graph) if and only if it is:
    1. Simple: The graph has no self-loops or multiple edges.
    2. Planar: The graph can be embedded in the plane without edge crossings.
    3. 3-Connected: The graph remains connected after removing any two vertices.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a polytope graph, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.octahedral_graph()  # Octahedral graph is a polytope graph
    >>> polytope_graph(G)
    True

    >>> G = nx.path_graph(5)  # Path graph is not a polytope graph
    >>> polytope_graph(G)
    False
    """
    # 1. Check if the graph is simple
    if not simple_graph(G):
        return False

    # 2. Check if the graph is planar
    is_planar, _ = nx.check_planarity(G)
    if not is_planar:
        return False

    # 3. Check if the graph is 3-connected
    if not nx.is_connected(G) or not nx.node_connectivity(G) >= 3:
        return False

    return True

def simple_polytope_graph(G):
    r"""
    Check if a graph is the graph of a simple polyhedron.

    A graph is the graph of a polyhedron (or a polytope graph) if and only if it is:
    1. Simple: The graph has no self-loops or multiple edges.
    2. Planar: The graph can be embedded in the plane without edge crossings.
    3. 3-Connected: The graph remains connected after removing any two vertices.
    4. 3-Regular: Each vertex has degree 3.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a polytope graph, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.cubical_graph()  # Octahedral graph is a simple polytope graph
    >>> simple_polytope_graph(G)
    True

    >>> G = nx.path_graph(5)  # Path graph is not a simple polytope graph
    >>> simple_polytope_graph(G)
    False
    """
    return simple_graph(G) and polytope_graph(G) and gc.connected_and_cubic(G)

def polytope_graph_with_p6_zero(G):
    r"""
    Check if a graph is the graph of a polyhedron with no hexagonal faces.

    A graph is the graph of a polyhedron (or a polytope graph) if and only if it is:
    1. Simple: The graph has no self-loops or multiple edges.
    2. Planar: The graph can be embedded in the plane without edge crossings.
    3. 3-Connected: The graph remains connected after removing any two vertices.
    4. No hexagonal faces: The graph has no hexagonal faces.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a polytope graph, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.octahedral_graph()  # Octahedral graph is a polytope graph with no hexagonal faces
    >>> polytope_graph_with_p6_zero(G)
    True

    >>> G = nx.cubical_graph()  # Cubical graph is a polytope graph with hexagonal faces
    >>> polytope_graph_with_p6_zero(G)
    False
    """
    return polytope_graph(G) and gc.p_gons(G, p=6) == 0


def simple_polytope_graph_with_p6_zero(G):
    r"""
    Check if a graph is the graph of a simple polyhedron with no hexagonal faces.

    A graph is the graph of a polyhedron (or a polytope graph) if and only if it is:
    1. Simple: The graph has no self-loops or multiple edges.
    2. Planar: The graph can be embedded in the plane without edge crossings.
    3. 3-Connected: The graph remains connected after removing any two vertices.
    4. 3-Regular: Each vertex has degree 3.
    5. No hexagonal faces: The graph has no hexagonal faces.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a polytope graph, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.octahedral_graph()  # Octahedral graph is a simple polytope graph with no hexagonal faces
    >>> simple_polytope_graph_with_p6_zero(G)
    True

    >>> G = nx.cubical_graph()  # Cubical graph is a simple polytope graph with hexagonal faces
    >>> simple_polytope_graph_with_p6_zero(G)
    False
    """
    return simple_polytope_graph(G) and gc.p_gons(G, p=6) == 0

def polytope_graph_with_p6_greater_than_zero(G):
    r"""
    Check if a graph is the graph of a polyhedron with at least one hexagonal face.

    A graph is the graph of a polyhedron (or a polytope graph) if and only if it is:
    1. Simple: The graph has no self-loops or multiple edges.
    2. Planar: The graph can be embedded in the plane without edge crossings.
    3. 3-Connected: The graph remains connected after removing any two vertices.
    4. At least one hexagonal face: The graph has at least one hexagonal face.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a polytope graph, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.cubical_graph()  # Cubical graph is a polytope graph with hexagonal faces
    >>> polytope_graph_with_p6_greater_than_zero(G)
    True

    >>> G = nx.octahedral_graph()  # Octahedral graph is a polytope graph with no hexagonal faces
    >>> polytope_graph_with_p6_greater_than_zero(G)
    False
    """
    return polytope_graph(G) and gc.p_gons(G, p=6) > 0

def simple_polytope_graph_with_p6_greater_than_zero(G):
    r"""
    Check if a graph is the graph of a simple polyhedron with at least one hexagonal face.

    A graph is the graph of a polyhedron (or a polytope graph) if and only if it is:
    1. Simple: The graph has no self-loops or multiple edges.
    2. Planar: The graph can be embedded in the plane without edge crossings.
    3. 3-Connected: The graph remains connected after removing any two vertices.
    4. 3-Regular: Each vertex has degree 3.
    5. At least one hexagonal face: The graph has at least one hexagonal face.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    bool
        True if the graph is a polytope graph, False otherwise.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.cubical_graph()  # Cubical graph is a simple polytope graph with hexagonal faces
    >>> simple_polytope_graph_with_p6_greater_than_zero(G)
    True

    >>> G = nx.octahedral_graph()  # Octahedral graph is a simple polytope graph with no hexagonal faces
    >>> simple_polytope_graph_with_p6_greater_than_zero(G)
    False
    """
    return simple_polytope_graph(G) and gc.p_gons(G, p=6) > 0


class PolytopeGraph(gc.SimpleGraph):
    r"""
    A subclass of SimpleGraph that ensures the graph satisfies polytope graph conditions.

    A polytope graph is defined as a graph that is:
    1. Simple: No self-loops or multiple edges.
    2. Planar: Can be embedded in the plane without edge crossings.
    3. 3-Connected: Remains connected after the removal of any two vertices.

    Methods
    -------
    is_polytope_graph()
        Checks if the graph satisfies the polytope graph conditions.
    """

    def __init__(self, edges=None, nodes=None, name=None, info=None, *args, **kwargs):
        """
        Initialize a PolytopeGraph instance.

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
            Arguments passed to the base `SimpleGraph` class.

        Raises
        ------
        ValueError
            If the initialized graph is not a valid polytope graph and is not empty.
        """
        super().__init__(edges=edges, nodes=nodes, name=name, info=info, *args, **kwargs)

        # Skip validation if the graph is empty
        if len(self.edges) == 0:
            return

        # Validate the graph
        if not self.is_polytope_graph():
            raise ValueError("The graph is not a valid polytope graph (simple, planar, and 3-connected).")

    def is_simple(self):
        """
        Check if the graph is simple.

        Returns
        -------
        bool
            True if the graph is simple, False otherwise.
        """
        if any(self.has_edge(u, u) for u in self.nodes):  # Check for self-loops
            return False
        if isinstance(self, nx.MultiGraph):
            for u, v in self.edges:
                if self.number_of_edges(u, v) > 1:  # Check for multiple edges
                    return False
        return True

    def is_planar(self):
        """
        Check if the graph is planar.

        Returns
        -------
        bool
            True if the graph is planar, False otherwise.
        """
        is_planar, _ = nx.check_planarity(self)
        return is_planar

    def is_3_connected(self):
        """
        Check if the graph is 3-connected.

        Returns
        -------
        bool
            True if the graph is 3-connected, False otherwise.
        """
        return nx.is_connected(self) and nx.node_connectivity(self) >= 3

    def is_polytope_graph(self):
        """
        Check if the graph satisfies the polytope graph conditions.

        Returns
        -------
        bool
            True if the graph is a valid polytope graph, False otherwise.
        """
        return self.is_simple() and self.is_planar() and self.is_3_connected()

    def draw(self, with_labels=True, node_color="lightblue", node_size=500, font_size=10):
        """
        Draw the graph using a planar layout with Matplotlib.

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

        Notes
        -----
        This method always uses a planar layout to ensure no edge crossings.

        Examples
        --------
        >>> G = nx.cubical_graph()
        >>> polytope = PolytopeGraph(edges=G.edges, nodes=G.nodes, name="Cube Graph")
        >>> polytope.draw()
        """
        if not self.is_planar():
            raise ValueError("The graph is not planar and cannot be drawn using a planar layout.")

        # Generate the planar layout
        pos = nx.planar_layout(self)

        # Draw the graph
        plt.figure(figsize=(8, 6))
        nx.draw(
            self,
            pos,
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
        String representation of the PolytopeGraph.

        Returns
        -------
        str
            A string summarizing the graph's name, information, and polytope validity.
        """
        description = super().__repr__()
        validity = "Valid Polytope Graph" if self.is_polytope_graph() else "Invalid Polytope Graph"
        return f"{description}\n{validity}"

    def read_edge_list(self, filepath, delimiter=None):
        """
        Read an edge list from a file (CSV or TXT), add edges to the graph,
        and validate that the graph remains a valid polytope graph.

        Parameters
        ----------
        filepath : str
            The path to the file containing the edge list.
        delimiter : str, optional
            The delimiter used in the file.

        Raises
        ------
        ValueError
            If the graph is not a valid polytope graph after reading the edge list.
        """
        super().read_edge_list(filepath, delimiter)
        if not self.is_polytope_graph():
            raise ValueError("The graph read from the file is not a valid polytope graph.")

    def read_adjacency_matrix(self, filepath, delimiter=None):
        """
        Read an adjacency matrix from a file, create the graph,
        and validate that it remains a valid polytope graph.

        Parameters
        ----------
        filepath : str
            The path to the file containing the adjacency matrix.
        delimiter : str, optional
            The delimiter used in the file.

        Raises
        ------
        ValueError
            If the graph is not a valid polytope graph after reading the adjacency matrix.
        """
        super().read_adjacency_matrix(filepath, delimiter)
        if not self.is_polytope_graph():
            raise ValueError("The graph read from the adjacency matrix is not a valid polytope graph.")


class SimplePolytopeGraph(PolytopeGraph):
    def __init__(self, edges=None, nodes=None, name=None, info=None, *args, **kwargs):
        """
        Initialize a SimplePolytopeGraph instance.

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
            Arguments passed to the base `PolytopeGraph` class.

        Raises
        ------
        ValueError
            If the initialized graph is not a valid simple polytope graph.
        """
        self._bypass_validation = True  # Temporarily bypass validation
        super().__init__(edges=edges, nodes=nodes, name=name, info=info, *args, **kwargs)
        self._bypass_validation = False

        if not self.is_3_regular():
            raise ValueError("The graph is not 3-regular, hence not a valid SimplePolytopeGraph.")

    def is_3_regular(self):
        """
        Check if the graph is 3-regular.

        Returns
        -------
        bool
            True if the graph is 3-regular, False otherwise.
        """
        degrees = [degree for _, degree in self.degree()]
        return all(degree == 3 for degree in degrees)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """
        Add an edge and validate the graph remains a valid simple polytope graph.

        Raises
        ------
        ValueError
            If adding the edge makes the graph invalid as a simple polytope graph.
        """
        super().add_edge(u_of_edge, v_of_edge, **attr)
        if not self._bypass_validation and not self.is_3_regular():
            self.remove_edge(u_of_edge, v_of_edge)  # Revert the addition
            raise ValueError(f"Adding edge ({u_of_edge}, {v_of_edge}) makes the graph invalid as a SimplePolytopeGraph.")

    def add_edges_from(self, ebunch_to_add, **attr):
        """
        Add multiple edges and validate the graph remains a valid simple polytope graph.

        Parameters
        ----------
        ebunch_to_add : iterable of edges
            An iterable of edges to add.
        **attr : keyword arguments
            Additional edge attributes.

        Raises
        ------
        ValueError
            If the graph is not valid after all edges are added.
        """
        self._bypass_validation = True  # Temporarily bypass validation
        super().add_edges_from(ebunch_to_add, **attr)
        self._bypass_validation = False

        if not self.is_3_regular():
            raise ValueError("The graph is not 3-regular, hence not a valid SimplePolytopeGraph.")


    def __repr__(self):
        """
        String representation of the SimplePolytopeGraph.

        Returns
        -------
        str
            A string summarizing the graph's name, information, and validity.
        """
        description = super().__repr__()
        validity = "Valid Simple Polytope Graph" if self.is_3_regular() else "Invalid Simple Polytope Graph"
        return f"{description}\n{validity}"

    def read_edge_list(self, filepath, delimiter=None):
        """
        Read an edge list from a file (CSV or TXT), add edges to the graph,
        and validate that the graph remains a valid simple polytope graph.

        Parameters
        ----------
        filepath : str
            The path to the file containing the edge list.
        delimiter : str, optional
            The delimiter used in the file.

        Raises
        ------
        ValueError
            If the graph is not a valid simple polytope graph after reading the edge list.
        """
        super().read_edge_list(filepath, delimiter)
        if not self.is_3_regular():
            raise ValueError("The graph read from the file is not a valid simple polytope graph (3-regular).")


    def read_adjacency_matrix(self, filepath, delimiter=None):
        """
        Read an adjacency matrix from a file, create the graph,
        and validate that it remains a valid simple polytope graph.

        Parameters
        ----------
        filepath : str
            The path to the file containing the adjacency matrix.
        delimiter : str, optional
            The delimiter used in the file.

        Raises
        ------
        ValueError
            If the graph is not a valid simple polytope graph after reading the adjacency matrix.
        """
        super().read_adjacency_matrix(filepath, delimiter)
        if not self.is_3_regular():
            raise ValueError("The graph read from the adjacency matrix is not a valid simple polytope graph (3-regular).")
