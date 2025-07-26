import networkx as nx
import graphcalc as gc
import matplotlib.pyplot as plt

__all__ = [
    'PolytopeGraph',
    'SimplePolytopeGraph',
]


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
    draw()
        Draws the graph using a planar layout with Matplotlib.
    read_edge_list(filepath, delimiter=None)
        Reads an edge list from a file and validates the graph as a polytope graph.
    read_adjacency_matrix(filepath, delimiter=None)
        Reads an adjacency matrix from a file and validates the graph as a polytope graph.
    is_simple()
        Checks if the graph is a simple polytope graph.

    Examples
    --------
    >>> import networkx as nx
    >>> from graphcalc.polytopes import PolytopeGraph
    >>> G = nx.cubical_graph()
    >>> polytope = PolytopeGraph(edges=G.edges, nodes=G.nodes, name="Cube Graph")
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

    def _is_planar(self):
        r"""
        Check if the graph is planar.

        Returns
        -------
        bool
            True if the graph is planar, False otherwise.
        """
        is_planar, _ = nx.check_planarity(self)
        return is_planar

    def _is_3_connected(self):
        r"""
        Check if the graph is 3-connected.

        Returns
        -------
        bool
            True if the graph is 3-connected, False otherwise.
        """
        return nx.is_connected(self) and nx.node_connectivity(self) >= 3

    def is_polytope_graph(self):
        r"""
        Check if the graph satisfies the polytope graph conditions.

        Returns
        -------
        bool
            True if the graph is a valid polytope graph, False otherwise.

        Examples
        --------
        >>> import networkx as nx
        >>> from graphcalc.polytopes import PolytopeGraph
        >>> G = nx.cubical_graph()

        >>> polytope = PolytopeGraph(edges=G.edges, nodes=G.nodes, name="Cube Graph")
        >>> polytope.is_polytope_graph()
        True
        """
        return self._is_planar() and self._is_3_connected()

    def is_simple(self):
        r"""
        Check if the graph is of a simple polytope, i.e., a 3-regular, 3-connected,
        and planar graph.

        Returns
        -------
        bool
            True if the graph is simple, False otherwise.

        Examples
        --------
        >>> import networkx as nx
        >>> from graphcalc.polytopes import PolytopeGraph
        >>> G = nx.cubical_graph()

        >>> polytope = PolytopeGraph(edges=G.edges, nodes=G.nodes, name="Cube Graph")
        >>> polytope.is_simple()
        True
        """
        return gc.connected_and_cubic(self) and self.is_polytope_graph()


    def draw(self, with_labels=True, node_color="lightblue", node_size=500, font_size=10):
        r"""
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
        >>> import networkx as nx
        >>> from graphcalc.polytopes import PolytopeGraph
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
        r"""
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

        Examples
        --------
        >>> from graphcalc.polytopes import PolytopeGraph
        >>> polytope = PolytopeGraph()
        >>> polytope.read_edge_list("polytope_edges.txt")
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

        Examples
        --------
        >>> from graphcalc.polytopes import PolytopeGraph
        >>> polytope = PolytopeGraph()
        >>> polytope.read_adjacency_matrix("polytope_adjacency.txt")
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

        Examples
        --------
        >>> from graphcalc.polytopes import SimplePolytopeGraph
        >>> G = nx.cubical_graph()
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
        r"""
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

        Examples
        --------
        >>> from graphcalc.polytopes import SimplePolytopeGraph
        >>> polytope = SimplePolytopeGraph()
        >>> polytope.read_edge_list("simple_polytope_edges.txt")
        """
        super().read_edge_list(filepath, delimiter)
        if not self.is_3_regular():
            raise ValueError("The graph read from the file is not a valid simple polytope graph (3-regular).")


    def read_adjacency_matrix(self, filepath, delimiter=None):
        r"""
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

        Examples
        --------
        >>> from graphcalc.polytopes import SimplePolytopeGraph
        >>> polytope = SimplePolytopeGraph()
        >>> polytope.read_adjacency_matrix("simple_polytope_adjacency.txt")
        """
        super().read_adjacency_matrix(filepath, delimiter)
        if not self.is_3_regular():
            raise ValueError("The graph read from the adjacency matrix is not a valid simple polytope graph (3-regular).")
