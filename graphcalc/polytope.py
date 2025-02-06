import networkx as nx
# import numpy as np

__all__ = ['p_vector', 'p_gons', 'fullerene']

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


def p_gons(graph, p=3):
    r"""
    Compute the number of p-sided faces in a planar graph.

    This function determines the count of faces with exactly `p` sides in a given planar graph
    by leveraging the p-vector. The graph must be planar and connected.

    Parameters
    ----------
    graph : networkx.Graph
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
    p_vector = p_vector(graph)
    return p_vector[p - 3] if p - 3 < len(p_vector) else 0

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
    p_vector = p_vector(G)

    # Ensure there are exactly 12 pentagonal faces
    if len(p_vector) < 1 or p_vector[0] != 12:
        return False

    # Ensure all other faces are hexagonal
    if any(p_vector[i] != 0 for i in range(1, len(p_vector) - 1)):
        return False

    return True
