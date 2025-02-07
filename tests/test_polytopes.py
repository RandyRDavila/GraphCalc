import pytest
import graphcalc as gc
from graphcalc.polytopes import (
    PolytopeGraph,
    SimplePolytopeGraph,
)
from graphcalc.polytopes.invariants import (
    p_vector,
    p_gons,
)
from graphcalc.polytopes.generators import (
    cube_graph,
    octahedron_graph,
    dodecahedron_graph,
    tetrahedron_graph,
    icosahedron_graph,
)

# Test the polytope generators
@pytest.mark.parametrize("generator, expected_nodes, expected_edges", [
    (cube_graph, 8, 12),           # Cube: 8 vertices, 12 edges
    (octahedron_graph, 6, 12),     # Octahedron: 6 vertices, 12 edges
    (dodecahedron_graph, 20, 30),  # Dodecahedron: 20 vertices, 30 edges
    (tetrahedron_graph, 4, 6),     # Tetrahedron: 4 vertices, 6 edges
    (icosahedron_graph, 12, 30),   # Icosahedron: 12 vertices, 30 edges
])
def test_polytope_generators(generator, expected_nodes, expected_edges):
    """
    Test that polytope generators create graphs with the correct number of nodes and edges.
    """
    G = generator()

    assert gc.order(G) == expected_nodes, f"{generator.__name__} failed: Incorrect number of nodes."
    assert gc.size(G) == expected_edges, f"{generator.__name__} failed: Incorrect number of edges."


# Test the p-vector function
@pytest.mark.parametrize("generator, expected_p_vector", [
    (cube_graph, [0, 6]),           # Cube: 6 square faces
    (octahedron_graph, [8]),     # Octahedron: 8 triangular faces
    (dodecahedron_graph, [0, 0, 12]),  # Dodecahedron: 12 pentagonal faces
    (tetrahedron_graph, [4]),    # Tetrahedron: 4 triangular faces
    (icosahedron_graph, [20]),   # Icosahedron: 20 triangular faces
])
def test_p_vector(generator, expected_p_vector):
    """
    Test the p_vector function for correctly computing face counts.
    """
    G = generator()
    assert p_vector(G) == expected_p_vector, f"{generator.__name__} failed: Incorrect p-vector."


# Test the p-gons function
@pytest.mark.parametrize("generator, k, expected_count", [
    (cube_graph, 4, 6),              # Cube: 6 square (4-sided) faces
    (octahedron_graph, 3, 8),        # Octahedron: 8 triangular (3-sided) faces
    (dodecahedron_graph, 5, 12),     # Dodecahedron: 12 pentagonal (5-sided) faces
    (tetrahedron_graph, 3, 4),       # Tetrahedron: 4 triangular (3-sided) faces
    (icosahedron_graph, 3, 20),      # Icosahedron: 20 triangular (3-sided) faces
])
def test_p_gons(generator, k, expected_count):
    """
    Test the p_gons function for correctly counting specific face types.
    """
    G = generator()
    assert p_gons(G, p=k) == expected_count, f"{generator.__name__} failed: Incorrect p-gon count for p={p}."


# Test for invalid inputs
def test_invalid_polytope_graph_class_input():
    """
    Ensure PolytopeGraph raises an error for non-planar or invalid graphs.
    """

    # Edge list for K5: non-planar non-polytope graph
    edge_list = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]

    # Expecting a ValueError when initializing a non-polytope graph
    with pytest.raises(ValueError, match=r"The graph is not a valid polytope graph \(simple, planar, and 3-connected\)\."):
        PolytopeGraph(edges=edge_list)


def test_invalid_simple_polytope_graph_class_input():
    """
    Ensure SimplePolytopeGraph raises an error for non-simple polytope graphs.
    """
    edge_list = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
    ]

    with pytest.raises(ValueError, match=r"The graph is not 3-regular, hence not a valid SimplePolytopeGraph."):
        SimplePolytopeGraph(edges=edge_list)
