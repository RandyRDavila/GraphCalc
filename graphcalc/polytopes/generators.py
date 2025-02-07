r"""
Polytope graph generators.

This module provides generators for graphs corresponding to polytopes.
These generators must be explicitly imported.

Examples
--------
>>> from graphcalc.generators.polytopes import cube_graph, octahedron_graph
>>> G = cube_graph()
>>> H = octahedron_graph()
"""

import graphcalc as gc
import networkx as nx

__all__ = [
    "cube_graph",
    "octahedron_graph",
    "dodecahedron_graph",
    "tetrahedron_graph",
    "icosahedron_graph",
    "convex_polytopes_text_example",
]

def cube_graph():
    r"""
    Generate the graph of a cube.

    Returns
    -------
    SimplePolytopeGraph
        The graph of a cube (3-regular polytope).
    """
    edges = nx.cubical_graph().edges
    return gc.SimplePolytopeGraph(edges=edges, name="Cube Graph")


def octahedron_graph():
    r"""
    Generate the graph of an octahedron.

    Returns
    -------
    PolytopeGraph
        The graph of an octahedron (planar, simple, and 3-connected).
    """
    return gc.PolytopeGraph(edges=nx.octahedral_graph().edges, name="Octahedron Graph")


def dodecahedron_graph():
    r"""
    Generate the graph of a dodecahedron.

    Returns
    -------
    SimplePolytopeGraph
        The graph of a dodecahedron (3-regular polytope).
    """
    return gc.SimplePolytopeGraph(edges=nx.dodecahedral_graph().edges, name="Dodecahedron Graph")


def tetrahedron_graph():
    r"""
    Generate the graph of a tetrahedron.

    Returns
    -------
    PolytopeGraph
        The graph of a tetrahedron (planar, simple, and 3-connected).
    """
    return gc.PolytopeGraph(edges=nx.tetrahedral_graph().edges, name="Tetrahedron Graph")


def icosahedron_graph():
    """
    Generate the graph of an icosahedron.

    Returns
    -------
    PolytopeGraph
        The graph of an icosahedron (planar, simple, and 3-connected).
    """
    return gc.PolytopeGraph(edges=nx.icosahedral_graph().edges, name="Icosahedron Graph")


def convex_polytopes_text_example(n=1):
    """
    Generate a polytope graph from the first predefined edge list.

    Returns
    -------
    PolytopeGraph
        A polytope graph constructed from edge_list_1.
    """
    if n == 1:
        edge_list = [
            (0, 1), (0, 5), (0, 6),
            (1, 2), (1, 7),
            (2, 3), (2, 8),
            (3, 4), (3, 9),
            (4, 5), (4, 10),
            (5, 14),
            (6, 7), (6, 12),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (11, 15),
            (12, 13),
            (13, 14), (13, 15),
            (14, 15)
        ]
    elif n == 2:
        edge_list = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 6), (2, 7),
        (3, 8), (3, 9),
        (4, 5),
        (4, 10),
        (5, 11),
        (6, 12), (6, 13),
        (7, 10), (7, 12),
        (8, 9), (8, 13),
        (9, 11),
        (10, 11),
        (12, 13)
    ]
    return gc.PolytopeGraph(edges=edge_list, name="Polytope from Edge List 1")
