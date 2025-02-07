r"""
Graph Generators
================

The `generators` module provides functions to generate various types of graphs, ranging from general-purpose
graphs to polytope-specific graphs. This module enables users to create predefined and random graph structures
for testing, experimentation, and research.

Key Features
------------
- **Simple Graphs**: Generate simple graph structures like paths, cycles, and complete graphs.
- **Polytope Graphs**: Generate graphs of polyhedra, such as cubes, tetrahedra, and dodecahedra.
- **Random Graphs**: Create random graphs with specified properties.

Submodules
----------
- `simple`: Functions for generating general graph structures.
- `polytopes`: Functions for generating graphs representing polytopes.

Examples
--------
>>> from graphcalc.generators.simple import cycle_graph
>>> G = cycle_graph(5)
>>> print(G.nodes())
>>> print(G.edges())

Dependencies
------------
This module relies on:
- `networkx`: For graph representation and manipulation.
"""
from graphcalc.generators.simple_graphs import *
