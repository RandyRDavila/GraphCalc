r"""
GraphCalc: A Python Library for Graph Analysis and Generation
=============================================================

GraphCalc is a comprehensive library for generating, analyzing, and computing properties of graphs.
It provides tools for working with general graphs, polytopes, and graph invariants, making it ideal
for research, teaching, and experimentation in graph theory.

Key Features
------------
- **Core Utilities**: Compute fundamental graph properties such as size, order, and connectivity.
- **Graph Generators**: Create simple graphs, random graphs, and polytope-specific graphs.
- **Graph Invariants**: Compute degree-based, spectral, domination, and zero-forcing invariants.
- **Polytope-Specific Tools**: Analyze graphs of polyhedra with specialized generators and invariants.

Submodules
----------
- `core`: Provides foundational graph utilities and neighborhood analysis.
- `data`: Tools for generating and managing graph datasets.
- `generators`: Functions for creating general and polytope-specific graphs.
- `invariants`: Compute various graph invariants, including degree and spectral properties.
- `polytopes`: Tools for analyzing and generating polytope graphs.

Examples
--------
>>> import graphcalc as gc
>>> G = gc.generators.simple.cycle_graph(6)  # Create a cycle graph with 6 nodes
>>> gc.core.size(G)
6
>>> gc.invariants.degree.degree_sequence(G)
[2, 2, 2, 2, 2, 2]

Dependencies
------------
GraphCalc relies on:
- `networkx`: For graph representation and manipulation.
- `numpy`: For numerical computations.
- `matplotlib`: For graph visualization.

"""

__version__ = "0.1.20"

from graphcalc.core import *
from graphcalc.data import *
from graphcalc.generators import *
from graphcalc.invariants import *
from graphcalc.polytopes import *
