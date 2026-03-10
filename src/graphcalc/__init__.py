r"""
GraphCalc: A Python Library for Graph Theory, Additive Combinatorics, Hypergraph Theory, and Quantum Information
===============================================================================================================

GraphCalc is a comprehensive Python library for generating, analyzing, and computing properties of
discrete mathematical structures arising in graph theory, additive combinatorics, hypergraph theory,
and quantum information. It provides tools for computation, experimentation, and research across these
areas, with support for both structural invariants and specialized mathematical constructions.

Key Features
------------
- **Graph Theory Utilities**: Compute fundamental graph properties such as size, order, connectivity,
  and a wide range of graph invariants.
- **Graph Generators**: Create simple graphs, random graphs, and polytope-specific graphs.
- **Additive Combinatorics Tools**: Work with additive sets, sumsets, difference sets, stabilizers,
  additive energy, and related invariants and predicates.
- **Hypergraph Theory Tools**: Analyze and compute properties of hypergraphs and associated combinatorial structures.
- **Quantum Information Tools**: Study mathematical structures motivated by quantum information and related invariants.
- **Polytope-Specific Tools**: Analyze graphs of polyhedra with specialized generators and invariants.

Submodules
----------
- `core`: Provides foundational graph utilities and neighborhood analysis.
- `data`: Tools for generating and managing graph datasets.
- `generators`: Functions for creating general and polytope-specific graphs.
- `invariants`: Compute various graph invariants, including degree and spectral properties.
- `polytopes`: Tools for analyzing and generating polytope graphs.
- `additive_combinatorics`: Tools for additive sets, sumsets, and additive combinatorial invariants.
- `hypergraphs`: Tools for hypergraph construction and analysis.
- `quantum`: Tools related to quantum information and associated discrete structures.

Dependencies
------------
GraphCalc relies on:
- `networkx`: For graph and combinatorial structure representation.
- `numpy`: For numerical computations.
- `matplotlib`: For visualization.
- `pandas`: For data handling and analysis.
- `PuLP`: For optimization-based computations.

"""

__version__ = "2.0.0"
