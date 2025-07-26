r"""
Graph Invariants
================

The `invariants` module provides functions to compute various graph invariants, which are properties of graphs
that remain unchanged under graph isomorphism. These include degree-based, spectral, domination, and zero-forcing
invariants.

Key Features
------------
- **Degree-Based Invariants**: Analyze degree sequences and related properties.
- **Spectral Properties**: Compute eigenvalues and other properties of adjacency and Laplacian matrices.
- **Domination**: Evaluate domination-related graph properties.
- **Zero-Forcing**: Compute zero-forcing sets and associated invariants.

Submodules
----------
- `degree`: Functions related to degree sequences and degree-based invariants.
- `spectral`: Functions for spectral graph theory.
- `domination`: Functions for domination properties.
- `zero_forcing`: Functions for zero-forcing invariants.

Examples
--------
>>> from graphcalc.invariants import degree_sequence
>>> from graphcalc.generators.simple import cycle_graph
>>> G = cycle_graph(5)
>>> degree_sequence(G)
[2, 2, 2, 2, 2]

Dependencies
------------
This module relies on:
- `networkx`: For graph representation and algorithms.
- `numpy`: For numerical computations.
"""

from graphcalc.invariants.classics import *
from graphcalc.invariants.degree import *
from graphcalc.invariants.domination import *
from graphcalc.invariants.spectral import *
from graphcalc.invariants.zero_forcing import *
