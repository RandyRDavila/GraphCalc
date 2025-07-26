r"""
Core Module
===========

The `core` module provides foundational utilities and functions for working with graphs in the `graphcalc` package.
It includes basic graph operations and properties that form the building blocks for higher-level functionality in
other submodules.

Key Features
------------
- **Basic Graph Operations**: Functions for computing the size, order, and connectivity of graphs.
- **Neighborhood Analysis**: Functions for analyzing neighborhoods and local properties of nodes in a graph.

Examples
--------
>>> from graphcalc.core import *
>>> G = SimpleGraph()
>>> G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
>>> size(G)
6
>>> order(G)
6
"""
from graphcalc.core.basics import *
from graphcalc.core.neighborhoods import *
