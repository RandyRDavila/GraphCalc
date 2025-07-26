r"""
Polytope Graphs
===============

This module provides tools for working with polytope graphs, including:
- Core utilities for validating polytope graph properties.
- Generators for creating standard polytope graphs like cubes and dodecahedrons.
- Invariants specific to polytope graphs, such as face counts.

Submodules:
-----------
- `core`: Core functionality for polytope graphs.
- `generators`: Functions for generating polytope graphs.
- `invariants`: Functions for computing polytope-specific invariants.
"""

from graphcalc.polytopes.core import *
from graphcalc.polytopes.invariants import *
from graphcalc.polytopes.generators import *
