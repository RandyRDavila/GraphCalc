# GraphCalc

[![Documentation Status](https://readthedocs.org/projects/graphcalc/badge/?version=latest)](https://graphcalc.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/graphcalc.svg?ts=20250726)](https://pypi.org/project/graphcalc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/888587732.svg)](https://doi.org/10.5281/zenodo.16907645)

## Overview

`GraphCalc` is a Python library for computing a broad range of graph-theoretic invariants, purpose-built to support research in combinatorics, network science, and automated reasoning. It offers exact implementations of over 100 functions, spanning classical invariants (e.g., independence number, chromatic number, spectral radius) and a wide array of lesser-known parameters central to contemporary graph theory.

Originally developed as the invariant engine for the automated conjecturing system TxGraffiti, `GraphCalc` has since matured into a general-purpose research tool that facilitates the large-scale construction of structured, high-resolution invariant datasets. These datasets, often organized into tabular “knowledge tables,” form the basis for symbolic pattern mining, hypothesis generation, and downstream machine reasoning. For example,

```python
>>> import graphcalc as gc
>>> from graphcalc.polytopes.generators import cube_graph, octahedron_graph
>>> graphs = [cube_graph(), octahedron_graph()]
>>> functions = ["order", "size", "spectral_radius", "independence_number"]
>>> gc.compute_knowledge_table(functions, graphs)
   order  size  spectral_radius  independence_number
0      8    12              3.0                    4
1      6    12              4.0                    2
```

## Features

- **Maximum Clique**: Finds the maximum clique in a given graph.
- **Chromatic Number**: Computes the minimum number of colors required for graph coloring.
- **Vertex and Edge Cover**: Determines vertex and edge covers.
- **Matching and Independence**: Calculates maximum matching and independent sets.
- **Domination Number and its Variants**: Calculates the domination number, total domination number, and many other domination variants.
- **Degree Sequence Invariants**: Calculates the residue, annihilaiton number, the slater number and more!
- **Zero Forcing**: Calculates the zero forcing number, the total zero forcing number, the positive semidefinite zero forcing number, and the power domination number.

## Installation

To install `graphcalc`, make sure you have Python 3.7 or higher, then install it:

```bash
pip install graphcalc
```

## Linear and Integer Programming Solvers

Many of the NP-hard graph invariant computations of GraphCalc depend on third-party solvers.At least one of the following is required if you intend to use solver-based functions (e.g., `gc.maximum_independent_set(G)`):

- **CBC** (recommended):

```bash
brew install cbc      # macOS
sudo apt install coinor-cbc  # Debian/Ubuntu
```

GraphCalc will attempt to automatically detect the solver if it is installed. You can also manually specify the solver in API calls.

## Example Graph Usage

```python
from graphcalc import (
    independence_number,
    domination_number,
    zero_forcing_number,
)
from graphcalc.generators import petersen_graph

# Calculate and print the independence number of the Petersen graph.
G = petersen_graph()
print(f"Petersen graph independence number = {independence_number(G)}")

# Calculate and print the domination number of the Petersen graph.
print(f"Petersen graph domination number = {domination_number(G)}")

# Calculate and print the zero forcing number of the Petersen graph.
print(f"Petersen graph zero forcing number = {zero_forcing_number(G)}")
```

## Example Polytope Usage

```python
import graphcalc as gc
from graphcalc.polytopes.generators import (
    cube_graph,
    octahedron_graph,
    dodecahedron_graph,
    tetrahedron_graph,
    icosahedron_graph,
    convex_polytopes_text_example,
)

# Generate polytope graphs (cubes, octahedra, etc.)
G1 = cube_graph()
G2 = octahedron_graph()
G3 = dodecahedron_graph()
G4 = tetrahedron_graph()
G5 = icosahedron_graph()
G6 = convex_polytopes_text_example(1)
G7 = convex_polytopes_text_example(2)


# Function names to compute
function_names = [
    "order", # number of vertices
    "size", # number of edges
    "p_vector",
    "independence_number",
    "vertex_cover_number",
    "maximum_degree",
    "average_degree",
    "minimum_degree",
    "spectral_radius",
    "diameter",
    "radius",
    "girth",
    "algebraic_connectivity",
    "largest_laplacian_eigenvalue",
    "second_largest_adjacency_eigenvalue",
    "smallest_adjacency_eigenvalue",
    "fullerene",
    ]

# Compute properties for multiple polytopes
graphs = [G1, G2, G3, G4, G5, G6, G7]
df = gc.compute_knowledge_table(function_names, graphs)
```

## Creating Simple Graphs, Polytope Graphs, and Simple Polytope Graphs

```python
import graphcalc as gc

# Draw a simple graph
G = gc.SimpleGraph(name="Example Graph")
G.add_edges_from([(0, 1), (1, 2), (2, 3)])
G.draw()
```

### Author

Randy Davila, PhD
Email: <rrd6@rice.edu>
