# GraphCalc
[![Documentation Status](https://readthedocs.org/projects/graphcalc/badge/?version=latest)](https://graphcalc.readthedocs.io/en/latest/?badge=latest)


## Overview

`graphcalc` is a Python package for performing a variety of graph computations, including maximum clique detection, chromatic number calculation, and vertex cover identification. It is built on top of `networkx` and provides efficient implementations of fundamental graph theory algorithms.

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


## Example Usage
```python
import networkx as nx
import graphcalc as gc

# Calculate and print the independence number of the Petersen graph.
G = nx.petersen_graph()
print(f"independence number of G = {gc.independence_number(G)}")

# Calculate and print the domination number of the Petersen graph.
print(f"domination number of G = {gc.domination_number(G)}")

# Calculate and print the zero forcing number of the Petersen graph.
print(f"zero forcing number of G = {gc.zero_forcing_number(G)}")
```


### Author
Randy Davila, PhD
