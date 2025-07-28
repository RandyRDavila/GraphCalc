---
title: 'GraphCalc: A Python Package for Computing Graph Invariants in Automated Conjecturing Systems'
tags:
  - Python
  - graph theory
  - graph invariants
  - optimization
authors:
  - name: Randy Davila
    orcid: 0000-0002-9908-3760
    # equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: RelationalAI, United States
   index: 1
 - name: Department of Computational Applied Mathematics & Operations Research,Rice University, United States
   index: 2
date: 7 May 2025
bibliography: paper.bib
---

## Summary

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

In contrast to general graph libraries such as NetworkX or igraph, `GraphCalc` emphasizes coverage over convenience, providing researchers with one of the most extensive collections of computable graph invariants available in open-source software. Its design is particularly suited to mathematical exploration on small to medium graphs, where symbolic relationships among invariants are often most visible and meaningful. By enabling fast, programmatic access to both standard and obscure graph parameters, `GraphCalc` lowers the barrier to large-scale experimentation and supports the discovery of new relationships in discrete mathematics.

The library includes exact implementations of:

- Fundamental quantities such as independence number, clique number, and chromatic number (via integer programming),
Spectral properties related to graph energy, i.e., eigenvalues of adjacency and Laplacian matrices,
- Degree-sequence-based invariants like residue, annihilation number, and Slater number,
- Propagation-based parameters including zero forcing, k-forcing, and power domination variants,
- Domination-type invariants such as total, connected, Roman, rainbow, and restrained domination,
- Structural predicates for classes like claw-free, triangle-free, diamond-free, and bull-free graphs.

Together, these functions cover an expansive and often underrepresented space of graph invariants. Many of these are unavailable in other Python libraries, and all are computed exactly using a mix of solver-based optimization, enumeration, and symbolic methods. As such, `GraphCalc` serves not only as a comprehensive computational toolkit for graph theorists, but as a foundational component in modern systems for automated conjecture generation and symbolic discovery.

Research in graph theory, combinatorics, and automated reasoning frequently relies on the accurate and efficient computation of graph invariants across large collections of discrete structures. While established tools such as SageMath [@sagemath], NetworkX [@osti_960616], and igraph [@csardi2006igraph] support a variety of such computations, they do not offer the breadth, depth, or conjecture-oriented design that distinguishes `GraphCalc`—particularly for researchers in extremal graph theory and related subfields.
The library implements over 100 exact functions, including many invariants that are unavailable or difficult to compute using existing packages. These span classical quantities, advanced domination-type parameters, spectral metrics, and structural graph properties. In addition to its broad invariant coverage, `GraphCalc` supports batch evaluation over graph collections, returning results as structured “knowledge tables”—tabular datasets suitable for downstream reasoning tasks. This workflow mirrors preprocessing pipelines in machine learning, where raw instances are transformed into rich feature representations.

By integrating high-resolution numerical data with automated conjecturing systems such as TxGraffiti, `GraphCalc` enables symbolic pattern discovery at scale—supporting both empirical investigation and the generation of novel mathematical conjectures.

To support this functionality, `GraphCalc` employs a hybrid computational strategy that blends solver-based optimization, exhaustive enumeration, and well-defined algorithms. Many NP-hard invariants—including the independence number, chromatic number, and total domination number—are computed exactly using integer programming models via the PuLP library [@mitchell2011pulp] and the COIN-OR CBC solver. This solver-enhanced foundation allows `GraphCalc` to compute dozens of invariants with provable guarantees, making it an essential tool for both mathematical experimentation and automated discovery.

The utility of `GraphCalc` is most clearly demonstrated through its integration with systems such as TxGraffiti and Optimist [@TxGraffiti; Davila2025InReverie; @optimist], which generate conjectures by analyzing numerical patterns across large families of graphs. These symbolic patterns emerge directly from the invariant-rich data produced by `GraphCalc`. In this capacity, the library has facilitated the formulation of numerous conjectures—many of which have been rigorously proven as new theorems, while others remain unresolved open problems.

Notably, these conjectures often connect well-studied invariants (e.g., independence number, domination number) with less commonly explored parameters (e.g., residue, Slater number, zero forcing number), revealing structural relationships that had previously gone unrecognized. By providing exact and broad invariant coverage—including many parameters absent from SageMath, NetworkX, igraph, and passagemath-graphs—`GraphCalc` substantially expands the conjectural search space accessible to modern automated reasoning systems.

## Features

`GraphCalc` offers a powerful suite of tools for computing, analyzing, and visualizing graph-theoretic invariants. It combines an intuitive Python interface with solver-enhanced backends, and supports both `NetworkX` graph objects and internal `SimpleGraph` and polytope types—making it versatile for both everyday users and advanced mathematical experimentation. Key features include:

- **Extensive invariant coverage**
  Compute a wide range of exact graph invariants—from classical quantities such as **chromatic number**, **maximum clique**, **vertex cover**, and **independence number**, to structural and degree-based invariants like **residue**, **Slater number**, and **annihilation number**.

- **Domination and forcing variants**
  Includes over a dozen domination-type parameters (e.g., **total**, **Roman**, **double Roman**, **restrained**, **outer-connected**) as well as propagation-based invariants such as **zero forcing**, **positive semidefinite zero forcing**, and **k-power domination**. All are computed exactly using integer programming or exhaustive search.

- **Spectral and structural analysis**
  Supports spectral computations including **adjacency** and **Laplacian eigenvalues**, **spectral radius**, and **algebraic connectivity**, along with Boolean checks for structural properties such as **planarity**, **claw-freeness**, **triangle-freeness**, and **subcubicity**.

- **Graph and polytope generators**
  Includes built-in generators for classical graphs (e.g., **Petersen**, **Barabási–Albert**, **Watts–Strogatz**, **grid graphs**) and convex 3D polytopes (e.g., **tetrahedra**, **cubes**, **fullerenes**)—useful for testing, visualization, and conjecture exploration.

- **Batch evaluation and knowledge tables**
  Compute multiple invariants across large graph collections using `compute_graph_properties_dataframe`, which returns results as structured **knowledge tables**. These tables integrate seamlessly with automated conjecturing systems like *TxGraffiti*.

- **Visualization and user experience**
  Built-in support for rendering graphs and polytopes, with fully type-annotated functions, thorough test coverage, and clear online documentation designed for both research and classroom use.

## Example Usage

The `graphcalc` package supports both single-graph queries and batch evaluation over collections of graphs. Below is a basic example using the **Petersen graph**:

```python
import graphcalc as gc
from graphcalc.generators import petersen_graph

# Create the Petersen graph
G = petersen_graph()

# Compute basic invariants
print("Independence number:", gc.independence_number(G))    # Output: 4
print("Chromatic number:", gc.chromatic_number(G))          # Output: 3
print("Connected?", gc.connected(G))                        # Output: True
```

You can also analyze polytope graphs and batch-compute properties:

```python
from graphcalc.polytopes.generators import cube_graph, octahedron_graph
graphs = [cube_graph(), octahedron_graph()]
functions = ["order", "size", "spectral_radius", "independence_number"]

df = gc.compute_graph_properties_dataframe(functions, graphs)
print(df)
```

This produces a clean, labeled `pandas.DataFrame` summarizing multiple invariant values for each graph—ideal for use in automated conjecturing pipelines or exploratory analysis.

## Relevance to Automated Discovery

Automated mathematical discovery has a rich history, dating back to symbolic logic programs like Wang’s *Program II* in the 1950s [@wang1960mechanical], and advancing significantly with systems such as Fajtlowicz’s *Graffiti* [@Fajtlowicz1988] and DeLaViña’s *Graffiti.pc* [@DeLaVina2005] in the 1980s and 1990s. These pioneering systems demonstrated that computers could do more than verify known mathematics—they could help **generate** it, particularly by formulating conjectures grounded in patterns among graph invariants. Notably, *Graffiti* included its own embedded module for computing such invariants, a design decision that enabled the system to generate over 60 published conjectures, many appearing in top mathematical journals.

`GraphCalc` continues this lineage. Originally developed as the internal invariant engine for the *TxGraffiti* system, it served for years as a private computational backend before being released as an open-source Python package. This decision was motivated by the growing interest in AI-assisted mathematical reasoning and the desire to make a high-quality, extensible invariant engine available for others to experiment with.

Today, `GraphCalc` powers the latest version of *TxGraffiti* and its agentic counterpart the *Optimist* [@TxGraffiti2023; @optimist], which analyze large families of graphs and polytopes to discover symbolic conjectures. To support this process, `GraphCalc` computes a rich spectrum of graph-theoretic quantities—from well-known invariants like **chromatic number** and **independence number** to more specialized parameters such as **positive semidefinite zero forcing**, **Slater number**, and **residue**.

While not optimized for massive-scale network analysis, `GraphCalc` excels in the domain where most mathematical conjectures are formed: **small to medium-sized graphs** that are easily visualized and reasoned about. This design philosophy echoes the foundational principles of Fajtlowicz’s original system, which emphasized working with “small but interesting” graphs as fertile ground for discovery. By transforming these structures into structured numerical profiles, `GraphCalc` enables automated systems to detect symbolic patterns and formulate conjectures that are both novel and mathematically meaningful.

## Acknowledgements

The authors gratefully acknowledge David Amos and Boris Brimkov for their foundational support during the development of `GraphCalc`. David Amos provided early technical insight and design feedback that shaped the architecture of the package, while Boris Brimkov contributed valuable mathematical guidance in selecting and implementing key graph invariants.

We also thank the many individuals who provided detailed comments on the manuscript and codebase during the revision process. Their feedback significantly improved the quality, clarity, and functionality of both the `GraphCalc` library and this paper.

## References
