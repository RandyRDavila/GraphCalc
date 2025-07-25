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
 - name: Department of Computational Applied Mathematics & Operations Research, Rice University, United States
   index: 1
 - name: RelationalAI, United States
   index: 2
date: 7 May 2025
bibliography: paper.bib
---

## Summary

`GraphCalc` is a Python library for computing a wide range of graph-theoretic invariants using a blend of exact enumeration, solver-based optimization, and seamless integration with `NetworkX`. While it supports `NetworkX` graphs natively, it also includes its own lightweight data structures for graphs and polytopes, enabling efficient experimentation, extension, and symbolic computation.

Originally developed to power *automated conjecturing systems**, `GraphCalc` forms the computational backbone of the *TxGraffiti* (*TexasGraffiti*) family of programs [@TxGraffiti2023; @TxGraffiti; @optimist]. These systems generate mathematical conjectures by analyzing large datasets of graphs enriched with invariant data—making the scope, precision, and diversity of `GraphCalc`'s computations essential for discovery.

The library supports both classical and advanced invariants. These include:

- Fundamental quantities such as *independence number*, *clique number*, and *chromatic number* (each computed exactly via integer programming),
- Spectral properties related to *graph energy* [@LiShiGutman2012],
- Degree-sequence-based invariants like *residue* [@residue], *annihilation number* [@LevitMandrescu2022], and *Slater number* [@GeRa2017],
- Dynamic coloring parameters such as the *zero forcing number* and its numerous variants [@AIMMINIMUMRANKSPECIALGRAPHSWORKGROUP20081628; @AMOS20151; @DAVILA2019115; @DavilaHenningMagnantPepper2018],
- Domination-type parameters including *total*, *connected**, *Roman**, and *rainbow*, *k-domination* [@HaHeHe_core; @HaHeHe_topics; @HeYe2010],
- Structural predicates that test for properties like *claw-free*, *triangle-free*, *diamond-free*, or *bull-free* graphs.

In total, `GraphCalc` provides over 100 graph-related functions for invariant evaluation, spectral analysis, structural testing, and graph generation—many of which are unavailable in other Python packages. All are computed *exactly*, leveraging integer programming, enumeration, or symbolic methods. As such, `GraphCalc` is not only a powerful research tool for graph theorists, but also a critical enabler for automated reasoning systems seeking to discover new mathematical truths.

## Statement of Need

Research in graph theory, combinatorics, and automated reasoning often depends on the accurate computation of graph invariants across large collections of discrete structures. While symbolic systems like `SageMath` [@sagemath] and general-purpose libraries such as `NetworkX` [@osti_960616] and `igraph` [@csardi2006igraph] support a range of invariant computations, they do not offer the *breadth**, *depth**, or *conjecture-oriented design** found in `GraphCalc`. The library implements over 100 exact functions—including many that are difficult to find elsewhere—covering classical invariants, advanced domination-type parameters, spectral quantities, and structural graph properties. Moreover, `GraphCalc` enables users to batch-evaluate entire collections of graphs and export the results as structured knowledge tables—tabular datasets that serve as input features for downstream reasoning systems. This design mirrors preprocessing pipelines in machine learning, where raw data is transformed into feature-rich representations. As a result, `GraphCalc` integrates seamlessly into automated conjecturing systems like *TxGraffiti*, where symbolic patterns are mined from numerical invariant data.

To support this broad functionality, `GraphCalc` employs a hybrid computational strategy that blends solver-based optimization, exhaustive enumeration, and well defined optimal algorithms. Many NP-hard invariants—including the independence number, chromatic number, and total domination number—are computed exactly using integer programming models via the `PuLP` library [@mitchell2011pulp] and the COIN-OR CBC solver [@forrest2005cbc]. Others, such as the residue, annihilation number, and Slater number, rely on enumeration or degree-sequence analysis tailored for structural graph theory. This solver-enhanced foundation enables `GraphCalc` to compute dozens of invariants with exact guarantees—making it an essential tool for both mathematical experimentation and automated discovery.

The utility of `GraphCalc` is most evident in its integration with automated conjecturing systems such as *TxGraffiti* and *Optimist* [@TxGraffiti; @optimist]. These systems generate symbolic conjectures by analyzing numerical patterns across large collections of graphs—patterns that emerge from the high-resolution invariant data provided by `GraphCalc`. In this capacity, `GraphCalc` has enabled the formulation of numerous conjectures, many of which have been proven as new theorems, while others remain unresolved open problems. Crucially, these conjectures often relate classical invariants (e.g., independence number or domination number) to lesser-known parameters (e.g., residue, Slater number, or zero forcing number), revealing novel structural relationships that were previously unexplored. By providing exact data on a broad spectrum of invariants—including dozens not found in `SageMath`, `NetworkX`, `igraph`, or `passagemath-graphs`—`GraphCalc` substantially enlarges the conjectural search space available to modern automated reasoning systems.

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
