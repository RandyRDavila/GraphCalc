---
title: 'GraphCalc: A Python Package for Computing Graph Invariants in Automated Conjecturing Systems'
tags:
  - Python
  - graph theory
  - graph invariants
  - optimization
authors:
  - name: Randy Davila
    orcid: 0000-0002-0471-8744
    # equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: RelationalAI, United States
   index: 1
 - name: Department of Computational Applied Mathematics & Operations Research,Rice University, United States
   index: 2
date: 20 Aug 2025
bibliography: paper.bib
---

## Summary

`GraphCalc` is a Python library for computing an extensive collection of graph-theoretic invariants, designed to support research in combinatorics, network science, and automated reasoning. It implements more than 100 exact functions, covering classical measures (e.g., independence number, chromatic number, spectral radius) alongside many lesser-known invariants central to extremal graph theory and domination theory.

Originally developed as the invariant engine for the automated conjecturing system *TxGraffiti* [@TxGraffiti], `GraphCalc` has grown into a general-purpose research tool for constructing large, structured datasets of graph invariants. These datasets—often organized into tabular *knowledge tables*-enable symbolic pattern mining, hypothesis generation, and automated conjecture discovery. For example:

```python
>>> import graphcalc as gc
>>> graphs = [gc.cube_graph(), gc.octahedron_graph()]
>>> functions = ["order", "size", "spectral_radius", "independence_number"]
>>> gc.compute_knowledge_table(functions, graphs)
   order  size  spectral_radius  independence_number
0      8    12              3.0                    4
1      6    12              4.0                    2
```

While general-purpose libraries like `NetworkX` [@osti_960616], `igraph` [@csardi2006igraph], and `SageMath` [@sagemath] provide broad graph functionality, they rarely support the wide range of nonstandard invariants used in combinatorics. GraphCalc fills this gap by offering exact implementations of many parameters unavailable elsewhere, including:

- Classical invariants such as the independence, clique, and chromatic numbers,
- Spectral properties of adjacency and Laplacian matrices,
- Degree-sequence-based invariants (e.g., residue, annihilation number, Slater number),
- Propagation parameters like zero forcing, k-forcing, and power domination,
- Domination-type parameters including Roman, rainbow, and restrained domination,
- Structural predicates (e.g., claw-free, triangle-free, cographs).

All functions are implemented exactly using integer programming, enumeration, or symbolic methods. For NP-hard invariants (e.g., independence number, chromatic number, domination variants), `GraphCalc` relies on mixed-integer programming models via `PuLP` [@mitchell2011pulp] and solvers such as `COIN-OR CBC`, ensuring correctness while still supporting small to medium sized graphs where symbolic relationships are most visible.

By enabling high-resolution invariant datasets, `GraphCalc` complements automated conjecturing systems like *TxGraffiti* and the *Optimist* [@optimist]. These systems analyze numerical patterns in `GraphCalc`’s output to generate new conjectures, many of which have already been proven as theorems. Thus, `GraphCalc` serves as both a comprehensive toolkit for graph theorists and a foundational component for symbolic discovery in modern mathematics.

## Features

`GraphCalc` offers a robust suite of tools for computing, analyzing, and visualizing graph-theoretic invariants. It combines an intuitive Python interface with solver-enhanced backends and supports both `NetworkX` graph objects and internal `SimpleGraph` and related types—making it versatile for everyday use, educational settings, and advanced mathematical experimentation. Key features include:

- **Extensive invariant coverage:** Compute a broad range of exact graph invariants, including classical quantities such as chromatic number, maximum clique, vertex cover, and independence number, as well as structural and degree-based invariants like residue, Slater number, and annihilation number.

- **Domination and forcing variants:** Includes over a dozen domination-type parameters (e.g., total, Roman, double Roman, restrained, outer-connected) and propagation-based parameters such as zero forcing, positive semidefinite zero forcing, and k-power domination. All are computed exactly using integer programming or exhaustive search.

- **Spectral and structural analysis:** Supports spectral computations, including adjacency and Laplacian eigenvalues, spectral radius, and algebraic connectivity, along with Boolean predicates for structural properties such as planarity, claw-freeness, triangle-freeness, and subcubicity.

- **Graph and polytope generators:** Provides built-in generators for classical graphs (derived from NetworkX) and convex 3D polytopes (e.g., tetrahedra, cubes, fullerenes)—useful for visualization, testing, and conjecture exploration.

- **Batch evaluation and knowledge tables:** Enables the evaluation of multiple invariants across entire graph collections using compute_graph_properties_dataframe, which returns results as structured knowledge tables. These tables integrate directly with automated conjecturing systems like TxGraffiti.

- **Visualization and user experience:** Offers built-in rendering for graphs and polytopes, fully type-annotated functions, extensive test coverage, and online documentation designed to support both research and instructional use.

## Example Usage

The `GraphCalc` package supports both single-graph queries and batch evaluation over collections of graphs and polytopes (see the previous section). Below is a basic example using the *Petersen graph*:

```python
>>> import graphcalc as gc
>>> # Create the Petersen graph
>>> G = gc.petersen_graph()
>>> # Compute selected invariants
>>> gc.independence_number(G)
4
>>> gc.residue(G)
3
>>> gc.claw_free(G)
False
```

## Relevance to Automated Discovery

Automated mathematical discovery has a rich history, dating back to symbolic logic programs like Wang’s *Program II* in the 1950s [@wang1960mechanical], and advancing significantly with systems such as Fajtlowicz’s *Graffiti* [@Fajtlowicz1988] and DeLaViña’s *Graffiti.pc* [@DeLaVina2005] in the 1980s and 1990s. These pioneering systems demonstrated that computers could do more than verify known mathematics—they could help *generate* it, particularly by formulating conjectures grounded in patterns among graph invariants. Notably, *Graffiti* included its own embedded module for computing such invariants, a design decision that enabled the system to generate over 60 published conjectures, many appearing in top mathematical journals.

`GraphCalc` continues this lineage. Originally developed as the internal invariant engine for the *TxGraffiti* system, it served for years as a private computational backend before being released as an open-source Python package. This decision was motivated by the growing interest in AI-assisted mathematical reasoning and the desire to make a high-quality, extensible invariant engine available for others to experiment with.

Today, `GraphCalc` powers the latest version of *TxGraffiti* and its agentic counterpart the *Optimist* [@TxGraffiti2023; @optimist], which analyze large families of graphs and polytopes to discover symbolic conjectures. To support this process, `GraphCalc` computes a rich spectrum of graph-theoretic quantities—from well-known invariants like *chromatic number* and *independence number* to more specialized parameters such as *positive semidefinite zero forcing*, *Slater number*, and *residue*.

While not optimized for massive-scale network analysis, `GraphCalc` excels in the domain where most mathematical conjectures are formed: *small to medium-sized graphs* that are easily visualized and reasoned about. This design philosophy echoes the foundational principles of Fajtlowicz’s original system, which emphasized working with “small but interesting” graphs as fertile ground for discovery. By transforming these structures into structured numerical profiles, `GraphCalc` enables automated systems to detect symbolic patterns and formulate conjectures that are both novel and mathematically meaningful.

## Acknowledgements

The authors gratefully acknowledge David Amos and Boris Brimkov for their foundational support during the development of `GraphCalc`. David Amos provided early technical insight and design feedback that shaped the architecture of the package, while Boris Brimkov contributed valuable mathematical guidance in selecting and implementing key graph invariants.

We also thank the referees who provided detailed comments on the manuscript and codebase during the revision process. Their feedback significantly improved the quality, clarity, and functionality of both the `GraphCalc` library and this paper.

## References
