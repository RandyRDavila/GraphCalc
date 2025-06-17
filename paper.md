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

# Summary

`GraphCalc` is a Python library for computing a wide range of graph-theoretic invariants using a blend of exact enumeration, solver-based optimization, and integrations with `NetworkX`. In addition to supporting `NetworkX` graphs, the package includes internal data structures for graphs and polytopes, facilitating experimentation and extension.

Originally developed to support **automated conjecturing systems**, `GraphCalc` powers the *GraffitiAI* family of programs [@TxGraffiti2023; @TxGraffiti; @optimist], where it is used to generate and organize numerical and Boolean data on simple graphs and polytopes in order to formulate computer-generated conjectures.

`GraphCalc` includes tools for computing both classical and specialized invariants. These include well-known parameters such as the **independence number**, **clique number**, and **chromatic number** (each computed exactly using optimization models), as well as lesser-known but actively studied invariants such as:

- Spectral properties related to **graph energy** [@LiShiGutman2012],
- Degree-sequence-based invariants like **residue** [@residue], **annihilation number** [@LevitMandrescu2022], and **Slater number** [@GeRa2017],
- Dynamic coloring-based parameters such as the **zero forcing number** and its many variants [@AIMMINIMUMRANKSPECIALGRAPHSWORKGROUP20081628; @AMOS20151; @DAVILA2019115; @DavilaHenningMagnantPepper2018],
- The widely studied **domination number** and its generalizations (**total**, **connected**, **Roman**, **k-domination**, etc.) [@HaHeHe_core; @HaHeHe_topics; @HeYe2010],
- Structural constraints defined by forbidden subgraphs, such as being **claw-free**, **triangle-free**, **diamond-free**, and **bull-free**.

Altogether, `GraphCalc` implements over 100 graph-related functions, including invariant computation, spectral analysis, structural testing, and graph generation. Many of these are unavailable in any other Python package, and all are computed exactly using integer programming or specialized algorithms. The library is also well-suited for rigorous graph-theoretic research, enabling users to verify counterexamples and explore structural hypotheses.

# Statement of Need

Many aspects of research in graph theory, as well as in automated reasoning and conjecture generation on discrete structures rely on the accurate computation of graph invariants across large collections of graphs. While important and general purpose libraries such as `NetworkX` [@osti_960616] and `igraph` [@csardi2006igraph] offer essential tools for graph manipulation and visualization, they provide limited support for computing many of the specialized and computationally difficult invariants central to a large amount of graph (and polyhedral) theoretical research.

`GraphCalc` fills this gap by offering a unified, solver enhanced framework for computing both classical and advanced graph invariants. Many parameters are evaluated via linear-integer programming models, relying on the `PuLP` optimization toolkit [@mitchell2011pulp] and the COIN‑OR CBC solver [@forrest2005cbc] to compute exact solutions for many NP‑hard invariants, or fallback to exhaustive enumeration when necessary. The library currently supports over 100 functions across a wide range of categories, including:

- Classical invariants: **independence number**, **clique number**, **chromatic number**, **vertex cover**, and **matching number**;
- Over a dozen **domination-type invariants**, including **total**, **Roman**, **double Roman**, **restrained**, **outer-connected**, and **k-domination**;
- Specialized parameters like **zero forcing**, **positive semidefinite forcing**, **connected forcing**, and **k-power domination**;
- Degree-based and spectral invariants such as **residue**, **annihilation number**, **Slater number**, **spectral radius**, and **algebraic connectivity**;
- Structural graph properties and Boolean predicates for detecting trees, regularity, planarity, forbidden subgraphs, and more.

Unlike broad symbolic systems such as SageMath [@sagemath], `GraphCalc` is purpose-built for seamless integration into Python **automated conjecturing systems**. It serves as the computational engine for the *TxGraffiti* and *Optimist* systems [@TxGraffiti; @optimist], which depend on accurate invariant computations to discover, test, and rank symbolic conjectures that relate seemingly unrelated structural properties of graphs and polytopes.

# Features

`GraphCalc` provides a comprehensive suite of graph-theoretic computations, combining familiar interfaces with powerful solver-based backends. It supports both `NetworkX` graph objects and its own lightweight `SimpleGraph` and polytope graph types, allowing users to flexibly model, analyze, and visualize a wide variety of graph structures. Key features include:

- **Extensive invariant support**: Compute classical parameters such as the **chromatic number**, **maximum clique**, **vertex cover**, **matching number**, and **independence number**, alongside specialized invariants like **residue**, **Slater number**, and **annihilation number**.
- **Domination and zero forcing variants**: Includes exact computation of more than a dozen domination-type invariants (e.g., **total**, **Roman**, **double Roman**, **restrained**) and advanced forcing parameters like **positive semidefinite zero forcing** and **k-power domination**.
- **Spectral and structural analysis**: Supports computation of eigenvalues (adjacency and Laplacian), **spectral radius**, **algebraic connectivity**, and detection of structural properties (e.g., planarity, claw-freeness, subcubicity).
- **Graph and polytope generators**: Includes generators for classical graphs (e.g., **Petersen**, **Barabási–Albert**, **Watts–Strogatz**) and convex polytopes (e.g., **cubes**, **tetrahedra**, **fullerenes**).
- **Batch evaluation**: Compute multiple invariants across many graphs using `compute_graph_properties_dataframe`, returning results in a clean, tabular format suitable for analysis and experimentation.
- **Visualization and usability**: Native support for drawing graphs and polytopes; functions are type-annotated and tested, with complete documentation hosted online.

# Example Usage

```python
import graphcalc as gc
from graphcalc.generators import petersen_graph

# Create the Petersen graph
G = petersen_graph()

# Compute basic invariants
print("Independence number:", gc.independence_number(G))    # Output: 4
print("Chromatic number:", gc.chromatic_number(G))          # Output: 3
print("Connected?", gc.connected(G))
```

You can also analyze polytope graphs and batch-compute properties:

```python
from graphcalc.polytopes.generators import cube_graph, octahedron_graph
graphs = [cube_graph(), octahedron_graph()]
functions = ["order", "size", "spectral_radius", "independence_number"]

df = gc.compute_graph_properties_dataframe(functions, graphs)
print(df)
```

# Relevance to Automated Discovery

Automated mathematical discovery has deep roots, dating back to early symbolic logic systems in the 1950s with Wang’s *Program II* [@wang1960mechanical], and gaining momentum in the 1980s with systems like Fajtlowicz’s *Graffiti* [@Fajtlowicz1988] and DeLaViña’s *Graffiti.pc* [@DeLaVina2005]. These programs demonstrated that computers could do more than verify mathematics—they could help discover it, particularly through the generation of conjectures based on structural graph invariants. Collectively, such systems have led to well over 100 published mathematical conjectures, many appearing in top-tier journals.

`GraphCalc` continues in this tradition, serving as the computational foundation for modern systems such as *TxGraffiti* and *Optimist* [@TxGraffiti2023; @optimist], which generate symbolic conjectures by analyzing numerical patterns across families of graphs and polytopes. To support this process, `GraphCalc` supplies high-quality, reproducible invariant data—ranging from classical quantities like **chromatic number** and **independence number** to more specialized parameters such as **positive semidefinite zero forcing**, **annihilation number**, and **Slater number**.

While `GraphCalc` is not designed for massive-scale network analysis, it excels in the domain where most mathematical conjectures are actually formed: small to medium-sized graphs that can be drawn, visualized, and reasoned about—graphs that fit comfortably within the scope of human mathematical intuition. This design philosophy echoes that of Fajtlowicz, who emphasized working with “small but interesting” graphs as the foundation for conjecture generation. By transforming these graphs into rich numerical profiles, `GraphCalc` enables automated systems to detect symbolic patterns and formulate conjectures that are not only structurally meaningful but also aligned with the way mathematicians discover and explore new ideas.

# Acknowledgements

The authors gratefully acknowledge David Amos and Boris Brimkov for their foundational support during the development of `GraphCalc`. David Amos provided technical insight and early design feedback, while Boris Brimkov’s mathematical input helped guide the selection and implementation of key graph invariants.

# References
