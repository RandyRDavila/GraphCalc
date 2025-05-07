---

title: 'GraphCalc: A Python Package for Computing Graph Invariants in Automated Conjecturing Systems'
date: 7 May 2025
authors:

* name: Randy Davila
  orcid: 0000-0002-9908-3760
  affiliation: "1"
  affiliations:
* name: Department of Computational & Applied Mathematics, Rice University, United States
  index: 1
  ror: 00pjdza24
  keywords:
* Python
* graph theory
* graph invariants
* automated conjecturing
* open source software
  bibliography: paper.bib

---

# Summary

`GraphCalc` is a lightweight Python library for efficiently computing a wide range of graph invariants. It plays a central role in the **TxGraffiti** family of automated conjecturing systems, where it serves as the primary computational engine for generating data-driven mathematical conjectures in graph theory. The package provides accessible tools for calculating over 50 structural graph properties, including chromatic number, independence number, domination parameters, degree statistics, and more. It is designed with extensibility and speed in mind, allowing users to plug in custom invariants or filter graphs based on logical predicates.

# Statement of Need

Modern research in graph theory and network science often requires the rapid computation of multiple graph invariants across large collections of graphs. While libraries such as NetworkX [@hagberg2008exploring] and igraph provide graph traversal and manipulation utilities, they often lack built-in support for specialized invariants central to theoretical graph research — such as total domination number or zero forcing number.

`GraphCalc` fills this gap by offering a unified interface for computing classical and specialized invariants. It is **purpose-built for use in automated conjecturing frameworks**, particularly **TxGraffiti** [@TxGraffiti] and its variants (The `Optimist` [@optimist], etc.), which rely on precise invariant computations across thousands of graphs for conjecture generation and evaluation.

# Features

`GraphCalc` supports both `networkx` graph objects and its own built-in graph types. This dual compatibility allows users to take advantage of familiar graph creation tools while benefiting from specialized features and optimizations specific to `GraphCalc`'s native graph representations.

* **Core invariants**: Computes over 50 classical and advanced graph invariants.
* **Graph filters**: Boolean functions to test properties like planarity, connectivity, and claw-freeness among many others.
* **Documentation and testing**: Full online documentation, type hints, and unit test coverage.

# Example Usage

```python
import graphcalc as gc

G = gc.petersen_graph()
gc.chromatic_number(G)       # Output: 3
gc.independence_number(G)      # Output: 4
gc.connected(G)           # Output: True
```

# Relevance to Automated Discovery

`GraphCalc` is not merely a utility—it is an **enabling technology** in the development of AI systems that perform **automated mathematical reasoning**. In systems like `TxGraffiti`[@TxGraffiti2023], conjectures are derived from numerical relationships among invariants computed by GraphCalc. The effectiveness of these conjecturing agents hinges directly on the reliability and breadth of invariant computations provided by this package.

# Acknowledgements

The authors would like to acknowledge David Amos. Without his assistance this package would have not been possible.

# References

Citations for tools and prior work referenced in this manuscript are included in the accompanying `paper.bib` file.
