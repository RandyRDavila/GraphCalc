Introduction
============

What is GraphCalc?
------------------

GraphCalc is a Python library for computation in graph theory, hypergraph
theory, combinatorics, discrete mathematics, and quantum information.

The library began as a graph-focused package for computing graph invariants and
working with structured graph families. It has since expanded into a broader
mathematical toolkit that supports multiple discrete domains while retaining a
common design philosophy: mathematically transparent objects, reusable
generators, invariant computation, property testing, and dataset-oriented
experimentation.

Scope
-----

GraphCalc is designed for users who want to work programmatically with
mathematical structures arising in discrete mathematics and quantum
information. Typical use cases include:

- constructing graphs, hypergraphs, and related combinatorial objects
- computing numerical invariants and structural properties
- generating examples and datasets for experimentation
- studying families of finite-dimensional quantum states, channels, and measurements
- integrating solver-based routines into mathematical workflows

Package overview
----------------

The main public areas of the library are:

``graphcalc.graphs``
    Tools for graphs, including core graph utilities, generators, invariant
    computation, polytope-related modules, and visualization helpers.

``graphcalc.hypergraphs``
    Tools for hypergraph construction, basic operations, and hypergraph
    invariants.

``graphcalc.quantum``
    Tools for finite-dimensional quantum states, channels, measurements, local
    channel actions, and dataset generation.

``graphcalc.solvers``
    Solver-backed routines and optimization-oriented helpers used across the
    library.

``graphcalc.utils``
    General-purpose supporting utilities shared across workflows.

Design philosophy
-----------------

GraphCalc emphasizes:

- clear mathematical APIs
- reusable computational building blocks
- support for experimentation and conjecture-driven workflows
- compatibility with standard Python scientific tooling
- a balance between conceptual clarity and practical computation

Typical workflows
-----------------

A typical GraphCalc workflow may involve one or more of the following:

- generating a graph or hypergraph from a structured family
- computing invariants or testing properties
- exporting collections of computed data for later analysis
- constructing and studying quantum states, channels, or measurements
- using solver-backed routines to evaluate combinatorial quantities

Where to go next
----------------

For installation instructions, see :doc:`installation`.

For a practical overview of common workflows, see :doc:`usage`.

For a package-level reference to the public API, see :doc:`api_reference`.
