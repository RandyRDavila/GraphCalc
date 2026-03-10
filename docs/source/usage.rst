Quickstart
==========

This page gives a brief overview of the kinds of workflows supported by
GraphCalc. The examples below are intentionally lightweight and are meant to
serve as starting points for deeper exploration of the library.

Working with graphs
-------------------

GraphCalc provides graph-focused functionality under ``graphcalc.graphs`` for
constructing graphs, computing invariants, and working with specialized graph
families.

Typical graph workflows include:

- generating graphs from named families
- computing classical and structural invariants
- studying graph neighborhoods, domination parameters, or spectral data
- working with graph families arising from polytope constructions

Working with hypergraphs
------------------------

The ``graphcalc.hypergraphs`` package provides tools for constructing
hypergraphs and computing hypergraph-specific invariants and properties.

Typical hypergraph workflows include:

- constructing uniform hypergraphs
- studying acyclicity, chromatic behavior, matching, and transversals
- computing structural descriptors and extremal quantities
- generating data for computational exploration

Working with quantum information
--------------------------------

The ``graphcalc.quantum`` package supports finite-dimensional quantum states,
channels, and measurements.

Typical quantum workflows include:

- constructing standard state families such as basis, Bell, GHZ, W, and Werner states
- computing quantities such as entropy, purity, fidelity, and negativity
- testing structural properties such as purity, separability-related conditions, or channel validity
- generating datasets from parameter grids and exporting them for analysis

Working with solver-backed routines
-----------------------------------

GraphCalc also includes solver-based functionality that can be used in discrete
and combinatorial workflows where optimization-based computations are needed.

Typical solver workflows include:

- computing quantities defined through optimization formulations
- integrating combinatorial constructions with linear or integer programming
- using generated objects as inputs to downstream solver routines

Next steps
----------

Use the :doc:`api_reference` for a complete module-level reference.

See :doc:`examples` for additional examples and patterns of use.
