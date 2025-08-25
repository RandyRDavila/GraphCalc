Usage Guide
===========

This guide shows common ways to use **GraphCalc** to compute graph invariants,
choose a solver, and build reusable analysis tables with pandas.

All functions accept either a ``networkx.Graph`` or ``graphcalc.SimpleGraph``.
Many invariants are NP-hard and use a MILP solver under the hood—see
:doc:`Installation` and :doc:`Using-Custom-Solvers`.

.. contents::
   :local:
   :depth: 2


Quick Start
-----------

.. code-block:: python

   import graphcalc as gc
   from graphcalc.generators import cycle_graph

   G = cycle_graph(5)   # C5

   # Independence number α(G)
   a = gc.independence_number(G)
   print("α(G) =", a)

   # Clique number ω(G)
   w = gc.clique_number(G)
   print("ω(G) =", w)

   # Chromatic number χ(G)
   chi = gc.chromatic_number(G)
   print("χ(G) =", chi)

   # Domination number γ(G)
   g = gc.domination_number(G)
   print("γ(G) =", g)


Choosing a Solver (optional)
----------------------------

GraphCalc auto-detects a solver (see :doc:`Installation`). You can override per-call:

.. code-block:: python

   # Pick CBC if available, and silence logs
   S = gc.maximum_independent_set(
       G,
       solver="cbc",
       solver_options={"msg": False, "timeLimit": 10, "threads": 2},
   )

Or set an environment variable globally:

.. code-block:: bash

   export GRAPHCALC_SOLVER=cbc   # or: highs, glpk, auto

For all supported forms (string / dict / class / instance / callable), see
:doc:`Using-Custom-Solvers`.


Core Recipes
------------

Maximum Independent Set & Independence Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import complete_graph
   G = complete_graph(4)

   S = gc.maximum_independent_set(G)   # set of vertices
   a = gc.independence_number(G)       # |S|

Minimum Vertex Cover & Vertex Cover Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import cycle_graph
   G = cycle_graph(4)

   VC = gc.minimum_vertex_cover(G)     # set of vertices
   tau = gc.vertex_cover_number(G)     # |VC|

Maximum Clique & Clique Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import complete_graph
   G = complete_graph(4)

   C = gc.maximum_clique(G)            # set of vertices
   w = gc.clique_number(G)             # |C|

Chromatic Number & Optimal Coloring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import cycle_graph
   G = cycle_graph(5)                  # C5

   chi = gc.chromatic_number(G)        # 3 for C5
   coloring = gc.optimal_proper_coloring(G)  # {color_index: [vertices]}

Maximum Matching & Matching Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import path_graph
   G = path_graph(6)

   M = gc.maximum_matching(G)          # set of edges (2-tuples)
   nu = gc.matching_number(G)          # |M|

Domination Variants (selected)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import cycle_graph
   G = cycle_graph(5)

   D  = gc.minimum_dominating_set(G)
   g  = gc.domination_number(G)

   DT = gc.minimum_total_domination_set(G)
   gt = gc.total_domination_number(G)

   DI = gc.minimum_independent_dominating_set(G)
   gi = gc.independent_domination_number(G)

   gr  = gc.roman_domination_number(G)
   gdr = gc.double_roman_domination_number(G)

   DR  = gc.minimum_restrained_dominating_set(G)
   grs = gc.restrained_domination_number(G)

Rainbow Domination
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import path_graph
   G = path_graph(4)

   colored, uncolored = gc.minimum_rainbow_dominating_function(G, k=2)
   gr2 = gc.two_rainbow_domination_number(G)
   gr3 = gc.three_rainbow_domination_number(G)


Pandas Knowledge Tables (batch evaluation)
------------------------------------------

GraphCalc includes helpers to evaluate **many** properties on **many** graphs
and return a tidy :mod:`pandas` DataFrame.

Compute a few selected properties for one graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import graphcalc as gc
   from graphcalc.generators import cycle_graph

   G = cycle_graph(6)
   props = ["spectral_radius", "algebraic_connectivity", "order", "size"]

   # Names can refer to functions in `graphcalc` or `networkx`
   row = gc.compute_graph_properties(props, G)   # dict {name: value}
   pd.Series(row)

.. note::

   Solver-backed invariants in this call use the **auto-detected** solver.
   To force a solver for batch runs, set ``GRAPHCALC_SOLVER`` (see
   :doc:`Installation`) or call those functions directly with ``solver=...``.


Compute a table for multiple graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import path_graph, cycle_graph

   G1 = cycle_graph(6)
   G2 = path_graph(5)

   props = ["order", "size", "independence_number", "chromatic_number", "spectral_radius"]
   df = gc.compute_knowledge_table(props, [G1, G2])
   print(df.head())

Expanding list-valued columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some invariants can return lists (e.g., eigenvalue vectors). Use
``expand_list_columns`` to split them into fixed columns:

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({"graph_id": [1, 2, 3], "p_vector": [[3, 0, 1], [2, 1], []]})
   wide = gc.expand_list_columns(df)
   print(wide.columns)   # includes 'p_vector[0]', 'p_vector[1]', ...

Full fingerprint with ``all_properties``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute a comprehensive “fingerprint” across the built-in list
``GRAPHCALC_PROPERTY_LIST``:

.. code-block:: python

   from graphcalc.generators import cycle_graph, path_graph

   G1 = cycle_graph(6)
   G2 = path_graph(5)

   df_all = gc.all_properties([G1, G2])
   print(df_all.columns[:10])   # shows the first few property names

Append another graph to an existing table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from graphcalc.generators import path_graph

   df = gc.all_properties([cycle_graph(5)])
   df = gc.append_graph_row(df, path_graph(4))   # returns a new DataFrame with one extra row


Exporting & downstream analysis
-------------------------------

.. code-block:: python

   # Save to CSV / Parquet
   df_all.to_csv("graphs.csv", index=False)
   df_all.to_parquet("graphs.parquet", index=False)

   # Quick correlations (only numeric columns)
   numeric = df_all.select_dtypes("number")
   corr = numeric.corr()
   print(corr.round(2))


Performance Tips
----------------

- **Solver choice matters.** CBC and HiGHS are strong defaults. Commercial solvers
  (e.g., Gurobi) may be faster on large instances.
- **Time limits / threads**: pass via ``solver_options`` when calling solver-backed
  functions directly.
- **Batch runs**: prefer setting ``GRAPHCALC_SOLVER`` once for the whole session.
- **Start small**: many invariants are NP-hard; large or dense graphs can be slow.


Troubleshooting
---------------

- **“No LP/MIP solver found.”**
  Install one solver (CBC / HiGHS / GLPK) or set ``GRAPHCALC_SOLVER``.
  See :doc:`Installation`.

- **“PuLP: cannot execute highs.”**
  You selected ``HiGHS_CMD`` but the ``highs`` executable isn’t on ``PATH``.
  Install it, or install the Python package ``highspy`` and use ``solver="highs"``.
  Or force CBC: ``solver="cbc"``. See :doc:`Installation`.

- **Long runtimes.**
  Use time limits (``solver_options={"timeLimit": ...}``) or switch to a faster solver.
  Consider heuristics for very large graphs.

See Also
--------

- :doc:`Installation` — installing solvers and verifying detection.
- :doc:`Using-Custom-Solvers` — every way to select/configure a solver.
