Examples
========

This section provides example scripts demonstrating advanced use cases for `graphcalc`.

Example 1: Computing the Independence and Domination Numbers
------------------------------------------------------------

.. code-block:: python

    import graphcalc.graphs as gc

    # Create a sample graph
    G = gc.cycle_graph(6)

    # Compute and print the independence number
    print("Independence Number:", gc.independence_number(G))

    # Compute and print the domination number
    print("Domination Number:", gc.domination_number(G))


Example 2: Inspecting Invariant Metadata
----------------------------------------

Many GraphCalc invariants and property functions include structured metadata,
such as a display name, notation, category, aliases, and a machine-readable
definition.

.. code-block:: python

    import graphcalc.graphs as gc
    from graphcalc.metadata import describe_object, get_graphcalc_metadata

    meta = get_graphcalc_metadata(gc.independence_number)

    print("Display name:", meta["display_name"])
    print("Notation:", meta["notation"])
    print("Category:", meta["category"])
    print("Definition:", meta["definition"])

    print()
    print(describe_object(gc.independence_number))


Example 3: Discovering Annotated Functions in a Module
------------------------------------------------------

You can also build a registry of annotated functions from a module.

.. code-block:: python

    import importlib
    from graphcalc.metadata import build_module_registry

    mod = importlib.import_module("graphcalc.hypergraphs.invariants.acyclicity")
    registry = build_module_registry(mod)

    print(sorted(registry))

    for key in sorted(registry):
        print(key, "->", registry[key]["display_name"])
        print("   definition:", registry[key]["definition"])
