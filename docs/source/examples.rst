Examples
========

This section provides example scripts demonstrating advanced use cases for `graphcalc`.

Example 1: Computing the Independence and Domination Numbers
------------------------------------------------------------

.. code-block:: python

    import graphcalc as gc

    # Create a sample graph
    G = gc.cycle_graph(6)

    # Compute and print the independence number
    print("Independence Number:", gc.independence_number(G))

    # Compute and print the domination number
    print("Domination Number:", gc.domination_number(G))


