Examples
========

This section provides example scripts demonstrating advanced use cases for `graphcalc`.

Example 1: Computing the Independence and Domination Numbers
------------------------------------------------------------

.. code-block:: python

    import networkx as nx
    from graphcalc import independence_number, domination_number

    # Create a sample graph
    G = nx.cycle_graph(6)

    # Compute and print the independence number
    print("Independence Number:", independence_number(G))

    # Compute and print the domination number
    print("Domination Number:", domination_number(G))


