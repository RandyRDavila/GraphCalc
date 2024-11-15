Usage Guide
===========

This guide demonstrates how to use `graphcalc` for computing various graph invariants.


Basic Usage Example
-------------------
The following example demonstrates how to calculate the independence number of a graph using `graphcalc`.

.. code-block:: python

   import networkx as nx
   import graphcalc as gc

   # Create a sample graph
   G = nx.cycle_graph(5)

   # Calculate and print the independence number
   print("Independence Number:", gc.independence_number(G))


Here’s how to calculate the domination number of a graph using `graphcalc`.

.. code-block:: python

   import networkx as nx
   import graphcalc as gc

   # Create a sample graph
   G = nx.cycle_graph(5)

   # Calculate and print the domination number
   print("Independence Number:", gc.domination_number(G))