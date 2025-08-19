import networkx as nx
import graphcalc as gc
import pandas as pd

__all__ = [
    "compute_graph_properties",
    "expand_list_columns",
    "compute_knowledge_table",
    "GRAPHCALC_PROPERTY_LIST",
    "all_properties",
    "append_graph_row",
]

def compute_graph_properties(function_names, graph, return_as_dict=True):
    r"""
    Compute graph properties based on a list of function names.

    This function takes a list of string function names (defined in either the `graphcalc` or
    `networkx` packages) and a NetworkX graph as input. It computes the values of these functions
    on the given graph and returns the results either as a dictionary (default) or a list.

    Parameters
    ----------
    function_names : list of str
        A list of function names (as strings) defined in the `graphcalc` or `networkx` packages.
    graph : networkx.Graph
        The input graph on which the functions will be evaluated.
    return_as_dict : bool, optional
        If True (default), returns a dictionary mapping function names to their computed values.
        If False, returns a list of computed values in the same order as the input `function_names`.

    Returns
    -------
    dict or list
        By default, a dictionary where keys are function names and values are the computed values.
        If `return_as_dict=False`, a list of computed values is returned.

    Raises
    ------
    AttributeError
        If a function name in `function_names` does not exist in either `graphcalc` or `networkx`.
    Exception
        If any function in `function_names` raises an error during execution.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph
    >>> G = cycle_graph(6)  # A cycle graph with 6 nodes
    >>> function_names = ["spectral_radius", "number_of_nodes"]
    >>> dictionary_solution = gc.compute_graph_properties(function_names, G)
    >>> list_solution = gc.compute_graph_properties(function_names, G, return_as_dict=False)
    """


    # Collect results
    results = {}
    for func_name in function_names:
        func = None
        # Check for function in graphcalc
        if hasattr(gc, func_name):
            func = getattr(gc, func_name)
        # Check for function in networkx
        elif hasattr(nx, func_name):
            func = getattr(nx, func_name)
        else:
            raise AttributeError(
                f"Function '{func_name}' does not exist in either 'graphcalc' or 'networkx'."
            )

        # Try to execute the function on the graph
        try:
            results[func_name] = func(graph)
        except Exception as e:
            raise Exception(f"Error while executing function '{func_name}': {e}")

    # Return results as a dictionary or a list
    if return_as_dict:
        return results
    else:
        return list(results.values())


def expand_list_columns(df: pd.DataFrame):
    r"""
    Expand columns with list entries into separate columns.

    For each column in the dataframe that contains lists as entries, this function:
    1. Finds the maximum length (N) of the lists in the column.
    2. Creates new columns for each index in the list, named as "<column_name>[i]".
    3. Fills missing entries with 0 for lists shorter than N.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe with list-valued columns.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with list-valued columns expanded into separate columns.

    Examples
    --------
    >>> data = {'graph_id': [1, 2, 3],
    ...         'p_vector': [[3, 0, 1], [2, 1], []]}
    >>> df = pd.DataFrame(data)
    >>> new_df = expand_list_columns(df)
    """
    df_expanded = df.copy()

    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            # Find the maximum list length in the column
            max_length = df[column].apply(lambda x: len(x) if isinstance(x, list) else 0).max()

            # Expand the column into separate columns
            for i in range(max_length):
                new_column_name = f"{column}[{i}]"
                df_expanded[new_column_name] = df[column].apply(
                    lambda x: x[i] if isinstance(x, list) and i < len(x) else 0
                )

            # Drop the original list column
            df_expanded.drop(columns=[column], inplace=True)

    return df_expanded


def compute_knowledge_table(function_names: list, graphs: list) -> pd.DataFrame:
    r"""
    Compute graph properties for a collection of NetworkX graphs and return a pandas DataFrame.

    This function takes a list of string function names (defined in the `graphcalc` package)
    and a collection of NetworkX graphs. It computes the specified properties for each graph
    and organizes the results in a DataFrame, where each row corresponds to a graph instance
    and each column corresponds to a function name and its computed value.

    Parameters
    ----------
    function_names : list of str
        A list of function names (as strings) defined in the `graphcalc` package.
    graphs : list of networkx.Graph
        A collection of NetworkX graphs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a graph and each column represents a computed
        graph property.

    Raises
    ------
    AttributeError
        If a function name in `function_names` does not exist in the `graphcalc` package.
    Exception
        If any function in `function_names` raises an error during execution for any graph.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import path_graph, cycle_graph
    >>> G1 = cycle_graph(6)
    >>> G2 = path_graph(5)
    >>> function_names = ["spectral_radius", "algebraic_connectivity"]
    >>> graphs = [G1, G2]
    >>> df = gc.compute_knowledge_table(function_names, graphs)
    """
    # Initialize a list to store results for each graph
    rows = []
    for graph in graphs:
        try:
            # Compute graph properties for this graph
            graph_properties = compute_graph_properties(function_names, graph)
            rows.append(graph_properties)
        except Exception as e:
            raise Exception(f"Error while processing a graph: {e}")

    # Create a DataFrame from the results
    df = pd.DataFrame(rows)
    return expand_list_columns(df)

GRAPHCALC_PROPERTY_LIST = [
    'order',
    'size',
    'connected',
    'diameter',
    'radius',
    'average_shortest_path_length',
    'bipartite',
    'chordal',
    'cubic',
    'eulerian',
    'planar',
    'regular',
    'subcubic',
    'tree',
    'K_4_free',
    'triangle_free',
    'claw_free',
    'planar',
    'cograph',
    'nontrivial',
    "independence_number",
    "clique_number",
    "chromatic_number",
    "vertex_cover_number",
    "edge_cover_number",
    "matching_number",
    "triameter",
    'average_degree',
    'maximum_degree',
    'minimum_degree',
    "slater",
    "sub_total_domination_number",
    "annihilation_number",
    "residue",
    "harmonic_index",
    "domination_number",
    "total_domination_number",
    "independent_domination_number",
    "outer_connected_domination_number",
    "roman_domination_number",
    "double_roman_domination_number",
    "two_rainbow_domination_number",
    "three_rainbow_domination_number",
    "min_maximal_matching_number",
    "restrained_domination_number",
    'algebraic_connectivity',
    'spectral_radius',
    'largest_laplacian_eigenvalue',
    'zero_adjacency_eigenvalues_count',
    'second_largest_adjacency_eigenvalue',
    'smallest_adjacency_eigenvalue',
    "zero_forcing_number",
    "two_forcing_number",
    "total_zero_forcing_number",
    "connected_zero_forcing_number",
    "positive_semidefinite_zero_forcing_number",
    "power_domination_number",
    "well_splitting_number",
]

def all_properties(graphs: list) -> pd.DataFrame:
    """
    Compute the full knowledge table of graph properties and invariants.

    This function evaluates all available invariants and Boolean properties
    implemented in the `graphcalc` package (as listed in
    ``GRAPHCALC_PROPERTY_LIST``) on each graph in the input collection.
    The results are aggregated into a pandas DataFrame, where each row
    corresponds to a graph instance and each column corresponds to a
    specific property or invariant.

    Parameters
    ----------
    graphs : list of networkx.Graph
        A collection of NetworkX graphs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where:
          - Rows correspond to the input graphs.
          - Columns correspond to the full set of `graphcalc` invariants
            and Boolean properties defined in ``GRAPHCALC_PROPERTY_LIST``.

    Raises
    ------
    Exception
        If any property function raises an error during execution for any graph.

    Notes
    -----
    This is a convenience wrapper around :func:`compute_knowledge_table`
    that uses the complete list of invariants and properties available in
    the `graphcalc` package. Use this if you want a comprehensive "fingerprint"
    of each graph in your dataset.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph, path_graph
    >>> G1 = cycle_graph(6)
    >>> G2 = path_graph(5)
    >>> df = gc.all_properties([G1, G2])
    >>> df.columns[:5]  # show a few property names
    Index(['order', 'size', 'connected', 'diameter', 'radius'], dtype='object')
    """
    return compute_knowledge_table(GRAPHCALC_PROPERTY_LIST, graphs)

def append_graph_row(df: pd.DataFrame, G) -> pd.DataFrame:
    """
    Append a new row to an existing knowledge table with the properties
    of a new graph.

    Parameters
    ----------
    df : pandas.DataFrame
        Existing knowledge table (as returned by `compute_full_knowledge_table`
        or `compute_knowledge_table`).
    G : networkx.Graph
        A new graph to analyze.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the additional row for G.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.generators import cycle_graph, path_graph
    >>> df = gc.all_properties([cycle_graph(5)])
    >>> df.shape[0]
    1
    >>> df = gc.append_graph_row(df, path_graph(4))
    >>> df.shape[0]
    2
    """
    row = compute_graph_properties(GRAPHCALC_PROPERTY_LIST, G)
    # turn dict -> DataFrame (1 row), then concat
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
