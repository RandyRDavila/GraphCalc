import networkx as nx
import graphcalc as gc
import pandas as pd

__all__ = [
    "compute_graph_properties",
    "expand_list_columns",
    "compute_graph_properties_dataframe",
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
    >>> import networkx as nx
    >>> from graphcalc import spectral_radius, algebraic_connectivity
    >>> G = nx.cycle_graph(6)  # A cycle graph with 6 nodes
    >>> function_names = ["spectral_radius", "number_of_nodes"]
    >>> compute_graph_properties(function_names, G)
    {'spectral_radius': 2.0, 'number_of_nodes': 6}

    >>> compute_graph_properties(function_names, G, return_as_dict=False)
    [2.0, 6]
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


def expand_list_columns(df):
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
    >>> expand_list_columns(df)
       graph_id  p_vector[0]  p_vector[1]  p_vector[2]
    0         1          3.0          0.0          1.0
    1         2          2.0          1.0          0.0
    2         3          0.0          0.0          0.0
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


def compute_graph_properties_dataframe(function_names, graphs):
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
    >>> import networkx as nx
    >>> from graphcalc import laplacian_matrix, spectral_radius, algebraic_connectivity
    >>> G1 = nx.cycle_graph(6)
    >>> G2 = nx.path_graph(5)
    >>> function_names = ["spectral_radius", "algebraic_connectivity"]
    >>> graphs = [G1, G2]
    >>> compute_graph_properties_dataframe(function_names, graphs)
       spectral_radius  algebraic_connectivity
    0             2.0                    2.0
    1             1.618033988749895         0.3819660112501051
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
