"""
Solver and type-checking utilities for graph calculations.

Functions
---------
get_default_solver
    Return the first available LP solver on the system.
require_graph_like
    Decorator to ensure the first argument is a graph-like object.
enforce_type
    Decorator factory to enforce the type of a positional argument.

Type Aliases
------------
GraphLike
    Union of `networkx.Graph` and `SimpleGraph`.
"""

from shutil import which
from functools import wraps
import networkx as nx
from graphcalc.core import SimpleGraph
from typing import Union, Dict, Set, Hashable
import pulp
from pulp import PULP_CBC_CMD, GLPK_CMD, HiGHS_CMD

__all__ = [
    'get_default_solver',
    'require_graph_like',
    'enforce_type',
    'GraphLike',
    '_extract_and_report',
]

GraphLike = Union[nx.Graph, SimpleGraph]
"""Type alias for objects accepted as graphs in this module."""

def get_default_solver():
    """
    Return the first available linear-programming solver backend.

    Checks for installed solver executables in this order:
      1. `highs`  (HiGHS_CMD)
      2. `cbc`    (PULP_CBC_CMD)
      3. `glpsol` (GLPK_CMD)

    Returns
    -------
    pulp.LpSolver_CMD
        An instance of the solver command class, configured with `msg=False`.

    Raises
    ------
    EnvironmentError
        If none of the supported solver executables are found on the PATH.
    """
    if which("highs"):
        return HiGHS_CMD(msg=False)
    elif which("cbc"):
        return PULP_CBC_CMD(msg=False)
    elif which("glpsol"):
        return GLPK_CMD(msg=False)
    else:
        raise EnvironmentError(
            "No supported solver found. Please install one:\n"
            "- brew install cbc or sudo apt install coinor-cbc  (classic)\n"
            "- brew install glpk   (fallback)\n"
            "- brew install highs  (fast, MIT license)\n"
        )

def require_graph_like(func):
    """
    Decorator that enforces the first argument to be graph-like.

    This decorator checks that the wrapped function’s first positional argument
    is an instance of `networkx.Graph` or `graphcalc.core.SimpleGraph`.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        A wrapped function that will raise `TypeError` if the first argument
        is not a supported graph type.

    Raises
    ------
    TypeError
        If the first argument is not a `networkx.Graph` or `SimpleGraph`.
    """
    @wraps(func)
    def wrapper(G, *args, **kwargs):
        if not isinstance(G, (nx.Graph, SimpleGraph)):
            raise TypeError(
                f"Function '{func.__name__}' requires a NetworkX Graph or SimpleGraph "
                f"as the first argument, but got {type(G).__name__}."
            )
        return func(G, *args, **kwargs)
    return wrapper

def enforce_type(arg_index, expected_types):
    """
    Decorator factory to enforce the type of a specific positional argument.

    Parameters
    ----------
    arg_index : int
        Index of the positional argument in `*args` to check.
    expected_types : type or tuple of types
        The expected type(s) for the argument at `arg_index`.

    Returns
    -------
    decorator : callable
        A decorator that wraps a function and raises `TypeError` if the
        specified argument is not an instance of `expected_types`.

    Raises
    ------
    TypeError
        When the argument at `arg_index` is not of type `expected_types`.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[arg_index], expected_types):
                raise TypeError(
                    f"Argument {arg_index} to '{func.__name__}' must be "
                    f"{expected_types}, but got {type(args[arg_index]).__name__}."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _extract_and_report(
    prob: pulp.LpProblem,
    variables: Dict[Hashable, pulp.LpVariable],
    *,
    verbose: bool = False
) -> Set[Hashable]:
    """
    Pulls status, objective, and solution out of a solved LP,
    prints details if verbose=True, and returns the solution set.
    """
    status = pulp.LpStatus[prob.status]
    obj_value = pulp.value(prob.objective)
    solution = {v for v, var in variables.items() if pulp.value(var) == 1}

    if verbose:
        print(f"Solver status: {status}")
        print(f"Objective value: {obj_value}")
        print("Selected nodes in solution:", solution)

    return solution
