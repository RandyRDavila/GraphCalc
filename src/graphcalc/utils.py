# src/graphcalc/utils.py
"""
General utilities for GraphCalc.

This module intentionally avoids solver resolution/selection logic;
that all lives in :mod:`graphcalc.solvers`.

Exports
-------
GraphLike
    Union of :class:`networkx.Graph` and :class:`graphcalc.core.SimpleGraph`.
enforce_type
    Decorator factory to enforce the type of a positional argument.
require_graph_like
    Decorator ensuring the first argument is a graph-like object.
_extract_and_report
    Helper to read solution status/objective/variables from a solved PuLP model.

Convenience re-exports from :mod:`graphcalc.solvers`
---------------------------------------------------
get_default_solver, resolve_solver, with_solver, solve_or_raise, SolverSpec
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Dict, Hashable, Set, Union

import networkx as nx
import pulp

from graphcalc.core import SimpleGraph

# Re-export solver utilities for convenience (no local solver code here)
from graphcalc.solvers import (  # noqa: F401
    get_default_solver,
    resolve_solver,
    with_solver,
    solve_or_raise,
    SolverSpec,
)

__all__ = [
    # local
    "GraphLike",
    "require_graph_like",
    "enforce_type",
    "_extract_and_report",
    # re-exports (public API)
    "get_default_solver",
    "resolve_solver",
    "with_solver",
    "solve_or_raise",
    "SolverSpec",
]

# --------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------
GraphLike = Union[nx.Graph, SimpleGraph]
"""Type alias for objects accepted as graphs in GraphCalc."""

# --------------------------------------------------------------------------------------
# Decorators
# --------------------------------------------------------------------------------------
def require_graph_like(func):
    """
    Decorator that enforces the first argument to be graph-like.

    Checks that the wrapped function’s first positional argument is an instance
    of :class:`networkx.Graph` or :class:`graphcalc.core.SimpleGraph`.

    Raises
    ------
    TypeError
        If the first argument is not a supported graph type.
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


def enforce_type(arg_index: int, expected_types):
    """
    Decorator factory to enforce the type of a specific positional argument.

    Parameters
    ----------
    arg_index : int
        Index of the positional argument in ``*args`` to check.
    expected_types : type or tuple[type, ...]
        The expected type(s) for the argument at ``arg_index``.

    Raises
    ------
    TypeError
        When the argument at ``arg_index`` is not of type ``expected_types``.
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

# --------------------------------------------------------------------------------------
# Small helper for extracting model solutions
# --------------------------------------------------------------------------------------
def _extract_and_report(
    prob: pulp.LpProblem,
    variables: Dict[Hashable, pulp.LpVariable],
    *,
    verbose: bool = False,
) -> Set[Hashable]:
    """
    Extract a 0–1 solution from a solved PuLP model, optionally printing details.

    Parameters
    ----------
    prob : pulp.LpProblem
        A solved PuLP model.
    variables : dict[hashable, pulp.LpVariable]
        Decision variables keyed by the object they represent (node, edge, color, etc.).
    verbose : bool, default=False
        If True, print solver status, objective value, and the extracted set.

    Returns
    -------
    set of hashable
        Keys whose corresponding variable has value 1 (within a >0.5 threshold).
    """
    status = pulp.LpStatus.get(prob.status, str(prob.status))
    obj_value = pulp.value(prob.objective)
    solution = {k for k, var in variables.items() if pulp.value(var) > 0.5}

    if verbose:
        print(f"Solver status : {status}")
        print(f"Objective     : {obj_value}")
        print(f"Selected keys : {solution}")

    return solution
