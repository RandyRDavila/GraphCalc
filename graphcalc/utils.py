from shutil import which
from pulp import PULP_CBC_CMD, GLPK_CMD, HiGHS_CMD

__all__ = [
    'get_default_solver',
    'require_graph_like',
    'enforce_type',
    'GraphLike'
]

def get_default_solver():
    if which("highs"):
        return HiGHS_CMD(msg=False)
    elif which("cbc"):
        return PULP_CBC_CMD(msg=False)
    elif which("glpsol"):
        return GLPK_CMD(msg=False)
    else:
        raise EnvironmentError(
            "No supported solver found. Please install one:\n"
            "- brew install highs  (fast, MIT license)\n"
            "- brew install cbc    (classic)\n"
            "- brew install glpk   (fallback)"
        )


from functools import wraps
import networkx as nx
from graphcalc.core import SimpleGraph

def require_graph_like(func):
    @wraps(func)
    def wrapper(G, *args, **kwargs):
        if not isinstance(G, (nx.Graph, SimpleGraph)):
            raise TypeError(
                f"Function '{func.__name__}' requires a NetworkX Graph or SimpleGraph as the first argument, "
                f"but got {type(G).__name__}."
            )
        return func(G, *args, **kwargs)
    return wrapper

def enforce_type(arg_index, expected_types):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[arg_index], expected_types):
                raise TypeError(
                    f"Argument {arg_index} to '{func.__name__}' must be {expected_types}, "
                    f"but got {type(args[arg_index]).__name__}."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


from typing import Union

GraphLike = Union[nx.Graph, SimpleGraph]
