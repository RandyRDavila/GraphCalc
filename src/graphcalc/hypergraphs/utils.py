# src/graphcalc/hypergraphs/utils.py
from __future__ import annotations

from functools import wraps
from typing import TypeAlias

from graphcalc.hypergraphs.core.basics import Hypergraph

__all__ = [
    "HypergraphLike",
    "require_hypergraph_like",
]

HypergraphLike: TypeAlias = Hypergraph


def require_hypergraph_like(func):
    @wraps(func)
    def wrapper(H, *args, **kwargs):
        if not isinstance(H, Hypergraph):
            raise TypeError(
                f"Function '{func.__name__}' requires a Hypergraph "
                f"as the first argument, but got {type(H).__name__}."
            )
        return func(H, *args, **kwargs)
    return wrapper
