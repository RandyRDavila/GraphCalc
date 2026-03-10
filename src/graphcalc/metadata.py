# src/graphcalc/metadata.py

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Optional


MetadataDict = Dict[str, Any]


def invariant_metadata(
    *,
    display_name: str,
    notation: Optional[str] = None,
    category: Optional[str] = None,
    aliases: tuple[str, ...] = (),
    definition: Optional[str] = None,
) -> Callable:
    r"""
    Attach structured metadata to a GraphCalc invariant or property function.

    Parameters
    ----------
    display_name : str
        Human-readable name of the invariant or property.
    notation : str, optional
        Standard mathematical notation, when applicable.
    category : str, optional
        A high-level grouping such as ``"spectral"`` or
        ``"advanced colorings"``.
    aliases : tuple of str, optional
        Alternative human-readable names for the same concept.

    Returns
    -------
    callable
        A decorator that stores metadata on the decorated function.
    """
    def decorator(func: Callable) -> Callable:
        func._graphcalc_metadata = {
            "display_name": display_name,
            "notation": notation,
            "category": category,
            "aliases": tuple(aliases),
            "definition": definition,
        }
        return func

    return decorator


def get_graphcalc_metadata(obj: Any) -> Optional[MetadataDict]:
    r"""
    Return GraphCalc metadata attached to an object, if present.

    Parameters
    ----------
    obj : object
        Any Python object.

    Returns
    -------
    dict or None
        The attached metadata dictionary, or ``None`` if no GraphCalc
        metadata is present.
    """
    return getattr(obj, "_graphcalc_metadata", None)


def extract_definition_section(obj: Any) -> Optional[str]:
    r"""
    Extract the ``Definition`` section from an object's docstring.

    This function assumes a NumPy/reStructuredText-style section header of
    the form:

    ``Definition``
    ``----------``

    Parameters
    ----------
    obj : object
        A function, class, or other documented object.

    Returns
    -------
    str or None
        The extracted definition text, or ``None`` if no ``Definition``
        section is present.
    """
    doc = inspect.getdoc(obj) or ""
    match = re.search(
        r"^Definition\n-+\n(.+?)(?=\n[A-Z][A-Za-z0-9 ()/,-]*\n-+\n|\Z)",
        doc,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip()


def describe_object(obj: Any) -> MetadataDict:
    r"""
    Build a combined description record for a GraphCalc object.

    Parameters
    ----------
    obj : object
        Any Python object.

    Returns
    -------
    dict
        A dictionary containing attached metadata, the object's name,
        and the extracted docstring definition when available.
    """
    meta = dict(get_graphcalc_metadata(obj) or {})
    meta.setdefault("name", getattr(obj, "__name__", None))
    if not meta.get("definition"):
        meta["definition"] = extract_definition_section(obj)
    return meta


def build_module_registry(module: Any) -> Dict[str, MetadataDict]:
    r"""
    Build a registry of GraphCalc metadata for a module.

    Parameters
    ----------
    module : module
        A Python module to inspect.

    Returns
    -------
    dict
        A dictionary keyed by function or object name for all callables in the
        module carrying GraphCalc metadata.
    """
    registry: Dict[str, MetadataDict] = {}

    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and get_graphcalc_metadata(obj) is not None:
            registry[name] = describe_object(obj)

    return registry
