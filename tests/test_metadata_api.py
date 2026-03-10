import importlib

import pytest

from graphcalc.metadata import build_module_registry, describe_object


def test_describe_object_on_annotated_function():
    mod = importlib.import_module("graphcalc.hypergraphs.invariants.basic")
    obj = mod.number_of_vertices
    desc = describe_object(obj)
    assert desc is not None
    assert "Number of vertices" in str(desc) or "number_of_vertices" in str(desc)


def test_build_module_registry_excludes_private_helpers():
    mod = importlib.import_module("graphcalc.hypergraphs.invariants.domination")
    registry = build_module_registry(mod)
    assert "_check_k_uniform" not in registry
    assert "_two_section_neighbors" not in registry
    assert "_isolated_vertices_in_two_section" not in registry
