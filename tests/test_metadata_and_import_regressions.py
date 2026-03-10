import importlib

import pytest


def test_public_package_imports():
    for name in [
        "graphcalc",
        "graphcalc.graphs",
        "graphcalc.hypergraphs",
        "graphcalc.quantum",
        "graphcalc.utils",
    ]:
        mod = importlib.import_module(name)
        assert mod is not None


def test_known_leaf_module_imports():
    for name in [
        "graphcalc.graphs.core.basics",
        "graphcalc.graphs.core.neighborhoods",
        "graphcalc.graphs.generators.simple_graphs",
        "graphcalc.hypergraphs.utils",
        "graphcalc.hypergraphs.invariants.basic",
        "graphcalc.hypergraphs.invariants.acyclicity",
    ]:
        mod = importlib.import_module(name)
        assert mod is not None


def test_build_module_registry_for_hypergraph_basic():
    metadata = importlib.import_module("graphcalc.metadata")
    build_module_registry = getattr(metadata, "build_module_registry")

    mod = importlib.import_module("graphcalc.hypergraphs.invariants.basic")
    registry = build_module_registry(mod)

    expected = {
        "number_of_vertices": "Number of vertices",
        "number_of_edges": "Number of edges",
        "is_empty": "Emptiness",
        "rank": "Rank",
        "co_rank": "Co-rank",
        "maximum_degree": "Maximum degree",
        "minimum_degree": "Minimum degree",
        "average_degree": "Average degree",
    }

    for key, display_name in expected.items():
        assert key in registry
        assert registry[key]["display_name"] == display_name
        assert registry[key]["definition"]
        assert registry[key]["category"]


def test_build_module_registry_for_hypergraph_acyclicity():
    metadata = importlib.import_module("graphcalc.metadata")
    build_module_registry = getattr(metadata, "build_module_registry")

    mod = importlib.import_module("graphcalc.hypergraphs.invariants.acyclicity")
    registry = build_module_registry(mod)

    expected = {
        "is_alpha_acyclic": "Alpha-acyclicity",
        "berge_girth": "Berge girth",
        "is_berge_acyclic": "Berge-acyclicity",
    }

    for key, display_name in expected.items():
        assert key in registry
        assert registry[key]["display_name"] == display_name
        assert registry[key]["definition"]
        assert registry[key]["category"]


def test_registry_entries_have_basic_shape():
    metadata = importlib.import_module("graphcalc.metadata")
    build_module_registry = getattr(metadata, "build_module_registry")

    module_names = [
        "graphcalc.hypergraphs.invariants.basic",
        "graphcalc.hypergraphs.invariants.acyclicity",
    ]

    for module_name in module_names:
        mod = importlib.import_module(module_name)
        registry = build_module_registry(mod)
        assert isinstance(registry, dict)
        assert registry

        for key, entry in registry.items():
            assert isinstance(key, str)
            assert isinstance(entry, dict)
            assert entry.get("display_name")
            assert entry.get("definition")
            assert entry.get("category")
            aliases = entry.get("aliases", ())
            assert isinstance(aliases, (tuple, list))


def test_describe_object_if_available():
    metadata = importlib.import_module("graphcalc.metadata")
    describe_object = getattr(metadata, "describe_object", None)
    if describe_object is None:
        pytest.skip("describe_object not available")

    mod = importlib.import_module("graphcalc.hypergraphs.invariants.basic")
    obj = mod.number_of_vertices
    desc = describe_object(obj)

    assert desc is not None
    text = str(desc)
    assert "Number of vertices" in text or "number_of_vertices" in text


def test_basic_hypergraph_invariants_small_examples():
    hc = importlib.import_module("graphcalc.hypergraphs")
    basic = importlib.import_module("graphcalc.hypergraphs.invariants.basic")

    H = hc.Hypergraph(E=[{1, 2}, {2, 3}, {2, 3, 4}])

    assert basic.number_of_vertices(H) == 4
    assert basic.number_of_edges(H) == 3
    assert basic.is_empty(H) is False
    assert basic.rank(H) == 3
    assert basic.co_rank(H) == 2
    assert basic.maximum_degree(H) == 3
    assert basic.minimum_degree(H) == 1
    assert basic.average_degree(H) == pytest.approx((1 + 3 + 2 + 1) / 4)


def test_hypergraph_acyclicity_small_examples():
    hc = importlib.import_module("graphcalc.hypergraphs")
    ac = importlib.import_module("graphcalc.hypergraphs.invariants.acyclicity")

    H1 = hc.Hypergraph(E=[{1, 2}, {2, 3}])
    assert ac.is_alpha_acyclic(H1) is True
    assert ac.is_berge_acyclic(H1) is True
    assert ac.berge_girth(H1) is None

    H2 = hc.Hypergraph(E=[{1, 2}, {2, 3}, {1, 3}])
    assert ac.is_berge_acyclic(H2) is False
    assert ac.berge_girth(H2) == 3
