import importlib

from graphcalc.metadata import build_module_registry

def test_hypergraph_basic_registry():
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
        assert "definition" in registry[key]
        assert registry[key]["definition"]
