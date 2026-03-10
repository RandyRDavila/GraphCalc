import importlib

MODULES = [
    "graphcalc.metadata",
    "graphcalc.utils",
    "graphcalc.graphs.core.basics",
    "graphcalc.graphs.core.neighborhoods",
    "graphcalc.hypergraphs.utils",
    "graphcalc.hypergraphs.invariants.basic",
]

def test_import_smoke():
    for name in MODULES:
        mod = importlib.import_module(name)
        assert mod is not None
