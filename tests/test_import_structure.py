import importlib


PUBLIC_MODULES = [
    "graphcalc",
    "graphcalc.graphs",
    "graphcalc.hypergraphs",
    "graphcalc.quantum",
    "graphcalc.utils",
    "graphcalc.metadata",
]

LEAF_MODULES = [
    "graphcalc.graphs.core.basics",
    "graphcalc.graphs.core.neighborhoods",
    "graphcalc.graphs.generators.simple_graphs",
    "graphcalc.hypergraphs.core.basics",
    "graphcalc.hypergraphs.utils",
    "graphcalc.hypergraphs.invariants.basic",
    "graphcalc.hypergraphs.invariants.acyclicity",
    "graphcalc.hypergraphs.invariants.chromatic",
    "graphcalc.hypergraphs.invariants.codegree",
    "graphcalc.hypergraphs.invariants.configurations",
    "graphcalc.hypergraphs.invariants.domination",
    "graphcalc.hypergraphs.invariants.dsi",
    "graphcalc.hypergraphs.invariants.independence",
    "graphcalc.hypergraphs.invariants.matching",
    "graphcalc.hypergraphs.invariants.partite",
    "graphcalc.hypergraphs.invariants.structure",
    "graphcalc.hypergraphs.invariants.transversals",
    "graphcalc.quantum.invariants",
    "graphcalc.quantum.properties",
    "graphcalc.quantum.channel_invariants",
    "graphcalc.quantum.channel_properties",
    "graphcalc.quantum.measurement_properties",
]


def test_public_modules_import():
    for name in PUBLIC_MODULES:
        assert importlib.import_module(name) is not None


def test_leaf_modules_import():
    for name in LEAF_MODULES:
        assert importlib.import_module(name) is not None
