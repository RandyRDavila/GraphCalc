import importlib

from graphcalc.metadata import build_module_registry

MODULES = [
    "graphcalc.hypergraphs.invariants.acyclicity",
    "graphcalc.hypergraphs.invariants.basic",
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


def test_registry_entries_have_required_shape():
    for module_name in MODULES:
        mod = importlib.import_module(module_name)
        registry = build_module_registry(mod)

        assert isinstance(registry, dict)
        assert registry

        for key, entry in registry.items():
            assert isinstance(key, str)
            assert isinstance(entry, dict)
            assert entry.get("display_name")
            assert entry.get("notation") is not None
            assert entry.get("category")
            assert entry.get("definition")
            aliases = entry.get("aliases", ())
            assert isinstance(aliases, (tuple, list))


def test_no_duplicate_display_names_within_module():
    for module_name in MODULES:
        mod = importlib.import_module(module_name)
        registry = build_module_registry(mod)
        display_names = [entry["display_name"] for entry in registry.values()]
        assert len(display_names) == len(set(display_names))
