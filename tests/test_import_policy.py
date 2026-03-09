import ast
from pathlib import Path

ROOT = Path("src/graphcalc")


def imported_modules(pyfile: Path) -> set[str]:
    tree = ast.parse(pyfile.read_text(), filename=str(pyfile))
    mods = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)

    return mods


def test_utils_does_not_import_package_aggregators():
    mods = imported_modules(ROOT / "utils.py")
    forbidden = {
        "graphcalc.graphs",
        "graphcalc.hypergraphs",
        "graphcalc.quantum",
    }
    assert forbidden.isdisjoint(mods)


def test_graph_core_modules_do_not_import_graphs_aggregator():
    for path in (ROOT / "graphs" / "core").glob("*.py"):
        if path.name == "__init__.py":
            continue
        mods = imported_modules(path)
        assert "graphcalc.graphs" not in mods, str(path)


def test_graph_generator_modules_do_not_import_graphs_aggregator():
    for path in (ROOT / "graphs" / "generators").glob("*.py"):
        if path.name == "__init__.py":
            continue
        mods = imported_modules(path)
        assert "graphcalc.graphs" not in mods, str(path)


def test_hypergraph_core_modules_do_not_import_hypergraphs_aggregator():
    for path in (ROOT / "hypergraphs" / "core").glob("*.py"):
        if path.name == "__init__.py":
            continue
        mods = imported_modules(path)
        assert "graphcalc.hypergraphs" not in mods, str(path)
