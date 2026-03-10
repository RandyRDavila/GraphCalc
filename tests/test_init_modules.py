from pathlib import Path

ROOT = Path("src/graphcalc")


def _active_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def test_graphs_init_exports_core_data_generators():
    lines = _active_lines(ROOT / "graphs" / "__init__.py")
    text = "\n".join(lines)

    assert "from . import core as core" in text
    assert "from . import data as data" in text
    assert "from . import generators as generators" in text
    assert "from .core import *" in text
    assert "from .data import *" in text
    assert "from .generators import *" in text


def test_hypergraphs_init_does_not_eagerly_import_invariants():
    lines = _active_lines(ROOT / "hypergraphs" / "__init__.py")
    text = "\n".join(lines)

    assert "from . import invariants" not in text
    assert "from .invariants import *" not in text
