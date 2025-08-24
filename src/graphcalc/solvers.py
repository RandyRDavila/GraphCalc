# src/graphcalc/solvers.py
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from shutil import which
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pulp
from pulp import GLPK_CMD, HiGHS_CMD, PULP_CBC_CMD

__all__ = [
    "SolverSpec",
    "resolve_solver",
    "get_default_solver",
    "solve_or_raise",
    "with_solver",
    "doctor",
]

# --------------------------------------------------------------------------------------
# Public type alias: accept strings, dict specs, callable factories, classes, or objects
# --------------------------------------------------------------------------------------
SolverSpec = Optional[
    Union[
        str,
        Dict[str, Any],
        pulp.apis.core.LpSolver,
        pulp.apis.core.LpSolver_CMD,
        Callable[[], Union[pulp.apis.core.LpSolver, pulp.apis.core.LpSolver_CMD]],
        Type[pulp.apis.core.LpSolver],
        Type[pulp.apis.core.LpSolver_CMD],
    ]
]

# Friendly user aliases -> canonical PuLP registry keys
_PULP_ALIASES: Dict[str, Optional[str]] = {
    # delegate
    "auto": None,
    "default": None,

    # HiGHS
    "highs": "HiGHS_CMD",
    "highs_cmd": "HiGHS_CMD",
    "highs_cmd.exe": "HiGHS_CMD",
    "highs.exe": "HiGHS_CMD",
    "highs.cmd": "HiGHS_CMD",

    # CBC
    "cbc": "PULP_CBC_CMD",
    "cbc.exe": "PULP_CBC_CMD",
    "coin_cmd": "COIN_CMD",

    # GLPK
    "glpk": "GLPK_CMD",
    "glpsol": "GLPK_CMD",
    "glpsol.exe": "GLPK_CMD",
}

# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
def _ensure_pulp_solver(obj: Any) -> pulp.apis.core.LpSolver:
    """Return obj if it's a PuLP solver instance; otherwise raise TypeError."""
    if isinstance(obj, (pulp.apis.core.LpSolver, pulp.apis.core.LpSolver_CMD)):
        return obj
    raise TypeError(
        f"Expected a PuLP solver instance, got {type(obj).__name__}. "
        "Pass a pulp solver object or a spec I can resolve."
    )


def _exec_names_for(key: str) -> List[str]:
    """Executable basenames we may look for on PATH / Conda."""
    k = key.lower()
    if k in {"highs", "highs_cmd"}:
        return ["highs", "highs.exe"]
    if k == "cbc":
        return ["cbc", "cbc.exe"]
    if k in {"glpk", "glpsol"}:
        return ["glpsol", "glpsol.exe"]
    return []


def _candidate_paths(exec_names: List[str]) -> List[str]:
    """Possible executable paths, including PATH and common Conda locations (Windows)."""
    out: List[str] = []

    # 1) PATH
    for name in exec_names:
        p = which(name)
        if p:
            out.append(p)

    # 2) Conda-style prefixes (Windows puts binaries under <env>\Library\bin\)
    prefixes: List[Path] = []
    for env_var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX"):
        v = os.getenv(env_var)
        if v:
            prefixes.append(Path(v))
    # Also try sys.prefix (useful inside venv/conda)
    prefixes.append(Path(sys.prefix))

    for prefix in prefixes:
        libbin = prefix / "Library" / "bin"
        if libbin.exists():
            for name in exec_names:
                p = libbin / name
                if p.exists():
                    out.append(str(p))

    # 3) Common Unix locations (mostly redundant with PATH, but harmless)
    for name in exec_names:
        for d in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"):
            p = Path(d) / name
            if p.exists():
                out.append(str(p))

    # Deduplicate, preserve order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _build_cmd_solver_from_path(solver_key: str, exec_path: Optional[str], *, msg: bool) -> pulp.apis.core.LpSolver_CMD:
    """Instantiate a command-line solver with a specific executable path."""
    k = solver_key.lower()
    if k in {"highs", "highs_cmd"}:
        return HiGHS_CMD(path=exec_path, msg=msg)
    if k == "cbc":
        return PULP_CBC_CMD(path=exec_path, msg=msg)
    if k in {"glpk", "glpsol"}:
        return GLPK_CMD(path=exec_path, msg=msg)
    raise ValueError(f"Unsupported command solver key for path binding: {solver_key!r}")


# --------------------------------------------------------------------------------------
# Default solver discovery (Windows/Conda aware + env overrides)
# --------------------------------------------------------------------------------------
def get_default_solver(msg: bool = False) -> pulp.apis.core.LpSolver_CMD:
    r"""
    Return the first available LP solver backend, honoring environment overrides.

    Order of preference:
      1) HiGHS
      2) CBC
      3) GLPK

    Environment overrides
    ---------------------
    - ``GRAPHCALC_SOLVER``:
        Friendly name (``"highs"``, ``"cbc"``, ``"glpsol"``) or PuLP key
        (``"HiGHS_CMD"``, ``"PULP_CBC_CMD"``, ``"GLPK_CMD"``), or ``"auto"``.
    - ``GRAPHCALC_SOLVER_PATH``:
        Full path to an executable (e.g., ``C:\miniconda3\envs\gc\Library\bin\highs.exe``).

    Returns
    -------
    pulp.apis.core.LpSolver_CMD
        Configured with ``msg`` according to the argument.
    """
    # 1) Read overrides
    forced = os.getenv("GRAPHCALC_SOLVER", "").strip()
    forced_key = forced or "auto"
    # Map friendly aliases to PuLP keys when necessary
    reg_name = _PULP_ALIASES.get(forced_key.lower(), forced_key)
    forced_path = os.getenv("GRAPHCALC_SOLVER_PATH", "").strip() or None

    # 2) If a path is forced, try to bind it to the chosen (or default) solver
    if forced_path:
        # If no specific solver is forced, try HiGHS -> CBC -> GLPK against this path
        trial_order = (
            ["highs", "cbc", "glpsol"] if reg_name is None else [forced_key]
        )
        for key in trial_order:
            # Attempt to instantiate with the provided path
            try:
                return _build_cmd_solver_from_path(key, forced_path, msg=msg)
            except Exception:
                continue  # fall through

    # 3) If solver name is forced (no path), try registry first (covers GUROBI/CPLEX etc.)
    if reg_name not in (None, ""):
        try:
            solver = pulp.getSolver(reg_name, msg=msg)
            return _ensure_pulp_solver(solver)
        except Exception:
            # If it was a friendly alias like "highs", also try executable discovery
            exec_names = _exec_names_for(forced_key)
            for p in _candidate_paths(exec_names):
                try:
                    return _build_cmd_solver_from_path(forced_key, p, msg=msg)
                except Exception:
                    pass
            # Could not honor forced solver; fall back to auto below.

            # Note: we intentionally do not raise here—fallback keeps behavior robust.

            ...

    # 4) Auto-discovery in preferred order
    for key in ("highs", "cbc", "glpsol"):
        # Try registry first in case the user has a Python API solver installed
        try:
            solver = pulp.getSolver(_PULP_ALIASES.get(key, key), msg=msg)
            return _ensure_pulp_solver(solver)
        except Exception:
            # Try known executable locations
            for p in _candidate_paths(_exec_names_for(key)):
                try:
                    return _build_cmd_solver_from_path(key, p, msg=msg)
                except Exception:
                    continue

    # 5) Nothing found: assemble platform-specific guidance
    if sys.platform.startswith("win"):
        hint = (
            "\nWindows tips:\n"
            "- If you use Conda:  conda install -c conda-forge highs coincbc glpk\n"
            "  (Executables typically land under <env>\\Library\\bin\\.)\n"
            "- Or set:\n"
            "    set GRAPHCALC_SOLVER=highs\n"
            "    set GRAPHCALC_SOLVER_PATH=C:\\path\\to\\highs.exe\n"
        )
    else:
        hint = (
            "\nmacOS/Linux tips:\n"
            "- Homebrew:  brew install highs glpk cbc\n"
            "- Ubuntu:    sudo apt-get install coinor-cbc glpk-utils  (use conda for HiGHS)\n"
            "- Conda:     conda install -c conda-forge highs coincbc glpk\n"
        )

    raise EnvironmentError(
        "No supported LP solver was found.\n"
        "Tried HiGHS, CBC, and GLPK via PuLP registry and common executable locations."
        + hint
    )


# --------------------------------------------------------------------------------------
# Flexible solver resolution (string/dict/class/callable/instance)
# --------------------------------------------------------------------------------------
def resolve_solver(
    solver: SolverSpec,
    *,
    msg: bool = False,
    solver_options: Optional[Dict[str, Any]] = None,
) -> pulp.apis.core.LpSolver:
    """
    Resolve a flexible solver spec into a PuLP solver instance.

    Accepts
    -------
    - None
        Use :func:`get_default_solver`.
    - str
        PuLP registry name (e.g., "HiGHS_CMD", "GUROBI_CMD") or friendly alias
        ("auto", "highs", "cbc", "glpsol").
    - dict
        {"name": <str>, "options": {...}} forwarded to :func:`pulp.getSolver`.
    - class
        Subclass of PuLP solver; will be instantiated with ``solver_options``.
    - callable
        Zero-argument factory returning a PuLP solver instance.
    - instance
        An already-constructed PuLP solver object.

    Notes
    -----
    - ``msg`` is propagated when we construct the solver (string/dict/class).
      If the caller provides an instance, we do not override its settings.
    """
    opts = dict(solver_options or {})
    opts.setdefault("msg", msg)

    # None -> default
    if solver is None:
        s = get_default_solver(msg=msg)
        if hasattr(s, "msg"):
            s.msg = msg
        return s

    # Already-constructed
    if isinstance(solver, (pulp.apis.core.LpSolver, pulp.apis.core.LpSolver_CMD)):
        return solver

    # Callable factory
    if callable(solver) and not inspect.isclass(solver):
        return _ensure_pulp_solver(solver())

    # Dict spec
    if isinstance(solver, dict):
        name = solver.get("name")
        if not isinstance(name, str):
            raise ValueError("Dict solver spec must include a string 'name'.")
        local_opts = dict(solver.get("options") or {})
        local_opts.setdefault("msg", msg)
        s = pulp.getSolver(_PULP_ALIASES.get(name.lower(), name), **local_opts)
        return _ensure_pulp_solver(s)

    # Class
    if inspect.isclass(solver):
        if not issubclass(solver, (pulp.apis.core.LpSolver, pulp.apis.core.LpSolver_CMD)):
            raise TypeError("Solver class must subclass a PuLP solver base.")
        return solver(**opts)

    # String
    if isinstance(solver, str):
        key = solver.strip()
        reg_name = _PULP_ALIASES.get(key.lower(), key)
        if reg_name is None:  # "auto"/"default"
            return resolve_solver(None, msg=msg, solver_options=opts)
        s = pulp.getSolver(reg_name, **opts)
        return _ensure_pulp_solver(s)

    raise TypeError(
        "Unrecognized solver specification. "
        "Use None, a string, a dict {'name','options'}, a callable returning a PuLP solver, "
        "a PuLP solver class, or a PuLP solver instance."
    )


# --------------------------------------------------------------------------------------
# Uniform solve & error handling
# --------------------------------------------------------------------------------------
def solve_or_raise(prob: pulp.LpProblem, solver: pulp.apis.core.LpSolver) -> None:
    """Solve a PuLP problem and raise ValueError if the status is not Optimal."""
    prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        raise ValueError(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")


# --------------------------------------------------------------------------------------
# Decorator to inject uniform solver kwargs + solve() helper into invariants
# --------------------------------------------------------------------------------------
from functools import wraps

def with_solver(fn):
    """
    Decorator that adds a uniform `(verbose=False, *, solver=None, solver_options=None)`
    interface to solver-backed functions without repeating boilerplate.

    The wrapped function must accept parameters `(verbose: bool = False, solve=None, **kwargs)`.
    The decorator injects a `solve(prob)` closure that resolves the solver with
    `resolve_solver` and calls `solve_or_raise`.
    """
    @wraps(fn)
    def wrapper(*args, verbose: bool = False, solver: SolverSpec = None,
                solver_options: Optional[Dict[str, Any]] = None, **kwargs):
        def _solve(prob: pulp.LpProblem):
            s = resolve_solver(solver, msg=verbose, solver_options=solver_options)
            solve_or_raise(prob, s)
        return fn(*args, verbose=verbose, solve=_solve, **kwargs)
    return wrapper


# --------------------------------------------------------------------------------------
# Diagnostics for users ("graphcalc doctor")
# --------------------------------------------------------------------------------------
def doctor() -> str:
    r"""
    Return a multi-line diagnostic string describing which solver GraphCalc would use
    right now, including how it was found.

    Examples
    --------
    >>> out = doctor()
    >>> isinstance(out, str)
    True
    >>> out.startswith("GraphCalc Solver Doctor")
    True
    >>> "Selected" in out and "Path trial(s)" in out
    True
    """
    lines: List[str] = []
    lines.append("GraphCalc Solver Doctor")
    lines.append("-----------------------")

    forced = os.getenv("GRAPHCALC_SOLVER", "").strip() or "(none)"
    forced_path = os.getenv("GRAPHCALC_SOLVER_PATH", "").strip() or "(none)"
    lines.append(f"Preferred (env) : GRAPHCALC_SOLVER={forced}")
    lines.append(f"Forced path     : GRAPHCALC_SOLVER_PATH={forced_path}")

    # What would resolve_solver(None) pick?
    try:
        s = get_default_solver(msg=False)
        cls = s.__class__.__name__
        # try to pull path attr for CMD solvers
        path_attr = getattr(s, "path", None)
        path_disp = str(path_attr) if path_attr else "(registry)"
        lines.append(f"Selected        : {cls}  [{path_disp}]")
    except Exception as e:
        lines.append(f"Selected        : <none> ({type(e).__name__}: {e})")

    # Show PATH trials
    trials = []
    for key in ("highs", "cbc", "glpsol"):
        for p in _candidate_paths(_exec_names_for(key)):
            trials.append(f"{key}: {p}")
    if trials:
        lines.append("Path trial(s)   :")
        for t in trials:
            lines.append(f"  - {t}")
    else:
        lines.append("Path trial(s)   : (no candidates found)")

    return "\n".join(lines)
