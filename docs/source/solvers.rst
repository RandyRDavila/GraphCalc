Using Custom Solvers
====================

GraphCalc can use many MILP solvers via `PuLP <https://coin-or.github.io/pulp/>`_.
This page shows **all the ways** you can pick or configure a solver when calling
GraphCalc functions (e.g., :func:`graphcalc.invariants.classics.maximum_independent_set`).

.. contents::
   :local:
   :depth: 2


Overview
--------

- If you **do nothing**, GraphCalc will auto-detect a solver in this order:

  1. ``HiGHS`` (Python API) if the package ``highspy`` is installed.
  2. A command-line solver found on ``PATH`` (in order): ``highs`` → ``cbc`` → ``glpsol``.

- You can **override** the choice:
  - per-call via the ``solver=...`` argument (several forms supported),
  - or globally via environment variables (``GRAPHCALC_SOLVER``, ``GRAPHCALC_SOLVER_PATH``).

- Extra arguments for solver configuration can be passed through
  ``solver_options={...}`` (when applicable; see below).


Accepted Forms for ``solver=...``
---------------------------------

You can pass any of the following to the ``solver=`` parameter.

**1) ``None`` (default / auto-detect)**

.. code-block:: python

   S = gc.maximum_independent_set(G)  # auto-detect

**2) String name** (friendly names or PuLP registry names)

.. code-block:: python

   S = gc.maximum_independent_set(G, solver="cbc")    # doctest: +SKIP
   S = gc.maximum_independent_set(G, solver="highs")  # doctest: +SKIP
   S = gc.maximum_independent_set(G, solver="glpk")   # doctest: +SKIP
   # also works with official PuLP names, e.g. "GUROBI_CMD" (if available)

**3) Dict: ``{"name": <str>, "options": {...}}``** (forwarded to ``pulp.getSolver``)

.. code-block:: python

   S = gc.maximum_independent_set(
       G,
       solver={"name": "GUROBI_CMD", "options": {"timeLimit": 10}},  # doctest: +SKIP
   )

**4) Class** (a PuLP solver class; GraphCalc instantiates it for you)

.. code-block:: python

   from pulp import PULP_CBC_CMD
   S = gc.maximum_independent_set(G, solver=PULP_CBC_CMD)  # doctest: +SKIP

**5) Instance** (an already-constructed PuLP solver)

.. code-block:: python

   from pulp import PULP_CBC_CMD
   S = gc.maximum_independent_set(G, solver=PULP_CBC_CMD(msg=True))  # doctest: +SKIP

**6) Callable** (returning a PuLP solver instance)

.. code-block:: python

   from pulp import PULP_CBC_CMD
   S = gc.maximum_independent_set(G, solver=lambda: PULP_CBC_CMD(msg=False, threads=2))  # doctest: +SKIP


Passing ``solver_options``
--------------------------

When you pass a **string** or a **class**, you can also provide
``solver_options={...}``. These are forwarded to the underlying solver
constructor (e.g., ``pulp.getSolver("cbc", **opts)`` or ``PULP_CBC_CMD(**opts)``).

.. code-block:: python

   # String + solver_options
   S = gc.maximum_independent_set(
       G,
       solver="cbc",
       solver_options={"msg": False, "timeLimit": 5, "threads": 2},  # doctest: +SKIP
   )

   # Class + solver_options
   from pulp import HiGHS_CMD
   S = gc.maximum_independent_set(
       G,
       solver=HiGHS_CMD,
       solver_options={"msg": True, "timeLimit": 10},  # doctest: +SKIP
   )

**Notes**

- If you pass an **instance** or a **callable**, ``solver_options`` is **ignored**
  (because your instance already encodes its settings).
- The exact option names vary by solver. Common, widely-supported keys include:

  - ``msg`` (bool): turn solver logging on/off.
  - ``timeLimit`` (seconds): wall-clock limit.
  - ``threads`` (int): limit solver threads.
  - Gap keys (CBC often uses ``fracGap`` or ``gapRel``; check PuLP docs).

- If an option isn’t recognized by the selected solver, PuLP may ignore it silently.


Environment Variables (Global Overrides)
----------------------------------------

You can choose a solver globally, without changing code:

.. code-block:: bash

   # one of: highs | cbc | glpk | auto
   export GRAPHCALC_SOLVER=cbc

   # optional: force an exact executable path for CMD-style solvers
   export GRAPHCALC_SOLVER_PATH=/usr/bin/cbc

On Windows PowerShell:

.. code-block:: powershell

   $env:GRAPHCALC_SOLVER = "cbc"
   $env:GRAPHCALC_SOLVER_PATH = "C:\Program Files\cbc\bin\cbc.exe"


See What GraphCalc Will Use
---------------------------

Use the built-in doctor to print the detection decision:

.. code-block:: python

   from graphcalc.solvers import doctor
   print(doctor())

Example:

.. code-block:: text

   GraphCalc Solver Doctor
   -----------------------
   Preferred (env) : GRAPHCALC_SOLVER=(none)
   Forced path     : GRAPHCALC_SOLVER_PATH=(none)
   Selected        : PULP_CBC_CMD  [cbc]
   Path trial(s)   :
     - highs : (not found)
     - cbc   : /usr/bin/cbc
     - glpsol: (not found)


Practical Examples
------------------

**Disable logs and cap runtime**

.. code-block:: python

   S = gc.maximum_clique(
       G,
       solver="cbc",
       solver_options={"msg": False, "timeLimit": 10},  # doctest: +SKIP
   )

**Ask HiGHS (Python API) first, otherwise HiGHS_CMD if installed**

.. code-block:: python

   # If you installed `highspy` (pip), `solver="highs"` will prefer the python API.
   # Without highspy but with the `highs` executable on PATH, it uses HiGHS_CMD.
   S = gc.independence_number(G, solver="highs")  # doctest: +SKIP

**Callable to centralize tuning**

.. code-block:: python

   def tuned_cbc():
       from pulp import PULP_CBC_CMD
       return PULP_CBC_CMD(msg=False, threads=2, timeLimit=5)

   chi = gc.chromatic_number(G, solver=tuned_cbc)  # doctest: +SKIP


Troubleshooting
---------------

**“The solver highs does not exist in PuLP.”**

- You passed a string that PuLP doesn’t recognize in your version, or you used
  the wrong case. Try ``"highs"`` or the registered name ``"HiGHS"`` (via PuLP).
  Alternatively, import the class: ``from pulp import HiGHS_CMD``.

**“PuLP: cannot execute highs.”**

- You selected ``HiGHS_CMD`` but the ``highs`` executable is not on ``PATH``.
  Install it (see :doc:`Installation`) or use the Python package
  ``highspy`` and set ``GRAPHCALC_SOLVER=highs``. As a quick fix,
  force CBC: ``GRAPHCALC_SOLVER=cbc``.

**Windows PATH quirks**

- Prefer ``pip install highspy`` (HiGHS Python API), or
  ``conda install -c conda-forge coincbc`` for CBC.

**Silencing solver output**

- Pass ``verbose=False`` (default) to GraphCalc functions, or set ``msg=False`` via
  ``solver_options`` (when using string/class forms).

**CI stability**

- On Ubuntu 22.04 GitHub runners, install ``coinor-cbc`` and set
  ``GRAPHCALC_SOLVER=cbc``. Or install ``highspy`` and set
  ``GRAPHCALC_SOLVER=highs``. See :doc:`Installation` for ready-to-use YAML.


See Also
--------

- :doc:`Installation` — how to install solvers on your platform.
- PuLP solver docs for detailed option names/behavior.
