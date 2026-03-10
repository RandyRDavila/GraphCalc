# graphcalc/graphs/invariants/critical_invariants.py

# ============================================================
# GENERAL VERTEX/EDGE DELETION (criticality & sensitivity)
# ============================================================

import graphcalc.graphs as gc
from graphcalc.metadata import invariant_metadata

__all__ = [
    "vertex_deletion_deltas",
    "vertex_critical_set",
    "vertex_critical_number",
    "vertex_deletion_max_jump",
    "edge_deletion_deltas",
    "edge_critical_number",
    "domination_vertex_increase_number",
    "domination_vertex_decrease_number",
    "domination_vertex_change_number",
    "domination_vertex_same_number",
    "domination_vertex_max_jump",
    "domination_edge_increase_number",
    "domination_edge_decrease_number",
    "domination_edge_change_number",
    "domination_edge_same_number",
    "domination_edge_max_jump",
]

def vertex_deletion_deltas(G, f):
    r"""
    Compute **single-vertex deletion deltas** for a graph parameter :math:`f`.

    For each vertex :math:`v \in V(G)`, form the induced vertex-deleted subgraph

    .. math::
        G - v \;:=\; G[V(G)\setminus\{v\}],

    and compute the deletion delta

    .. math::
        \Delta_v f(G) \;:=\; f(G - v) - f(G).

    The function returns the mapping :math:`v \mapsto \Delta_v f(G)` for all vertices of :math:`G`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    f : callable
        A function ``f(H) -> number`` defined on graphs ``H``.

        Examples include invariants/parameters such as ``order``, ``size``,
        ``independence_number``, ``clique_number``, ``domination_number``,
        and ``chromatic_number``.

        The function is evaluated first on ``G`` and then on each induced
        subgraph ``G - v``.

    Returns
    -------
    dict
        A dictionary mapping each vertex ``v`` of ``G`` to the numeric value
        ``f(G - v) - f(G)``.

        If :math:`|V(G)|=0`, returns the empty dictionary ``{}``.

    Notes
    -----
    - These deltas are useful for **sensitivity** and **criticality** analyses.
    - The induced subgraph ``G - v`` is passed to ``f`` as a ``.copy()`` to protect against accidental mutation inside ``f``. If ``f`` is guaranteed to be read-only, you may omit ``.copy()`` for speed.

    Raises
    ------
    Exception
        Propagates any exception raised by ``f`` when applied to ``G`` or to any vertex-deleted
        induced subgraph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.path_graph(4)
    >>> d = gc.vertex_deletion_deltas(G, gc.order)
    >>> d[0], d[1], d[2], d[3]
    (-1, -1, -1, -1)
    """
    n = gc.order(G)
    if n == 0:
        return {}

    base = f(G)
    nodes = list(G.nodes())
    node_set = set(nodes)

    deltas = {}
    for v in nodes:
        H = G.subgraph(node_set - {v}).copy()
        deltas[v] = f(H) - base
    return deltas

def vertex_critical_set(G, f, *, kind="change"):
    r"""
    Select vertices by how a graph parameter :math:`f` changes under single-vertex deletion.

    For each vertex :math:`v \in V(G)`, define the vertex-deletion delta

    .. math::

        \Delta_v f(G) \;:=\; f(G - v) - f(G),

    where :math:`G - v` is the induced subgraph obtained by deleting :math:`v`.

    This function returns the subset of vertices classified by the sign (or nonzero-ness)
    of :math:`\Delta_v f(G)`:

    - ``kind='change'``: vertices with :math:`\Delta_v f(G) \neq 0` (deletion changes the value)
    - ``kind='increase'``: vertices with :math:`\Delta_v f(G) > 0` (value increases after deletion)
    - ``kind='decrease'``: vertices with :math:`\Delta_v f(G) < 0` (value decreases after deletion)
    - ``kind='same'``: vertices with :math:`\Delta_v f(G) = 0` (value unchanged after deletion)

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    f : callable
        A function ``f(H) -> number`` defined on graphs ``H``. The value is evaluated on ``G`` and on
        each induced subgraph ``G - v``.
    kind : {'change', 'increase', 'decrease', 'same'}, optional
        Which deletion behavior to select.

    Returns
    -------
    set
        A set of vertices ``v`` satisfying the requested deletion behavior with respect to :math:`f`.

        If :math:`|V(G)|=0`, returns the empty set.

    Raises
    ------
    ValueError
        If ``kind`` is not one of ``{'change','increase','decrease','same'}``.
    Exception
        Propagates any exception raised by ``f`` when evaluated on ``G`` or on any vertex-deleted
        induced subgraph.

    Notes
    -----
    - This is a generic vertex-sensitivity / “criticality” selector. Terminology varies by context:
        * For **maximization** parameters (e.g. :math:`\alpha`, :math:`\omega`), authors sometimes call vertices with ``kind='decrease'`` *critical*, since deleting them reduces the optimum.
        * For **minimization** parameters (e.g. :math:`\chi`, :math:`\gamma`), authors sometimes call vertices with ``kind='increase'`` *critical*, since deleting them increases the minimum.
    - This function does not assume whether :math:`f` is a max or min parameter; it simply filters by the sign of :math:`\Delta_v f(G)`.
    - The deltas are computed by :func:`vertex_deletion_deltas`, which forms induced subgraphs and
      evaluates ``f`` on them.

    Complexity
    ----------
    Dominated by computing the deltas: :math:`O(|V(G)|)` evaluations of ``f`` on graphs with one vertex
    deleted (plus one evaluation on ``G`` itself). The additional filtering is :math:`O(|V(G)|)`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.path_graph(5)
    >>> # Vertices whose deletion changes the independence number:
    >>> gc.vertex_critical_set(G, gc.independence_number, kind="change")  # doctest: +ELLIPSIS
    {...}
    """
    if kind not in {"change", "increase", "decrease", "same"}:
        raise ValueError("kind must be one of {'change','increase','decrease','same'}")

    deltas = vertex_deletion_deltas(G, f)
    if kind == "change":
        return {v for v, d in deltas.items() if d != 0}
    if kind == "increase":
        return {v for v, d in deltas.items() if d > 0}
    if kind == "decrease":
        return {v for v, d in deltas.items() if d < 0}
    return {v for v, d in deltas.items() if d == 0}  # kind == "same"

def vertex_critical_number(G, f, *, kind="change"):
    r"""
    Count vertices by how a graph parameter :math:`f` changes under single-vertex deletion.

    This is the cardinality of :func:`vertex_critical_set`:

    .. math::

        \bigl|\{\, v \in V(G) : \Delta_v f(G)\ \text{satisfies the requested condition}\,\}\bigr|,

    where

    .. math::

        \Delta_v f(G) \;=\; f(G - v) - f(G)

    and :math:`G-v` denotes the induced subgraph obtained by deleting :math:`v`.

    The ``kind`` argument specifies which condition is used:

    - ``kind='change'``: :math:`\Delta_v f(G) \neq 0`
    - ``kind='increase'``: :math:`\Delta_v f(G) > 0`
    - ``kind='decrease'``: :math:`\Delta_v f(G) < 0`
    - ``kind='same'``: :math:`\Delta_v f(G) = 0`

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    f : callable
        A function ``f(H) -> number`` defined on graphs ``H``.
    kind : {'change', 'increase', 'decrease', 'same'}, optional
        Which deletion behavior to count.

    Returns
    -------
    int
        The number of vertices in :math:`G` whose deletion has the specified effect on :math:`f`.

        If :math:`|V(G)|=0`, this returns 0.

    Raises
    ------
    ValueError
        If ``kind`` is not one of ``{'change','increase','decrease','same'}``.
    Exception
        Propagates any exception raised by ``f`` when evaluated on ``G`` or on a vertex-deleted
        induced subgraph (via :func:`vertex_critical_set` / :func:`vertex_deletion_deltas`).

    Notes
    -----
    This is a thin wrapper around :func:`vertex_critical_set` that returns only the count.

    Complexity
    ----------
    Same as :func:`vertex_critical_set`: dominated by :math:`O(|V(G)|)` evaluations of ``f`` on
    vertex-deleted induced subgraphs (plus one evaluation on ``G``).

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.path_graph(6)
    >>> gc.vertex_critical_number(G, gc.independence_number, kind="change")
    0
    """
    return len(vertex_critical_set(G, f, kind=kind))

def vertex_deletion_max_jump(G, f):
    r"""
    Compute the **maximum one-vertex deletion jump** of a graph parameter :math:`f`.

    For each vertex :math:`v \in V(G)`, consider the vertex-deletion delta

    .. math::

        \Delta_v f(G) \;=\; f(G - v) - f(G),

    where :math:`G - v` is the induced subgraph obtained by deleting :math:`v`.
    This function returns the maximum absolute change over all single-vertex deletions:

    .. math::

        J_{\max}(G;f) \;=\; \max_{v \in V(G)} \bigl|f(G - v) - f(G)\bigr|
        \;=\; \max_{v \in V(G)} |\Delta_v f(G)|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    f : callable
        A function ``f(H) -> number`` defined on graphs ``H``.

    Returns
    -------
    number
        The maximum absolute delta :math:`\max_v |\Delta_v f(G)|`. If :math:`|V(G)| = 0`,
        returns 0.

    Raises
    ------
    Exception
        Propagates any exception raised by ``f`` when evaluated on ``G`` or on any vertex-deleted
        induced subgraph (via :func:`vertex_deletion_deltas`).

    Notes
    -----
    - This is a simple **one-vertex sensitivity** measure for :math:`f`: it quantifies the largest
      single-vertex deletion impact on the parameter value.
    - If :math:`G` is empty, there are no deletions to consider, so the maximum jump is defined here
      to be 0.
    - The deltas are computed by :func:`vertex_deletion_deltas`, which evaluates ``f`` on ``G`` and on
      each induced subgraph ``G-v``.

    Complexity
    ----------
    Dominated by computing the deltas: :math:`O(|V(G)|)` evaluations of ``f`` on graphs with one vertex
    deleted (plus one evaluation on ``G``), and then an :math:`O(|V(G)|)` scan to take the maximum.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.path_graph(6)
    >>> gc.vertex_deletion_max_jump(G, gc.zero_forcing_number)
    1
    """
    deltas = vertex_deletion_deltas(G, f)
    return 0 if not deltas else max(abs(d) for d in deltas.values())

def edge_deletion_deltas(G, f):
    r"""
    Compute **single-edge deletion deltas** for a graph parameter :math:`f`.

    For each edge :math:`e \in E(G)`, form the graph obtained by deleting that edge while
    keeping all vertices, and compute the delta:

    .. math::

        \Delta_e f(G) \;:=\; f(G - e) - f(G),

    where :math:`G-e` denotes the graph with edge :math:`e` removed (vertex set unchanged).

    The function returns the mapping :math:`e \mapsto \Delta_e f(G)` for all edges of :math:`G`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. This routine is primarily intended for **simple undirected graphs**.

        - For a ``DiGraph``, “edge deletion” refers to deleting a directed arc, and edges are ordered.
          This function still iterates ``G.edges()`` but then collapses each edge to an unordered
          key (a ``frozenset``), which is generally **not** appropriate for directed graphs.
        - For a ``MultiGraph`` / ``MultiDiGraph``, parallel edges require edge keys to distinguish
          copies; this function does not accept edge keys and is therefore not suitable without
          adaptation.

    f : callable
        A function ``f(H) -> number`` defined on graphs ``H``. The value is evaluated on ``G`` and on
        each edge-deleted graph ``G-e``.

    Returns
    -------
    dict
        A dictionary mapping each edge to the numeric value ``f(G - e) - f(G)``.

        Edges are keyed as ``frozenset({u, v})`` so they are treated as **unordered**, and node labels
        need not be comparable. For a simple undirected graph, this keying gives a one-to-one
        correspondence between edges and dictionary entries.

        If :math:`|E(G)| = 0`, returns an empty dictionary ``{}``.

    Raises
    ------
    Exception
        Propagates any exception raised by ``f`` when evaluated on ``G`` or on any edge-deleted graph.

    Notes
    -----
    - The implementation copies ``G`` once per edge and deletes that edge. This protects the original
      graph from mutation regardless of whether ``f`` mutates its input.
    - These deltas are useful for edge-sensitivity / edge-criticality analyses: they quantify how
      much the parameter changes when a single edge is removed.

    Complexity
    ----------
    Let :math:`m=|E(G)|`. This routine performs :math:`m+1` evaluations of ``f`` (once on ``G`` and once
    for each of the :math:`m` single-edge deletions). The total runtime is dominated by the cost of
    evaluating ``f`` on graphs with one edge removed, plus the overhead of copying the graph:

    .. math::

        O\!\left(m \cdot T_f(n, m-1)\right),

    where :math:`T_f` is the time to evaluate ``f`` on a graph of the given size.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.cycle_graph(5)
    >>> d = gc.edge_deletion_deltas(G, gc.chromatic_number)
    >>> all(v <= 0 for v in d.values())  # removing an edge cannot increase chi for simple graphs
    True
    """
    if gc.size(G) == 0:
        return {}

    base = f(G)
    deltas = {}

    for (u, v) in G.edges():
        H = G.copy()
        H.remove_edge(u, v)
        e = frozenset((u, v))  # canonical undirected key
        deltas[e] = f(H) - base

    return deltas

def edge_critical_number(G, f, *, kind="change"):
    r"""
    Count edges by how a graph parameter :math:`f` changes under single-edge deletion.

    For each edge :math:`e \in E(G)`, define the edge-deletion delta

    .. math::

        \Delta_e f(G) \;:=\; f(G - e) - f(G),

    where :math:`G-e` is the graph obtained from :math:`G` by deleting the edge :math:`e`
    (keeping all vertices). This function counts how many edges fall into a given sign class
    of :math:`\Delta_e f(G)`:

    - ``kind='change'``: edges with :math:`\Delta_e f(G) \neq 0`
    - ``kind='increase'``: edges with :math:`\Delta_e f(G) > 0`
    - ``kind='decrease'``: edges with :math:`\Delta_e f(G) < 0`
    - ``kind='same'``: edges with :math:`\Delta_e f(G) = 0`

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph, intended for **simple undirected** graphs (so that edges are naturally
        unordered and uniquely determined by their endpoints). The deltas are computed by
        :func:`edge_deletion_deltas`, which is not designed for multigraph edge keys.
    f : callable
        A function ``f(H) -> number`` defined on graphs ``H``.
    kind : {'change', 'increase', 'decrease', 'same'}, optional
        Which deletion behavior to count.

    Returns
    -------
    int
        The number of edges :math:`e` satisfying the requested condition on
        :math:`\Delta_e f(G)`. If :math:`|E(G)|=0`, returns 0.

    Raises
    ------
    ValueError
        If ``kind`` is not one of ``{'change','increase','decrease','same'}``.
    Exception
        Propagates any exception raised by ``f`` when evaluated on ``G`` or on any edge-deleted graph (via :func:`edge_deletion_deltas`).

    Notes
    -----
    - Terminology varies: some authors reserve “edge-critical” for a specific direction of change, e.g.:
        * for **maximization** parameters (such as :math:`\alpha` or :math:`\omega`), edges with ``kind='decrease'``;
        * for **minimization** parameters (such as :math:`\chi`), edges with ``kind='increase'``.
    - This function is agnostic and provides all four sign-based categories via ``kind``.
    - This is a counting wrapper; if you want the edges themselves, define an analogous ``edge_critical_set`` using the same delta predicate.

    Complexity
    ----------
    Dominated by :func:`edge_deletion_deltas`: :math:`O(|E(G)|)` evaluations of ``f`` on graphs with
    one edge deleted (plus one evaluation on ``G``), and then an :math:`O(|E(G)|)` scan to count.

    """
    if kind not in {"change", "increase", "decrease", "same"}:
        raise ValueError("kind must be one of {'change','increase','decrease','same'}")

    deltas = edge_deletion_deltas(G, f)

    if kind == "change":
        return sum(1 for d in deltas.values() if d != 0)
    if kind == "increase":
        return sum(1 for d in deltas.values() if d > 0)
    if kind == "decrease":
        return sum(1 for d in deltas.values() if d < 0)
    return sum(1 for d in deltas.values() if d == 0)  # kind == "same"

@invariant_metadata(
    display_name="Domination vertex increase number",
    notation=r"c_v^{+}(\gamma)(G)",
    category="domination criticality",
    aliases=("domination increase-critical vertex number",),
    definition=(
        "The domination vertex increase number of a graph G is the number of "
        "vertices v such that deleting v increases the domination number, that is, "
        "gamma(G-v) > gamma(G)."
    ),
)
def domination_vertex_increase_number(G):
    r"""
    Count **domination-increase-critical vertices** of a graph.

    Let :math:`\gamma(G)` denote the domination number of a graph :math:`G`. For each vertex
    :math:`v \in V(G)`, define the vertex-deletion delta

    .. math::

        \Delta_v \gamma(G) \;:=\; \gamma(G-v) - \gamma(G),

    where :math:`G-v` is the induced subgraph obtained by deleting :math:`v`.

    This function returns the number of vertices whose deletion **increases** the domination number:

    .. math::

        c_v^+(\gamma)(G)
        \;=\;
        \bigl|\{\, v \in V(G) : \gamma(G-v) > \gamma(G) \,\}\bigr|
        \;=\;
        \bigl|\{\, v \in V(G) : \Delta_v \gamma(G) > 0 \,\}\bigr|.

    Intuition: these are vertices whose removal makes the remaining graph harder to dominate
    (more dominators are needed).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The number of vertices :math:`v` with :math:`\gamma(G-v) > \gamma(G)`.
        Returns 0 if :math:`G` has no vertices.

    Notes
    -----
    - For minimization parameters like :math:`\gamma`, this “increase under deletion” notion is
      the closest analogue of a *critical vertex* in the sense that deleting it worsens the optimum.
    - This is a thin wrapper around :func:`vertex_critical_number` with
      ``f = gc.domination_number`` and ``kind = 'increase'``.

    Complexity
    ----------
    Dominated by :func:`vertex_critical_number` / :func:`vertex_deletion_deltas`: performs
    :math:`O(|V(G)|)` evaluations of ``gc.domination_number`` on vertex-deleted induced subgraphs.

    See Also
    --------
    domination_vertex_decrease_number
    domination_vertex_change_number
    domination_vertex_max_jump
    """
    return vertex_critical_number(G, gc.domination_number, kind="increase")


@invariant_metadata(
    display_name="Domination vertex decrease number",
    notation=r"c_v^{-}(\gamma)(G)",
    category="domination criticality",
    aliases=("domination decrease-critical vertex number",),
    definition=(
        "The domination vertex decrease number of a graph G is the number of "
        "vertices v such that deleting v decreases the domination number, that is, "
        "gamma(G-v) < gamma(G)."
    ),
)
def domination_vertex_decrease_number(G):
    r"""
    Count vertices whose deletion **decreases** the domination number.

    Using :math:`\gamma(G)` for domination number and :math:`\Delta_v\gamma(G)=\gamma(G-v)-\gamma(G)`,
    this function returns

    .. math::

        c_v^-(\gamma)(G)
        \;=\;
        \bigl|\{\, v \in V(G) : \gamma(G-v) < \gamma(G) \,\}\bigr|
        \;=\;
        \bigl|\{\, v \in V(G) : \Delta_v \gamma(G) < 0 \,\}\bigr|.

    Intuition: deleting such a vertex makes the remaining graph easier to dominate.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The number of vertices :math:`v` with :math:`\gamma(G-v) < \gamma(G)`.
        Returns 0 if :math:`G` has no vertices.

    Notes
    -----
    This is a thin wrapper around :func:`vertex_critical_number` with
    ``f = gc.domination_number`` and ``kind = 'decrease'``.

    Complexity
    ----------
    :math:`O(|V(G)|)` evaluations of ``gc.domination_number`` on vertex-deleted induced subgraphs.

    See Also
    --------
    domination_vertex_increase_number
    domination_vertex_change_number
    """
    return vertex_critical_number(G, gc.domination_number, kind="decrease")


@invariant_metadata(
    display_name="Domination vertex change number",
    notation=r"c_v(\gamma)(G)",
    category="domination criticality",
    aliases=("domination vertex sensitivity number",),
    definition=(
        "The domination vertex change number of a graph G is the number of "
        "vertices v such that deleting v changes the domination number, that is, "
        "gamma(G-v) != gamma(G)."
    ),
)
def domination_vertex_change_number(G):
    r"""
    Count vertices whose deletion **changes** the domination number.

    This returns the number of vertices :math:`v` such that

    .. math::

        \gamma(G-v) \ne \gamma(G).

    Equivalently, it counts vertices with nonzero deletion delta
    :math:`\Delta_v\gamma(G)=\gamma(G-v)-\gamma(G)\ne 0`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The number of vertices whose deletion changes :math:`\gamma(G)`.
        Returns 0 if :math:`G` has no vertices.

    Notes
    -----
    This is a thin wrapper around :func:`vertex_critical_number` with
    ``f = gc.domination_number`` and ``kind = 'change'``.

    Complexity
    ----------
    :math:`O(|V(G)|)` evaluations of ``gc.domination_number`` on vertex-deleted induced subgraphs.

    See Also
    --------
    domination_vertex_increase_number
    domination_vertex_decrease_number
    """
    return vertex_critical_number(G, gc.domination_number, kind="change")


@invariant_metadata(
    display_name="Domination vertex same number",
    notation=r"c_v^{0}(\gamma)(G)",
    category="domination criticality",
    aliases=("domination vertex unchanged number",),
    definition=(
        "The domination vertex same number of a graph G is the number of "
        "vertices v such that deleting v leaves the domination number unchanged, "
        "that is, gamma(G-v) = gamma(G)."
    ),
)
def domination_vertex_same_number(G):
    r"""
    Count vertices whose deletion leaves the domination number unchanged.

    This returns the number of vertices :math:`v` such that

    .. math::

        \gamma(G-v) = \gamma(G).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The number of vertices whose deletion does not change :math:`\gamma(G)`.
        Returns 0 if :math:`G` has no vertices.

    Notes
    -----
    This is a thin wrapper around :func:`vertex_critical_number` with
    ``f = gc.domination_number`` and ``kind = 'same'``.

    Complexity
    ----------
    :math:`O(|V(G)|)` evaluations of ``gc.domination_number`` on vertex-deleted induced subgraphs.
    """
    return vertex_critical_number(G, gc.domination_number, kind="same")


@invariant_metadata(
    display_name="Domination vertex max jump",
    notation=r"\max_{v\in V(G)} \left|\gamma(G-v)-\gamma(G)\right|",
    category="domination criticality",
    aliases=("maximum domination vertex deletion jump",),
    definition=(
        "The domination vertex max jump of a graph G is the maximum absolute "
        "change in the domination number over all single-vertex deletions."
    ),
)
def domination_vertex_max_jump(G):
    r"""
    Compute the maximum absolute change in domination number under deletion of a single vertex.

    Let :math:`\gamma(G)` denote the domination number. This function returns:

    .. math::

        \max_{v\in V(G)} \bigl|\gamma(G-v) - \gamma(G)\bigr|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The maximum absolute vertex-deletion delta for :math:`\gamma`. Returns 0 if :math:`G` has
        no vertices.

    Notes
    -----
    This is a one-vertex sensitivity measure for domination number, implemented as a thin wrapper
    around :func:`vertex_deletion_max_jump` with ``f = gc.domination_number``.

    Complexity
    ----------
    :math:`O(|V(G)|)` evaluations of ``gc.domination_number`` on vertex-deleted induced subgraphs.
    """
    return vertex_deletion_max_jump(G, gc.domination_number)


@invariant_metadata(
    display_name="Domination edge increase number",
    notation=r"c_e^{+}(\gamma)(G)",
    category="domination criticality",
    aliases=("domination increase-critical edge number",),
    definition=(
        "The domination edge increase number of a graph G is the number of "
        "edges e such that deleting e increases the domination number, that is, "
        "gamma(G-e) > gamma(G)."
    ),
)
def domination_edge_increase_number(G):
    r"""
    Count edges whose deletion **increases** the domination number.

    For each edge :math:`e \in E(G)`, define the edge-deletion delta

    .. math::

        \Delta_e \gamma(G) \;:=\; \gamma(G-e) - \gamma(G),

    where :math:`G-e` is the graph obtained by deleting :math:`e` (keeping all vertices).

    This function returns

    .. math::

        c_e^+(\gamma)(G)
        \;=\;
        \bigl|\{\, e \in E(G) : \gamma(G-e) > \gamma(G) \,\}\bigr|.

    Intuition: these are edges whose removal makes the graph harder to dominate (typically by
    reducing adjacency options).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph, intended for simple undirected graphs.

    Returns
    -------
    int
        The number of edges :math:`e` with :math:`\gamma(G-e) > \gamma(G)`. Returns 0 if :math:`G`
        has no edges.

    Notes
    -----
    This is a thin wrapper around :func:`edge_critical_number` with
    ``f = gc.domination_number`` and ``kind = 'increase'``.

    Complexity
    ----------
    :math:`O(|E(G)|)` evaluations of ``gc.domination_number`` on single-edge-deleted graphs.
    """
    return edge_critical_number(G, gc.domination_number, kind="increase")


@invariant_metadata(
    display_name="Domination edge decrease number",
    notation=r"c_e^{-}(\gamma)(G)",
    category="domination criticality",
    aliases=("domination decrease-critical edge number",),
    definition=(
        "The domination edge decrease number of a graph G is the number of "
        "edges e such that deleting e decreases the domination number, that is, "
        "gamma(G-e) < gamma(G)."
    ),
)
def domination_edge_decrease_number(G):
    r"""
    Count edges whose deletion **decreases** the domination number.

    This returns the number of edges :math:`e` such that

    .. math::

        \gamma(G-e) < \gamma(G).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph, intended for simple undirected graphs.

    Returns
    -------
    int
        The number of edges whose deletion decreases :math:`\gamma(G)`. Returns 0 if :math:`G` has
        no edges.

    Notes
    -----
    This is a thin wrapper around :func:`edge_critical_number` with
    ``f = gc.domination_number`` and ``kind = 'decrease'``.

    Complexity
    ----------
    :math:`O(|E(G)|)` evaluations of ``gc.domination_number`` on single-edge-deleted graphs.
    """
    return edge_critical_number(G, gc.domination_number, kind="decrease")


@invariant_metadata(
    display_name="Domination edge change number",
    notation=r"c_e(\gamma)(G)",
    category="domination criticality",
    aliases=("domination edge sensitivity number",),
    definition=(
        "The domination edge change number of a graph G is the number of "
        "edges e such that deleting e changes the domination number, that is, "
        "gamma(G-e) != gamma(G)."
    ),
)
def domination_edge_change_number(G):
    r"""
    Count edges whose deletion **changes** the domination number.

    This returns the number of edges :math:`e` such that

    .. math::

        \gamma(G-e) \ne \gamma(G).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph, intended for simple undirected graphs.

    Returns
    -------
    int
        The number of edges whose deletion changes :math:`\gamma(G)`. Returns 0 if :math:`G` has
        no edges.

    Notes
    -----
    This is a thin wrapper around :func:`edge_critical_number` with
    ``f = gc.domination_number`` and ``kind = 'change'``.

    Complexity
    ----------
    :math:`O(|E(G)|)` evaluations of ``gc.domination_number`` on single-edge-deleted graphs.
    """
    return edge_critical_number(G, gc.domination_number, kind="change")


@invariant_metadata(
    display_name="Domination edge same number",
    notation=r"c_e^{0}(\gamma)(G)",
    category="domination criticality",
    aliases=("domination edge unchanged number",),
    definition=(
        "The domination edge same number of a graph G is the number of "
        "edges e such that deleting e leaves the domination number unchanged, "
        "that is, gamma(G-e) = gamma(G)."
    ),
)
def domination_edge_same_number(G):
    r"""
    Count edges whose deletion leaves the domination number unchanged.

    This returns the number of edges :math:`e` such that

    .. math::

        \gamma(G-e) = \gamma(G).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph, intended for simple undirected graphs.

    Returns
    -------
    int
        The number of edges whose deletion does not change :math:`\gamma(G)`. Returns 0 if :math:`G`
        has no edges.

    Notes
    -----
    This is a thin wrapper around :func:`edge_critical_number` with
    ``f = gc.domination_number`` and ``kind = 'same'``.

    Complexity
    ----------
    :math:`O(|E(G)|)` evaluations of ``gc.domination_number`` on single-edge-deleted graphs.
    """
    return edge_critical_number(G, gc.domination_number, kind="same")


@invariant_metadata(
    display_name="Domination edge max jump",
    notation=r"\max_{e\in E(G)} \left|\gamma(G-e)-\gamma(G)\right|",
    category="domination criticality",
    aliases=("maximum domination edge deletion jump",),
    definition=(
        "The domination edge max jump of a graph G is the maximum absolute "
        "change in the domination number over all single-edge deletions."
    ),
)
def domination_edge_max_jump(G):
    r"""
    Compute the maximum absolute change in domination number under deletion of a single edge.

    This returns:

    .. math::

        \max_{e\in E(G)} \bigl|\gamma(G-e) - \gamma(G)\bigr|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph, intended for simple undirected graphs.

    Returns
    -------
    int
        The maximum absolute edge-deletion delta for :math:`\gamma`. Returns 0 if :math:`G` has
        no edges.

    Notes
    -----
    This computes edge-deletion deltas via :func:`edge_deletion_deltas` and returns the maximum
    absolute value (0 if there are no edges).

    Complexity
    ----------
    :math:`O(|E(G)|)` evaluations of ``gc.domination_number`` on single-edge-deleted graphs.
    """
    deltas = edge_deletion_deltas(G, gc.domination_number)
    return 0 if not deltas else max(abs(d) for d in deltas.values())
