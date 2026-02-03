import networkx as nx
import graphcalc as gc

__all__ = [
    "local_parameter",
    "local_parameter_radius",
    "local_independence_number",
    "local_clique_number",
    "local_domination_number",
    "local_chromatic_number",
    "local_zero_forcing_number",
    "local_residue",
    "local_harmonic_index",
]

# ============================================================
# GENERAL LOCAL OPERATORS
# ============================================================

def local_parameter(G, f, *, neighborhood="open", agg="max"):
    r"""
    Apply a graph parameter locally to neighborhood-induced subgraphs and aggregate the results.

    For each vertex :math:`v \in V(G)`, this operator forms the induced subgraph on either the
    **open neighborhood** :math:`N(v)` or the **closed neighborhood** :math:`N[v]=N(v)\cup\{v\}`,
    evaluates a graph parameter :math:`f` on that induced subgraph, and then aggregates the local
    values over all vertices.

    Formally, let

    .. math::

        S_v \;=\;
        \begin{cases}
        N(v), & \text{if neighborhood = 'open'},\\
        N[v], & \text{if neighborhood = 'closed'}.
        \end{cases}

    This function computes:

    .. math::

        \operatorname{Agg}_{v\in V(G)}\; f\!\left(G[S_v]\right),

    where :math:`G[S_v]` is the induced subgraph on :math:`S_v`, and :math:`\operatorname{Agg}`
    is one of ``max``, ``min``, ``sum``, or the arithmetic mean ``avg``.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are defined using ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** under the standard adjacency notion. If ``G`` is a
        directed graph, NetworkX defines ``neighbors`` as successors, which generally yields a
        different “out-neighborhood” notion.
    f : callable
        A function that accepts a graph ``H`` (a NetworkX graph) and returns a numeric value.
        Typical examples include invariant/parameter functions such as ``order``, ``size``,
        independence number, clique number, domination number, zero forcing number, etc.

        This implementation calls ``f`` on a **copy** of each induced subgraph to guard against
        accidental mutation inside ``f``.
    neighborhood : {'open', 'closed'}, optional
        Which neighborhood to use for the local induced subgraphs:
        - ``'open'``: :math:`S_v = N(v)` (neighbors only)
        - ``'closed'``: :math:`S_v = N[v]` (neighbors plus the vertex itself)
    agg : {'max', 'min', 'sum', 'avg'}, optional
        Aggregation operator applied to the multiset of local values
        :math:`\{ f(G[S_v]) : v\in V(G)\}`.

    Returns
    -------
    number
        The aggregated value. If :math:`|V(G)| = 0`, returns 0.

    Notes
    -----
    - If :math:`v` is isolated, then :math:`N(v)=\varnothing`. In that case:
        * with ``neighborhood='open'`` the induced graph is empty, ``G[∅]``;
        * with ``neighborhood='closed'`` the induced graph is the single-vertex graph, ``G[{v}]``.
      Ensure your parameter function ``f`` is defined on these graphs.
    - The induced subgraphs are formed independently for each vertex; overlapping neighborhoods
      are allowed and expected.
    - If ``f`` is guaranteed to be pure/read-only, you may remove the ``.copy()`` calls for speed.

    Complexity
    ----------
    This constructs one induced subgraph per vertex and evaluates ``f`` on each. The total runtime
    is dominated by:
      - the cost of forming induced subgraphs on :math:`N(v)` or :math:`N[v]`, and
      - the cost of evaluating ``f`` on each such subgraph.
    In symbols, the cost is roughly :math:`\sum_{v\in V(G)} T_f(|S_v|, |E(G[S_v])|)` plus overhead.

    Raises
    ------
    ValueError
        If ``neighborhood`` is not in ``{'open','closed'}`` or ``agg`` is not in
        ``{'max','min','sum','avg'}``.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(5)
    >>> # Max degree inside open neighborhoods: each open neighborhood is an induced subgraph
    >>> gc.local_parameter(G, gc.maximum_degree, neighborhood="open", agg="max")
    0
    >>> # Max order of closed neighborhoods: max |N[v]| over v
    >>> local_parameter(G, gc.order, neighborhood="closed", agg="max")
    3
    """
    n = gc.order(G)
    if n == 0:
        return 0

    if neighborhood not in {"open", "closed"}:
        raise ValueError("neighborhood must be 'open' or 'closed'")
    if agg not in {"max", "min", "sum", "avg"}:
        raise ValueError("agg must be one of {'max','min','sum','avg'}")

    vals = []
    for v in G.nodes():
        S = set(G.neighbors(v))
        if neighborhood == "closed":
            S.add(v)

        H = G.subgraph(S).copy()
        vals.append(f(H))

    if agg == "max":
        return max(vals)
    if agg == "min":
        return min(vals)
    if agg == "sum":
        return sum(vals)
    return sum(vals) / len(vals)  # agg == "avg"


def local_parameter_radius(G, f, *, r=1, closed=True, agg="max"):
    r"""
    Apply a graph parameter locally to **radius-:math:`r` distance balls** and aggregate.

    For each vertex :math:`v \in V(G)`, form the vertex set of the (closed) distance ball

    .. math::

        B_r(v) \;=\; \{\, u \in V(G) : d_G(u,v) \le r \,\},

    where :math:`d_G` is the unweighted shortest-path distance in :math:`G`. The function then
    evaluates a graph parameter :math:`f` on the induced subgraph :math:`G[B_r(v)]` (or on the
    induced **open ball** if ``closed=False``), and aggregates the resulting values over all vertices.

    If ``closed=False``, the center vertex :math:`v` is removed from the ball before inducing:
    :math:`B_r(v) \setminus \{v\}`.

    Common special cases
    --------------------
    - ``r=1, closed=False``  gives induced open neighborhoods :math:`G[N(v)]`.
    - ``r=1, closed=True``   gives induced closed neighborhoods :math:`G[N[v]]`.
    - ``r=2, closed=True``   gives induced closed distance-2 balls.
    - ``r=2, closed=False``  gives induced open distance-2 balls (distance ≤ 2, excluding the center).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Distances are computed with unweighted BFS via
        :func:`networkx.single_source_shortest_path_length`. This is primarily intended for
        **undirected graphs**; for directed graphs, distances respect edge directions.
    f : callable
        A function accepting a graph ``H`` and returning a numeric value.
        This implementation calls ``f`` on a **copy** of each induced ball subgraph.
    r : int, optional
        Radius :math:`r \ge 0`. If ``r=0``, each ball is ``{v}`` (or empty if ``closed=False``).
    closed : bool, optional
        If True, include the center vertex :math:`v` in the ball; if False, exclude it.
    agg : {'max', 'min', 'sum', 'avg'}, optional
        Aggregation operator applied to the multiset of local values
        :math:`\{ f(G[B_r(v)]) : v\in V(G)\}` (or open balls when ``closed=False``).

    Returns
    -------
    number
        The aggregated value. If :math:`|V(G)| = 0`, returns 0.

    Raises
    ------
    ValueError
        If ``r < 0`` or if ``agg`` is not in ``{'max','min','sum','avg'}``.

    Notes
    -----
    - The ball is computed with BFS distances in the *original graph* ``G`` and then induced.
      This is different from, e.g., taking the radius-:math:`r` neighborhood inside some already
      induced subgraph.
    - Ensure your parameter ``f`` is defined on the empty graph, since open balls can be empty
      when ``r=0`` or when a vertex is isolated and you exclude the center.

    Complexity
    ----------
    For each vertex, a BFS to depth ``r`` is performed. In the worst case (when ``r`` is large
    relative to the diameter), this is essentially one BFS per vertex, so the time is
    :math:`O(|V||E|)` for sparse graphs (more precisely :math:`O(\sum_v (|V|+|E|))` in the worst case).
    The dominant cost is usually the BFS calls plus evaluation of ``f`` on each induced ball.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(6)
    >>> # Max size of closed radius-2 balls in a path on 6 vertices:
    >>> gc.local_parameter_radius(G, gc.order, r=2, closed=True, agg="max")
    5
    >>> # Average size of open radius-1 balls is the average degree:
    >>> gc.local_parameter_radius(G, gc.order, r=1, closed=False, agg="avg")
    1.6666666666666667
    """
    if r < 0:
        raise ValueError("r must be >= 0")
    n = gc.order(G)
    if n == 0:
        return 0

    vals = []
    for v in G.nodes():
        dist = nx.single_source_shortest_path_length(G, v, cutoff=r)
        S = set(dist.keys())
        if not closed:
            S.discard(v)
        H = G.subgraph(S).copy()
        vals.append(f(H))

    if agg == "max":
        return max(vals)
    if agg == "min":
        return min(vals)
    if agg == "sum":
        return sum(vals)
    if agg == "avg":
        return sum(vals) / len(vals)

    raise ValueError("agg must be one of {'max','min','sum','avg'}")

def local_independence_number(G):
    r"""
    Compute the **local independence number** of a graph :math:`G` (with respect to open neighborhoods).

    The local independence number is defined as the maximum independence number attained by the
    induced subgraph on an open neighborhood:

    .. math::

        \alpha_{\mathrm{loc}}(G) \;=\; \max_{v \in V(G)} \alpha\!\bigl(G[N(v)]\bigr),

    where :math:`N(v)` is the **open neighborhood** of :math:`v` (the set of neighbors of :math:`v`),
    :math:`G[N(v)]` is the induced subgraph on :math:`N(v)`, and :math:`\alpha(H)` denotes the
    independence number of a graph :math:`H`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are taken using ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** with the standard adjacency notion. For directed graphs,
        NetworkX interprets ``neighbors`` as successors, which yields an out-neighborhood variant.

    Returns
    -------
    int
        The value :math:`\alpha_{\mathrm{loc}}(G)`. If :math:`G` has no vertices, returns 0.

    Notes
    -----
    - If :math:`v` is isolated, then :math:`N(v)=\varnothing` and :math:`G[N(v)]` is the empty graph,
      so :math:`\alpha(G[N(v)]) = 0`. Thus isolated vertices do not increase the maximum.
    - This is a neighborhood-based refinement of the global independence number. In general,
      :math:`\alpha_{\mathrm{loc}}(G)` need not equal :math:`\alpha(G)` and may be smaller or larger
      (e.g., in graphs with a high-degree vertex whose neighborhood is sparse).
    - For a complete graph :math:`K_n` (n ≥ 2), every open neighborhood induces :math:`K_{n-1}`,
      so :math:`\alpha_{\mathrm{loc}}(K_n)=1`.

    Complexity
    ----------
    This function delegates to :func:`local_parameter` with ``gc.independence_number``.
    The runtime is therefore dominated by the cost of computing the independence number on each
    neighborhood-induced subgraph. For exact independence number routines, this may be exponential
    in the neighborhood sizes.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> # In a star, the center's neighborhood is an independent set of size n-1.
    >>> G = nx.star_graph(5)  # 6 vertices total: center + 5 leaves
    >>> gc.local_independence_number(G)
    5

    >>> # In a complete graph, every neighborhood is a clique.
    >>> H = nx.complete_graph(6)
    >>> gc.local_independence_number(H)
    1
    """
    return local_parameter(G, gc.independence_number, neighborhood="open", agg="max")

def local_clique_number(G):
    r"""
    Compute the **local clique number** of a graph :math:`G` (with respect to open neighborhoods).

    The local clique number is defined as the maximum clique number attained by an induced
    open neighborhood:

    .. math::

        \omega_{\mathrm{loc}}(G) \;=\; \max_{v \in V(G)} \omega\!\bigl(G[N(v)]\bigr),

    where :math:`N(v)` is the **open neighborhood** of :math:`v` (the set of neighbors of :math:`v`),
    :math:`G[N(v)]` is the subgraph induced by :math:`N(v)`, and :math:`\omega(H)` denotes the
    **clique number** of :math:`H` (the size of a largest complete subgraph of :math:`H`).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are computed via ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** under the standard adjacency notion. For directed graphs,
        NetworkX interprets ``neighbors`` as successors, which yields an out-neighborhood variant.

    Returns
    -------
    int
        The value :math:`\omega_{\mathrm{loc}}(G)`. If :math:`G` has no vertices, returns 0.

    Notes
    -----
    - If :math:`v` is isolated, then :math:`N(v)=\varnothing` and :math:`G[N(v)]` is empty, so
      :math:`\omega(G[N(v)]) = 0`. Thus isolated vertices do not increase the maximum.
    - :math:`\omega(G[N(v)])` measures how “clique-like” the neighborhood of :math:`v` is. In
      particular, it equals the maximum number of neighbors of :math:`v` that are pairwise adjacent.
    - For a complete graph :math:`K_n` (n ≥ 2), every open neighborhood induces :math:`K_{n-1}`, so
      :math:`\omega_{\mathrm{loc}}(K_n)=n-1`.
    - For a bipartite graph, every open neighborhood induces an independent set, so
      :math:`\omega_{\mathrm{loc}}(G) \le 1` (and equals 1 whenever there is an edge).

    Complexity
    ----------
    This function delegates to :func:`local_parameter` with ``gc.clique_number``.
    The runtime is therefore dominated by the cost of computing the clique number on each
    neighborhood-induced subgraph. Exact clique number routines are typically exponential in
    neighborhood size.

    Examples
    --------
    >>> import networkx as nx
    >>> # In a star, each leaf has neighborhood {center} (clique number 1),
    >>> # and the center has neighborhood consisting of independent leaves (clique number 1).
    >>> G = nx.star_graph(5)
    >>> gc.local_clique_number(G)
    1

    >>> # In a complete graph, neighborhoods are complete.
    >>> H = nx.complete_graph(6)
    >>> gc.local_clique_number(H)
    5

    >>> # In a triangle, each vertex's neighborhood is an edge (clique number 2).
    >>> T = nx.complete_graph(3)
    >>> gc.local_clique_number(T)
    2
    """
    return local_parameter(G, gc.clique_number, neighborhood="open", agg="max")

def local_domination_number(G):
    r"""
    Compute the **local domination number** of a graph :math:`G` (with respect to open neighborhoods).

    The local domination number is defined as the maximum domination number attained by an induced
    open neighborhood:

    .. math::

        \gamma_{\mathrm{loc}}(G) \;=\; \max_{v \in V(G)} \gamma\!\bigl(G[N(v)]\bigr),

    where :math:`N(v)` is the **open neighborhood** of :math:`v` (the set of neighbors of :math:`v`),
    :math:`G[N(v)]` is the subgraph induced by :math:`N(v)`, and :math:`\gamma(H)` denotes the
    **domination number** of a graph :math:`H`.

    Recall that the domination number :math:`\gamma(H)` is

    .. math::

        \gamma(H) \;=\; \min\{\, |D| : D \subseteq V(H)\ \text{and}\ N_H[D] = V(H) \,\},

    i.e., the minimum size of a vertex set :math:`D` such that every vertex of :math:`H` lies in
    :math:`D` or has a neighbor in :math:`D` (here :math:`N_H[D]` denotes the closed neighborhood of
    :math:`D` in :math:`H`).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are computed via ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** under the standard adjacency notion. For directed graphs,
        NetworkX interprets ``neighbors`` as successors, which yields an out-neighborhood variant.

    Returns
    -------
    int
        The value :math:`\gamma_{\mathrm{loc}}(G)`. If :math:`G` has no vertices, returns 0.

    Notes
    -----
    - If :math:`v` is isolated, then :math:`N(v)=\varnothing` and :math:`G[N(v)]` is the empty graph.
      Under the standard convention used by most domination-number implementations, the empty graph
      has domination number 0, so isolated vertices do not increase the maximum.
    - This is a **local** parameter: it measures how “difficult” it is to dominate each open-neighborhood
      subgraph, rather than dominating :math:`G` itself. In general, :math:`\gamma_{\mathrm{loc}}(G)`
      is not equal to :math:`\gamma(G)`.

    Complexity
    ----------
    This function delegates to :func:`local_parameter` with ``gc.domination_number``.
    The runtime is therefore dominated by the cost of computing the domination number on each
    neighborhood-induced subgraph. Exact domination number routines are typically exponential in
    neighborhood size.

    Examples
    --------
    >>> import networkx as nx
    >>> # In a star, the center's neighborhood is an independent set of size n-1,
    >>> # whose domination number equals n-1 (each isolated vertex must be chosen).
    >>> G = nx.star_graph(5)
    >>> gc.local_domination_number(G)
    5

    >>> # In a complete graph, each neighborhood is complete, so domination number is 1.
    >>> H = nx.complete_graph(6)
    >>> gc.local_domination_number(H)
    1
    """
    return local_parameter(G, gc.domination_number, neighborhood="open", agg="max")

def local_zero_forcing_number(G):
    r"""
    Compute the **local zero forcing number** of a graph :math:`G` (with respect to open neighborhoods).

    The local zero forcing number is defined as the maximum zero forcing number attained by an induced
    open neighborhood:

    .. math::

        Z_{\mathrm{loc}}(G) \;=\; \max_{v \in V(G)} Z\!\bigl(G[N(v)]\bigr),

    where :math:`N(v)` is the **open neighborhood** of :math:`v`, :math:`G[N(v)]` is the subgraph
    induced by :math:`N(v)`, and :math:`Z(H)` denotes the **zero forcing number** of a graph :math:`H`.

    Zero forcing number (brief definition)
    --------------------------------------
    Given a graph :math:`H` and an initial set :math:`S \subseteq V(H)` of colored vertices, apply the
    *standard zero forcing rule* repeatedly:

    - If a colored vertex has **exactly one** uncolored neighbor, it *forces* that neighbor to become colored.

    The set :math:`S` is a **zero forcing set** if this process eventually colors all vertices of :math:`H`.
    The zero forcing number is the minimum size of such a set:

    .. math::

        Z(H) \;=\; \min\{\, |S| : S \subseteq V(H)\ \text{is a zero forcing set for } H \,\}.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are computed via ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** under the standard adjacency notion. For directed graphs,
        NetworkX interprets ``neighbors`` as successors, which yields an out-neighborhood variant.

    Returns
    -------
    int
        The value :math:`Z_{\mathrm{loc}}(G)`. If :math:`G` has no vertices, returns 0.

    Notes
    -----
    - If :math:`v` is isolated, then :math:`N(v)=\varnothing` and :math:`G[N(v)]` is the empty graph.
      Under the standard convention used by most zero-forcing implementations, the empty graph has
      zero forcing number 0, so isolated vertices do not increase the maximum.
    - This is a **local** refinement: it measures how large a forcing set is needed to control each
      neighborhood-induced subgraph, rather than :math:`G` itself. In general,
      :math:`Z_{\mathrm{loc}}(G)` is not equal to :math:`Z(G)`.

    Complexity
    ----------
    This function delegates to :func:`local_parameter` with ``gc.zero_forcing_number``.
    The runtime is therefore dominated by the cost of computing the zero forcing number on each
    neighborhood-induced subgraph. Exact zero forcing number routines are typically exponential in
    neighborhood size.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> # In a complete graph, each neighborhood is complete, and Z(K_m)=m-1.
    >>> K6 = nx.complete_graph(6)
    >>> gc.local_zero_forcing_number(K6)  # neighborhoods are K5
    4

    >>> # In a star, the center's neighborhood is an independent set of size n-1, whose Z equals n-1.
    >>> S = nx.star_graph(5)  # 6 vertices total: center + 5 leaves
    >>> gc.local_zero_forcing_number(S)
    5
    """
    return local_parameter(G, gc.zero_forcing_number, neighborhood="open", agg="max")

def local_residue(G):
    r"""
    Compute the **local Havel–Hakimi residue** of a graph :math:`G` (with respect to open neighborhoods).

    This parameter is defined by applying the **Havel–Hakimi residue** (as implemented by
    :func:`graphcalc.residue`) to each induced open neighborhood subgraph and taking the maximum:

    .. math::

        \operatorname{res}_{\mathrm{loc}}(G)
        \;=\;
        \max_{v \in V(G)} \operatorname{res}\!\bigl(G[N(v)]\bigr),

    where :math:`N(v)` is the **open neighborhood** of :math:`v`, :math:`G[N(v)]` is the subgraph
    induced by :math:`N(v)`, and :math:`\operatorname{res}(H)` denotes the **Havel–Hakimi residue**
    of :math:`H`.

    Havel–Hakimi residue (brief description)
    ----------------------------------------
    The Havel–Hakimi process operates on a (nonnegative) integer degree sequence
    :math:`d=(d_1,\dots,d_n)`:

    1. Sort the sequence in nonincreasing order.
    2. Remove the first term :math:`d_1`.
    3. Subtract 1 from each of the next :math:`d_1` terms.
    4. Repeat until no positive terms remain (or the process fails for non-graphical sequences).

    When applied to the **degree sequence of a graph** :math:`H`, this iterative reduction
    produces a terminal sequence with some number of zeros. The **residue** :math:`\operatorname{res}(H)`
    (in the Havel–Hakimi sense) is the number of zeros in this terminal sequence. Equivalently, it is
    the number of vertices left with degree 0 at termination of the Havel–Hakimi reduction started from
    the degree sequence of :math:`H`.

    (This matches the convention used by :func:`graphcalc.residue`.)

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are computed via ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** under the standard adjacency notion. For directed graphs,
        NetworkX interprets ``neighbors`` as successors; if you want an undirected notion, convert via
        ``G.to_undirected()`` first.

    Returns
    -------
    int
        The value :math:`\operatorname{res}_{\mathrm{loc}}(G)`. If :math:`G` has no vertices,
        returns 0.

    Notes
    -----
    - If :math:`v` is isolated, then :math:`N(v)=\varnothing` and :math:`G[N(v)]` is the empty graph.
      The Havel–Hakimi residue of the empty graph is 0 under the usual convention, so isolated vertices
      do not increase the maximum.
    - This is a **local** construction: it measures the Havel–Hakimi residue on neighborhood-induced
      subgraphs rather than on :math:`G` itself.
    - The Havel–Hakimi residue is defined via degree sequences and depends only on the degree sequence
      of the input graph, not on its particular labeling.

    Complexity
    ----------
    This function delegates to :func:`local_parameter` with ``gc.residue``. The runtime is dominated
    by the cost of evaluating ``gc.residue`` on each neighborhood-induced subgraph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> # For a complete graph K_n, each open neighborhood is K_{n-1} (regular),
    >>> # so the local residue is the HH-residue of K_{n-1}.
    >>> G = nx.complete_graph(6)
    >>> gc.local_residue(G)
    1
    """
    return local_parameter(G, gc.residue, neighborhood="open", agg="max")

def local_harmonic_index(G):
    r"""
    Compute the **local harmonic index** of a graph :math:`G` (with respect to open neighborhoods).

    This parameter is defined by applying the harmonic index to each induced open neighborhood
    subgraph and taking the maximum:

    .. math::

        H_{\mathrm{loc}}(G) \;=\; \max_{v \in V(G)} H\!\bigl(G[N(v)]\bigr),

    where :math:`N(v)` is the **open neighborhood** of :math:`v`, :math:`G[N(v)]` is the subgraph
    induced by :math:`N(v)`, and :math:`H(\cdot)` denotes the **harmonic index** as implemented by
    :func:`graphcalc.harmonic_index` (here referenced as ``gc.harmonic_index``).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are computed via ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** under the standard adjacency notion. For directed graphs,
        NetworkX interprets ``neighbors`` as successors, which yields an out-neighborhood variant.

    Returns
    -------
    number
        The value :math:`H_{\mathrm{loc}}(G)`. If :math:`G` has no vertices, returns 0.

    Notes
    -----
    - A widely used definition of the (edge-based) harmonic index of a simple undirected graph :math:`H` is

      .. math::

          H(H) \;=\; \sum_{xy \in E(H)} \frac{2}{\deg_H(x) + \deg_H(y)}.

      This function uses the **exact convention implemented by** ``gc.harmonic_index``; the above
      formula is included for orientation.
    - If :math:`v` is isolated, then :math:`N(v)=\varnothing` and :math:`G[N(v)]` is the empty graph.
      Under the standard edge-sum convention, the harmonic index of an edgeless graph is 0 (empty sum),
      so isolated vertices do not increase the maximum.
    - This is a **local** refinement: it measures harmonic index on neighborhood-induced subgraphs rather
      than on :math:`G` itself.

    Complexity
    ----------
    This function delegates to :func:`local_parameter` with ``gc.harmonic_index``. The runtime is
    dominated by the cost of evaluating ``gc.harmonic_index`` on each neighborhood-induced subgraph.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.path_graph(6)
    >>> local_harmonic_index(G)  # doctest: +SKIP
    ...  # depends on gc.harmonic_index implementation details
    """
    return local_parameter(G, gc.harmonic_index, neighborhood="open", agg="max")

def local_chromatic_number(G):
    r"""
    Compute the **local chromatic number** of a graph :math:`G` (with respect to open neighborhoods).

    This parameter is defined by taking, over all vertices :math:`v`, the chromatic number of the
    subgraph induced by the open neighborhood :math:`N(v)` and then maximizing:

    .. math::

        \chi_{\mathrm{loc}}(G) \;=\; \max_{v \in V(G)} \chi\!\bigl(G[N(v)]\bigr),

    where :math:`N(v)` is the **open neighborhood** of :math:`v`, :math:`G[N(v)]` denotes the induced
    subgraph on :math:`N(v)`, and :math:`\chi(H)` is the **chromatic number** of :math:`H` (the minimum
    number of colors in a proper vertex coloring of :math:`H`).

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. Neighborhoods are computed via ``G.neighbors(v)``, so this is primarily
        intended for **undirected graphs** under the standard adjacency notion. For directed graphs,
        NetworkX interprets ``neighbors`` as successors, which yields an out-neighborhood variant.

    Returns
    -------
    int
        The value :math:`\chi_{\mathrm{loc}}(G)`. If :math:`G` has no vertices, returns 0.

    Notes
    -----
    - Empty/isolated-neighborhood convention: if :math:`v` is isolated, then :math:`N(v)=\varnothing`
      and :math:`G[N(v)]` is the empty graph. Many conventions set :math:`\chi(\varnothing)=0`, while
      some implementations return 1. This function uses whatever convention is implemented by
      ``gc.chromatic_number`` when applied to the empty graph. This only affects the value when
      :math:`G` has isolated vertices.
    - This is a **local** refinement of coloring complexity: it measures the strongest coloring
      requirement among neighborhood-induced subgraphs, rather than coloring :math:`G` itself.
      In general, :math:`\chi_{\mathrm{loc}}(G)` need not equal :math:`\chi(G)`.

    Complexity
    ----------
    This function delegates to :func:`local_parameter` with ``gc.chromatic_number``.
    The runtime is therefore dominated by the cost of computing the chromatic number on each
    neighborhood-induced subgraph. Exact chromatic number routines are typically exponential in
    neighborhood size.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> # Complete graph: each neighborhood is complete, so chi_loc(K_n) = n-1.
    >>> K6 = nx.complete_graph(6)
    >>> gc.local_chromatic_number(K6)
    5

    >>> # Bipartite graphs have neighborhoods inducing independent sets, so chi_loc is at most 1
    >>> # (or 0 on isolated neighborhoods depending on the empty-graph convention).
    >>> P6 = nx.path_graph(6)
    >>> gc.local_chromatic_number(P6)
    1
    """
    return local_parameter(G, gc.chromatic_number, neighborhood="open", agg="max")
