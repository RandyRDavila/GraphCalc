# ============================================================
# GENERAL "CORE" FUNCTIONS (intersection of all optimum sets)
# ============================================================

from itertools import combinations
import networkx as nx
import graphcalc as gc

__all__ = [
    "core_set_minimum",
    "core_number_minimum",
    "core_set_maximum_fast",
    "core_number_maximum_fast",
    "alpha_core_set",
    "alpha_core_number",
    "clique_core_set",
    "clique_core_number",
    "domination_core_set",
    "domination_core_number",
    "zero_forcing_core_set",
    "zero_forcing_core_number",
]

def core_set_minimum(G, k_func, is_valid_set):
    r"""
    Compute the **minimum core**: the intersection of all optimal (minimum-cardinality) valid sets.

    Many graph parameters are defined as the minimum size of a vertex set satisfying some property.
    Examples include dominating sets, zero forcing sets, feedback vertex sets, etc.  This function
    takes such a property (via a membership oracle) and returns the set of vertices that appear in
    **every** minimum-size solution.

    Formal definition
    -----------------
    Let :math:`P` be a property of vertex subsets (e.g., “is a dominating set”), and suppose
    :math:`k(G)` is the minimum cardinality of a set satisfying :math:`P`:

    .. math::

        k(G) \;=\; \min\{\, |S| : S \subseteq V(G),\ P(G,S)\,\}.

    The **minimum core** (sometimes called the *core of minimum solutions*) is

    .. math::

        \operatorname{Core}_{\min}(G)
        \;=\;
        \bigcap\{\, S \subseteq V(G) : P(G,S)\ \text{and}\ |S|=k(G)\,\}.

    Intuitively, :math:`\operatorname{Core}_{\min}(G)` is the set of vertices that are
    *forced* to occur in every optimal solution.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    k_func : callable
        A function ``k_func(G) -> int`` returning the optimal minimum size :math:`k(G)`.
        Typical examples are ``gc.domination_number`` or ``gc.zero_forcing_number``.
    is_valid_set : callable
        A predicate ``is_valid_set(G, S) -> bool`` that decides the property :math:`P(G,S)`,
        where ``S`` is an iterable of vertices (e.g. a tuple from combinations).

        Important: ``is_valid_set`` should interpret ``S`` as a *vertex subset*; it should not
        depend on order or multiplicity.

    Returns
    -------
    set
        The set :math:`\operatorname{Core}_{\min}(G)`.

        Conventions:
        - If :math:`|V(G)|=0`, returns the empty set.
        - If :math:`k(G)=0`, returns the empty set (since the empty set is a minimum solution, and the intersection over the family of minimum solutions is empty).
        - If the running intersection becomes empty during enumeration, the function returns early.

        If no valid set of size :math:`k(G)` is found (which indicates an inconsistency between
        ``k_func`` and ``is_valid_set``), this implementation returns the empty set.

    Raises
    ------
    ValueError
        If ``k_func(G)`` does not return an integer, or returns a value outside
        :math:`[0, |V(G)|]`.

    Notes
    -----
    - **Exact brute force.** This routine enumerates all :math:`k`-subsets of :math:`V(G)` and tests validity. It is intended only for small graphs and/or small :math:`k`.
    - Correctness depends on consistency: ``k_func`` must return the true minimum size for the property tested by ``is_valid_set``. If they disagree, the output is not meaningful.

    Complexity
    ----------
    Let :math:`n=|V(G)|` and :math:`k=k(G)`. In the worst case, the function performs
    :math:`\binom{n}{k}` calls to ``is_valid_set``. Thus the runtime is exponential in :math:`n`
    in general, and also depends on the cost of the validity test itself.

    """
    n = gc.order(G)
    if n == 0:
        return set()

    k = k_func(G)
    if not isinstance(k, int):
        raise ValueError(f"k_func(G) must return an int, got {type(k)}")
    if k < 0 or k > n:
        raise ValueError(f"k_func(G) returned k={k}, but must satisfy 0 <= k <= n={n}")

    if k == 0:
        return set()

    nodes = list(G.nodes())
    core = None  # running intersection over all minimum solutions found

    for S in combinations(nodes, k):
        if is_valid_set(G, S):
            Sset = set(S)
            core = Sset if core is None else (core & Sset)
            if not core:
                return set()  # early exit: intersection already empty

    # If k is correct, at least one valid set of size k should exist.
    # If none were found, return empty set (signals inconsistency but stays safe).
    return core if core is not None else set()


def core_number_minimum(G, k_func, is_valid_set):
    r"""
    Compute the **minimum core number**: the size of the minimum core.

    This is the cardinality of the intersection of all minimum-cardinality valid sets:

    .. math::

        c_{\min}(G) \;=\; \bigl|\operatorname{Core}_{\min}(G)\bigr|,

    where :math:`\operatorname{Core}_{\min}(G)` is returned by :func:`core_set_minimum`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    k_func : callable
        A function ``k_func(G) -> int`` returning the optimal minimum size :math:`k(G)` for the chosen property.
    is_valid_set : callable
        A predicate ``is_valid_set(G, S) -> bool`` deciding validity of a candidate vertex set.

    Returns
    -------
    int
        The size of the minimum core, i.e. the number of vertices that appear in **every** minimum solution. Returns 0 for the empty graph, and also returns 0 when :math:`k(G)=0`.

    Notes
    -----
    This is a thin wrapper around :func:`core_set_minimum`.

    Complexity
    ----------
    Same as :func:`core_set_minimum`.
    """
    return len(core_set_minimum(G, k_func, is_valid_set))

def core_set_maximum_fast(G, f_max):
    r"""
    Compute the **maximum core** (intersection of all maximum solutions) via vertex-deletion tests.

    Many maximization parameters are defined as the optimum value attained by some vertex subset
    (e.g., maximum independent set, maximum clique). For such parameters, one can often detect
    whether a vertex is **forced** to appear in every maximum solution by checking whether deleting
    that vertex decreases the optimum.

    This function returns the set of vertices that belong to **every** maximum solution, under the
    following (crucial) assumption:

    Assumption (vertex-forcing by deletion)
    ---------------------------------------
    For the parameter value

    .. math::

        M(G) \;=\; f_{\max}(G),

    the following equivalence holds for every vertex :math:`v \in V(G)`:

    .. math::

        v \text{ belongs to every maximum solution } \iff f_{\max}(G - v) < f_{\max}(G),

    where :math:`G - v` denotes the induced subgraph obtained by deleting :math:`v`.

    Intuition:
    - If deleting :math:`v` does **not** decrease the optimum value, then there exists an optimal solution contained entirely in :math:`V(G)\setminus\{v\}`; hence :math:`v` is not forced.
    - If deleting :math:`v` **does** decrease the optimum value, then no optimal solution can avoid :math:`v`, so :math:`v` lies in the intersection of all optimal solutions.

    For example, this equivalence holds for:
    - :math:`\alpha(G)` (maximum independent set size): if :math:`\alpha(G-v) = \alpha(G)`, then there is a maximum independent set avoiding :math:`v`.
    - :math:`\omega(G)` (maximum clique size): if :math:`\omega(G-v) = \omega(G)`, then there is a maximum clique avoiding :math:`v`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    f_max : callable
        A function ``f_max(G) -> int`` returning a maximization optimum value.
        Typical examples include ``gc.independence_number`` (for :math:`\alpha`) or
        ``gc.clique_number`` (for :math:`\omega`).

        The correctness of this routine depends on ``f_max`` satisfying the
        *vertex-forcing by deletion* equivalence stated above.

    Returns
    -------
    set
        The **maximum core**:

        .. math::

            \operatorname{Core}_{\max}(G)
            \;=\;
            \bigcap\{\, S \subseteq V(G) : S \text{ is a maximum solution for } f_{\max} \,\},

        i.e., the set of vertices that appear in **every** maximum solution.

        If :math:`|V(G)| = 0`, returns the empty set.

    Notes
    -----
    - This method is exact **provided the stated equivalence holds** for your parameter. It is not valid for arbitrary maximization invariants, especially those not realized by vertex subsets in a straightforward induced-subgraph way.
    - Compared to enumerating all maximum solutions, this is typically much faster: it performs one deletion test per vertex.

    Complexity
    ----------
    The function performs :math:`|V(G)|` evaluations of ``f_max`` on induced subgraphs with one
    vertex deleted. Thus, the total time is dominated by:

    .. math::

        O\!\left(|V(G)| \cdot T_{f_{\max}}(|V(G)|-1, |E(G-v)|)\right),

    where :math:`T_{f_{\max}}` is the time needed to compute ``f_max`` on a graph.
    The additional overhead from forming induced subgraphs is typically :math:`O(|V|+|E|)` per vertex
    (depending on graph representation).
    """
    n = gc.order(G)
    if n == 0:
        return set()

    base = f_max(G)
    node_set = set(G.nodes())

    core = set()
    for v in list(node_set):
        H = G.subgraph(node_set - {v}).copy()
        if f_max(H) < base:
            core.add(v)
    return core


def core_number_maximum_fast(G, f_max):
    r"""
    Compute the **maximum core number**: the size of the maximum core.

    This is the cardinality of the set of vertices that appear in every maximum solution:

    .. math::

        c_{\max}(G) \;=\; \bigl|\operatorname{Core}_{\max}(G)\bigr|,

    where :math:`\operatorname{Core}_{\max}(G)` is returned by
    :func:`core_set_maximum_fast`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.
    f_max : callable
        A function ``f_max(G) -> int`` satisfying the deletion-test equivalence described in
        :func:`core_set_maximum_fast`.

    Returns
    -------
    int
        The size of the maximum core, i.e., the number of vertices forced to appear in **every**
        maximum solution. Returns 0 for the empty graph.

    Notes
    -----
    This is a thin wrapper around :func:`core_set_maximum_fast`.

    Complexity
    ----------
    Same as :func:`core_set_maximum_fast`.

    """
    return len(core_set_maximum_fast(G, f_max))

def alpha_core_set(G):
    r"""
    Compute the **α-core** of a graph: the intersection of all maximum independent sets.

    Let :math:`\alpha(G)` denote the independence number of :math:`G`, i.e., the maximum size of an
    independent set. The **α-core** (also called the *core with respect to maximum independent sets*)
    is the set of vertices that belong to **every** maximum independent set:

    .. math::

        \operatorname{core}_\alpha(G)
        \;=\;
        \bigcap \{\, I \subseteq V(G) : I \text{ is an independent set and } |I|=\alpha(G) \,\}.

    This implementation uses the standard deletion characterization for independence number:

    .. math::

        v \in \operatorname{core}_\alpha(G) \iff \alpha(G - v) < \alpha(G),

    where :math:`G-v` is the induced subgraph obtained by deleting :math:`v`. Equivalently, if
    :math:`\alpha(G-v)=\alpha(G)`, then there exists a maximum independent set avoiding :math:`v`,
    so :math:`v` is not in the intersection of all maximum independent sets.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    set
        The α-core of :math:`G`, i.e., the set of vertices contained in every maximum independent set.
        Returns the empty set if :math:`G` has no vertices.

    Notes
    -----
    - The deletion test used here is exact for :math:`\alpha(G)` because any maximum independent set
      of :math:`G-v` is also an independent set of :math:`G` that avoids :math:`v`.
    - This method requires only :math:`|V(G)|` evaluations of :math:`\alpha(\cdot)` on vertex-deleted
      induced subgraphs, and is typically much faster than enumerating all maximum independent sets.

    Complexity
    ----------
    Performs :math:`|V(G)|` calls to ``gc.independence_number`` on graphs with one vertex deleted.
    Total runtime therefore depends on the complexity of computing :math:`\alpha(G)` in your backend.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> # In a star, every maximum independent set consists of all leaves, so the α-core is the set of leaves.
    >>> G = nx.star_graph(4)  # 5 vertices: center + 4 leaves
    >>> gc.alpha_core_set(G) == set(range(1, 5))
    True
    """
    return core_set_maximum_fast(G, gc.independence_number)


def alpha_core_number(G):
    r"""
    Compute the **α-core number** of a graph: the size of the intersection of all maximum independent sets.

    If :math:`\operatorname{core}_\alpha(G)` denotes the α-core (the set of vertices contained in every
    maximum independent set), this function returns its cardinality:

    .. math::

        |\operatorname{core}_\alpha(G)|
        \;=\;
        \left|\bigcap \{\, I \subseteq V(G) : I \text{ is an independent set and } |I|=\alpha(G) \,\}\right|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The α-core number of :math:`G`. Returns 0 if :math:`G` has no vertices.

    Notes
    -----
    This is a thin wrapper around :func:`alpha_core_set`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> G = nx.star_graph(4)
    >>> gc.alpha_core_number(G)
    4
    """
    return len(alpha_core_set(G))

def clique_core_set(G):
    r"""
    Compute the **clique core** of a graph :math:`G`: the intersection of all maximum cliques.

    Let :math:`\omega(G)` denote the **clique number** of :math:`G`, i.e., the maximum size of a
    clique in :math:`G`. The **clique core** is the set of vertices that appear in **every**
    maximum clique:

    .. math::

        \operatorname{core}_\omega(G)
        \;=\;
        \bigcap \{\, C \subseteq V(G) : C \text{ is a clique in } G \text{ and } |C|=\omega(G)\,\}.

    Equivalently, :math:`\operatorname{core}_\omega(G)` is the set of vertices that are *forced*
    to occur in any maximum clique.

    Implementation (deletion test)
    ------------------------------
    This implementation uses the standard deletion characterization for the clique number:

    .. math::

        v \in \operatorname{core}_\omega(G) \iff \omega(G - v) < \omega(G),

    where :math:`G-v` is the induced subgraph obtained by deleting :math:`v`.

    Justification: if :math:`\omega(G-v)=\omega(G)`, then :math:`G-v` has a maximum clique of size
    :math:`\omega(G)` that avoids :math:`v`, so :math:`v` is not in the intersection. Conversely, if
    deleting :math:`v` reduces :math:`\omega`, then no maximum clique can avoid :math:`v`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. For standard clique semantics, this is intended for **simple undirected**
        graphs.

    Returns
    -------
    set
        The clique core :math:`\operatorname{core}_\omega(G)`, i.e., the set of vertices contained in
        every maximum clique. If :math:`G` has no vertices, returns the empty set.

    Notes
    -----
    - If :math:`G` has a **unique** maximum clique, then the clique core equals that clique.
    - If :math:`G` has multiple maximum cliques with little overlap, the clique core may be empty.
    - Correctness of the deletion test is specific to parameters like :math:`\omega(G)` that are
      realized by vertex subsets in induced subgraphs (a maximum clique of :math:`G-v` is also a clique
      in :math:`G` that avoids :math:`v`).

    Complexity
    ----------
    Performs :math:`|V(G)|` calls to ``gc.clique_number`` on graphs with one vertex deleted. The total
    runtime is therefore dominated by the complexity of computing :math:`\omega(G)` in your backend.

    See Also
    --------
    clique_core_number : Returns the cardinality :math:`|\operatorname{core}_\omega(G)|`.
    core_set_maximum_fast : Generic deletion-test routine used internally.
    """
    return core_set_maximum_fast(G, gc.clique_number)


def clique_core_number(G):
    r"""
    Compute the **clique core number** of a graph :math:`G`: the size of the clique core.

    If :math:`\operatorname{core}_\omega(G)` denotes the intersection of all maximum cliques of
    :math:`G`, this function returns its cardinality:

    .. math::

        |\operatorname{core}_\omega(G)|
        \;=\;
        \left|\bigcap \{\, C \subseteq V(G) : C \text{ is a clique and } |C|=\omega(G)\,\}\right|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph (intended for simple undirected graphs).

    Returns
    -------
    int
        The number of vertices that appear in every maximum clique of :math:`G`.
        Returns 0 if :math:`G` has no vertices or if the clique core is empty.

    Notes
    -----
    This quantity can be interpreted as a robustness measure for maximum cliques: it counts how many
    vertices are unavoidable in any maximum clique.

    Complexity
    ----------
    Same as :func:`clique_core_set`, since this is a thin wrapper.

    See Also
    --------
    clique_core_set : Returns the clique core set itself.
    """
    return len(clique_core_set(G))

def domination_core_set(G):
    r"""
    Compute the **domination core** of a graph :math:`G`: the intersection of all minimum dominating sets.

    Let :math:`\gamma(G)` denote the **domination number** of :math:`G`, i.e., the minimum size of a
    dominating set. The **domination core** is the set of vertices that appear in **every**
    minimum dominating set:

    .. math::

        \operatorname{core}_\gamma(G)
        \;=\;
        \bigcap \{\, S \subseteq V(G) : S \text{ is dominating in } G \text{ and } |S|=\gamma(G)\,\}.

    Equivalently, :math:`\operatorname{core}_\gamma(G)` is the set of vertices that are *forced*
    to occur in any minimum dominating set.

    A set :math:`S \subseteq V(G)` is **dominating** if every vertex of :math:`G` is in :math:`S` or
    has a neighbor in :math:`S`, i.e., :math:`N[S]=V(G)`.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. This is intended primarily for **simple undirected** graphs under the standard adjacency notion.

    Returns
    -------
    set
        The domination core :math:`\operatorname{core}_\gamma(G)`.

    Conventions:
    - If :math:`|V(G)|=0`, returns the empty set.
    - If :math:`\gamma(G)=0` (which occurs only for the empty graph under standard conventions), returns the empty set.
    - If the intersection of all minimum dominating sets is empty, returns the empty set.

    Notes
    -----
    - This function delegates to :func:`core_set_minimum` using: ``k_func = gc.domination_number`` and ``is_valid_set = gc.is_dominating_set``.
    - The implementation is **exact** but may be expensive: it enumerates all :math:`\gamma(G)`-subsets of :math:`V(G)` and tests which ones are dominating.

    Complexity
    ----------
    Let :math:`n=|V(G)|` and :math:`k=\gamma(G)`. In the worst case, the function performs
    :math:`\binom{n}{k}` dominance checks, so the runtime is exponential in :math:`n` in general
    (and depends on the cost of ``gc.is_dominating_set``).

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> # In a star, every minimum dominating set is {center}, so the domination core is {center}.
    >>> G = nx.star_graph(4)
    >>> gc.domination_core_set(G) == {0}
    True
    """
    return core_set_minimum(G, gc.domination_number, gc.is_dominating_set)


def domination_core_number(G):
    r"""
    Compute the **domination core number** of a graph :math:`G`: the size of the domination core.

    If :math:`\operatorname{core}_\gamma(G)` denotes the intersection of all minimum dominating sets
    of :math:`G`, this function returns its cardinality:

    .. math::

        |\operatorname{core}_\gamma(G)|
        \;=\;
        \left|\bigcap \{\, S \subseteq V(G) : S \text{ is dominating and } |S|=\gamma(G)\,\}\right|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The number of vertices that appear in every minimum dominating set of :math:`G`.
        Returns 0 if :math:`G` is empty or if the domination core is empty.

    Notes
    -----
    This is a thin wrapper around :func:`domination_core_set`.

    Complexity
    ----------
    Same as :func:`domination_core_set`, since this function simply takes the set cardinality.

    See Also
    --------
    domination_core_set : Returns the domination core set itself.
    """
    return len(domination_core_set(G))

def zero_forcing_core_set(G):
    r"""
    Compute the **zero forcing core** of a graph :math:`G`: the intersection of all minimum zero forcing sets.

    Let :math:`Z(G)` denote the **zero forcing number** of :math:`G`, i.e., the minimum size of a
    zero forcing set under the standard zero forcing rule. The **zero forcing core** is the set of
    vertices that appear in **every** minimum zero forcing set:

    .. math::

        \operatorname{core}_Z(G)
        \;=\;
        \bigcap \{\, S \subseteq V(G) : S \text{ is zero forcing in } G \text{ and } |S|=Z(G)\,\}.

    Equivalently, :math:`\operatorname{core}_Z(G)` is the set of vertices that are *forced* to occur
    in any minimum zero forcing set.

    Zero forcing (brief definition)
    -------------------------------
    Start with a set :math:`S` of initially colored vertices. Repeatedly apply the **color-change rule**:
    a colored vertex with **exactly one** uncolored neighbor forces that neighbor to become colored.
    A set :math:`S` is a **zero forcing set** if this process eventually colors all vertices. The
    zero forcing number :math:`Z(G)` is the minimum size of such a set.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph. This is intended primarily for **simple undirected** graphs under the standard adjacency notion.

    Returns
    -------
    set
        The zero forcing core :math:`\operatorname{core}_Z(G)`.

        Conventions:
        - If :math:`|V(G)|=0`, returns the empty set.
        - If :math:`Z(G)=0` (which occurs only for the empty graph under standard conventions), returns the empty set.
        - If the intersection of all minimum zero forcing sets is empty, returns the empty set.

    Notes
    -----
    - This function delegates to :func:`core_set_minimum` using:
        * ``k_func = gc.zero_forcing_number`` and
        * ``is_valid_set = gc.is_zero_forcing_set``.
    - The implementation is **exact** but may be expensive: it enumerates all :math:`Z(G)`-subsets of :math:`V(G)` and tests which ones are zero forcing.

    Complexity
    ----------
    Let :math:`n=|V(G)|` and :math:`k=Z(G)`. In the worst case, the function performs
    :math:`\binom{n}{k}` zero-forcing checks, so the runtime is exponential in :math:`n` in general
    (and depends on the cost of ``gc.is_zero_forcing_set``).

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc as gc
    >>> # In a path, there are minimum zero forcing sets of size 1 (either endpoint),
    >>> # so the intersection is empty.
    >>> G = nx.path_graph(6)
    >>> gc.zero_forcing_core_set(G)
    set()
    """
    return core_set_minimum(G, gc.zero_forcing_number, gc.is_zero_forcing_set)


def zero_forcing_core_number(G):
    r"""
    Compute the **zero forcing core number** of a graph :math:`G`: the size of the zero forcing core.

    If :math:`\operatorname{core}_Z(G)` denotes the intersection of all minimum zero forcing sets of
    :math:`G`, this function returns its cardinality:

    .. math::

        |\operatorname{core}_Z(G)|
        \;=\;
        \left|\bigcap \{\, S \subseteq V(G) : S \text{ is zero forcing and } |S|=Z(G)\,\}\right|.

    Parameters
    ----------
    G : networkx.Graph-like
        A finite graph.

    Returns
    -------
    int
        The number of vertices that appear in every minimum zero forcing set of :math:`G`.
        Returns 0 if :math:`G` is empty or if the zero forcing core is empty.

    Notes
    -----
    This is a thin wrapper around :func:`zero_forcing_core_set`.

    Complexity
    ----------
    Same as :func:`zero_forcing_core_set`.

    See Also
    --------
    zero_forcing_core_set : Returns the zero forcing core set itself.
    """
    return len(zero_forcing_core_set(G))
