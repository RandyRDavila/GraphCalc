# graphcalc/graphs/invariants/transversal_invariants.py

import itertools
import networkx as nx
import graphcalc.graphs as gc
from graphcalc.metadata import invariant_metadata

__all__ = [
    "maximal_clique_transversal_number",
    "maximal_independent_set_transversal_number",
    "minimal_dominating_set_transversal_number",
]

@invariant_metadata(
    display_name="Maximal-clique transversal number",
    notation=r"\tau_{\max\mathrm{clique}}(G)",
    category="transversal invariants",
    aliases=("maximal clique hitting number",),
    definition=(
        "The maximal-clique transversal number of a graph G is the minimum "
        "cardinality of a vertex set that intersects every maximal clique of G."
    ),
)
def maximal_clique_transversal_number(G):
    r"""
    Compute the maximal-clique transversal number of :math:`G`.

    A **maximal-clique transversal** is a set :math:`S \subseteq V(G)` that intersects
    every *maximal* clique of :math:`G`. Equivalently, if :math:`\mathcal{C}` is the
    family of maximal cliques of :math:`G`, then :math:`S` is feasible iff
    :math:`S \cap C \neq \varnothing` for all :math:`C \in \mathcal{C}`.

    This function returns the minimum possible size:
    :math:`\tau_{\max\mathrm{clique}}(G) = \min\{ |S| : S \cap C \neq \varnothing \;\; \forall C \in \mathcal{C}\}`.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite simple undirected graphs).

    Notes
    -----
    This is an exact brute-force hitting-set computation:

    1. Enumerate maximal cliques using :func:`networkx.find_cliques`.
    2. Search subsets :math:`S \subseteq V(G)` in increasing cardinality until one intersects every maximal clique.

    Conventions:
    - If :math:`|V(G)| = 0`, returns ``0``.
    - For an edgeless graph on :math:`n` vertices, returns :math:`n` (each vertex is a maximal clique).

    Complexity can be exponential in :math:`n` (both the number of maximal cliques and
    the subset search). Intended only for small graphs.

    Returns
    -------
    int
        The minimum size of a vertex set that meets every maximal clique of :math:`G`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.complete_graph(4)
    >>> gc.maximal_clique_transversal_number(G)
    1
    >>> H = nx.empty_graph(5)
    >>> gc.maximal_clique_transversal_number(H)
    5
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0

    maximal_cliques = [set(C) for C in nx.find_cliques(G)]
    # For simple graphs with n>0, this list is nonempty (at least singleton cliques).
    if not maximal_cliques:
        return 0

    nodes = list(G.nodes())
    for k in range(0, n + 1):
        for S in itertools.combinations(nodes, k):
            Sset = set(S)
            if all(Sset & C for C in maximal_cliques):
                return k

    # Fallback: always possible with S = V(G)
    return n


@invariant_metadata(
    display_name="Maximal-independent-set transversal number",
    notation=r"\tau_{\max\mathrm{ind}}(G)",
    category="transversal invariants",
    aliases=("maximal independent set hitting number",),
    definition=(
        "The maximal-independent-set transversal number of a graph G is the minimum "
        "cardinality of a vertex set that intersects every maximal independent set of G."
    ),
)
def maximal_independent_set_transversal_number(G):
    r"""
    Compute the maximal-independent-set transversal number of :math:`G`.

    A set :math:`S \subseteq V(G)` is a transversal of maximal independent sets if it
    intersects every *maximal* independent set :math:`I` of :math:`G`, i.e.,
    :math:`S \cap I \neq \varnothing` for all maximal independent sets :math:`I`.

    This function returns the minimum possible size of such a set.

    Notes
    -----
    Maximal independent sets of :math:`G` are exactly maximal cliques of the complement
    graph :math:`\overline{G}`. Therefore,
    :math:`\tau_{\max\mathrm{ind}}(G) = \tau_{\max\mathrm{clique}}(\overline{G})`,
    and this function delegates to
    ``maximal_clique_transversal_number(nx.complement(G))``.

    Conventions:
    - If :math:`|V(G)| = 0`, returns ``0``.
    - For a complete graph :math:`K_n`, every maximal independent set is a singleton, so the answer is :math:`n`.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite simple undirected graphs).

    Returns
    -------
    int
        The minimum size of a vertex set intersecting every maximal independent set of :math:`G`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.complete_graph(4)
    >>> gc.maximal_independent_set_transversal_number(G)
    4
    >>> H = nx.empty_graph(5)
    >>> gc.maximal_independent_set_transversal_number(H)
    1
    """
    if G.number_of_nodes() == 0:
        return 0
    H = nx.complement(G)
    return maximal_clique_transversal_number(H)

def _is_minimal_dominating_set(G, S):
    r"""
    Test whether :math:`S` is an inclusion-minimal dominating set of :math:`G`.

    A set :math:`S \subseteq V(G)` is **inclusion-minimal dominating** if:

    - :math:`S` is a dominating set, and
    - no proper subset of :math:`S` is a dominating set.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    S : iterable
        Candidate vertex set.

    Returns
    -------
    bool
        ``True`` iff :math:`S` is an inclusion-minimal dominating set of :math:`G`.
    """
    S = set(S)
    if not gc.is_dominating_set(G, S):
        return False
    # Check minimality by removing one vertex at a time
    for u in list(S):
        if gc.is_dominating_set(G, S - {u}):
            return False
    return True


def _all_minimal_dominating_sets(G):
    r"""
    Enumerate all inclusion-minimal dominating sets of :math:`G` (brute force).

    Notes
    -----
    This routine checks all subsets of :math:`V(G)` and retains those that are
    dominating and inclusion-minimal. It is exponential in :math:`|V(G)|` and is
    intended only for very small graphs.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    list of set
        A list of all inclusion-minimal dominating sets of :math:`G`.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    mins = []

    # Enumerate all subsets; collect those that are inclusion-minimal dominating
    for k in range(0, n + 1):
        for S in itertools.combinations(nodes, k):
            if _is_minimal_dominating_set(G, S):
                mins.append(set(S))
    return mins


@invariant_metadata(
    display_name="Minimal-dominating-set transversal number",
    notation=r"\tau_{\min\mathrm{dom}}(G)",
    category="transversal invariants",
    aliases=("minimal dominating set hitting number",),
    definition=(
        "The minimal-dominating-set transversal number of a graph G is the minimum "
        "cardinality of a vertex set that intersects every inclusion-minimal dominating set of G."
    ),
)
def minimal_dominating_set_transversal_number(G, max_n=20):
    r"""
    Compute the minimal-dominating-set transversal number of :math:`G`.

    Let :math:`\mathcal{D}` be the family of all **inclusion-minimal dominating sets**
    of :math:`G`. A set :math:`S \subseteq V(G)` is a transversal (hitting set) for
    :math:`\mathcal{D}` if
    :math:`S \cap D \neq \varnothing` for all :math:`D \in \mathcal{D}`.

    This function returns the minimum possible size:
    :math:`\tau_{\min\mathrm{dom}}(G) = \min\{ |S| : S \cap D \neq \varnothing \;\; \forall D \in \mathcal{D}\}`.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (intended for finite simple undirected graphs).
    max_n : int, default=20
        Safety cutoff on :math:`|V(G)|`. This routine enumerates all inclusion-minimal
        dominating sets and then solves a hitting-set problem over :math:`V(G)`, both
        of which can be exponential.

    Notes
    -----
    Exact brute force:

    1. Enumerate all inclusion-minimal dominating sets :math:`\mathcal{D}`.
    2. Search subsets :math:`S \subseteq V(G)` in increasing cardinality until one intersects every :math:`D \in \mathcal{D}`.

    Conventions:
    - If :math:`|V(G)| = 0`, returns ``0``.
    - If :math:`|V(G)| > 0`, then :math:`\mathcal{D}` is nonempty, so the answer is at least ``1``.

    Returns
    -------
    int
        The minimum size of a vertex set intersecting every inclusion-minimal dominating set of :math:`G`.

    Raises
    ------
    ValueError
        If :math:`|V(G)| > \texttt{max_n}`.

    Examples
    --------
    >>> import networkx as nx
    >>> import graphcalc.graphs as gc
    >>> G = nx.complete_graph(4)
    >>> gc.minimal_dominating_set_transversal_number(G)
    4
    >>> H = nx.star_graph(5)
    >>> gc.minimal_dominating_set_transversal_number(H)
    2
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0
    if n > max_n:
        raise ValueError(
            f"minimal_dominating_set_transversal_number brute force intended for n <= {max_n}, got n={n}"
        )

    minimal_dom_sets = _all_minimal_dominating_sets(G)
    # For n>0 this should be nonempty; keep a guard anyway.
    if not minimal_dom_sets:
        return 0

    nodes = list(G.nodes())
    for k in range(1, n + 1):
        for S in itertools.combinations(nodes, k):
            Sset = set(S)
            if all(Sset & D for D in minimal_dom_sets):
                return k

    return n
