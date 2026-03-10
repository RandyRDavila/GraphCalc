# src/graphcalc/hypergraphs/invariants/dsi.py

from __future__ import annotations

import heapq
from typing import Optional

from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
from graphcalc.metadata import invariant_metadata

__all__ = [
    "degree_sequence",
    "reverse_degree_sequence",
    "hh_residue_graph_degree_sequence",
    "generalized_havel_hakimi_residue",
    "generalized_annihilation_number",
]


def _infer_k_uniform(H: HypergraphLike, k: Optional[int] = None) -> int:
    """
    Infer or validate the target uniformity parameter.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int or None, default=None
        Desired uniformity. If None, infer from the first hyperedge when
        possible.

    Returns
    -------
    int
        The inferred or supplied value of ``k``.

    Raises
    ------
    ValueError
        If ``k`` is None and the hypergraph has no hyperedges.
    ValueError
        If ``k`` is negative.
    """
    if k is not None:
        k = int(k)
        if k < 0:
            raise ValueError("k must be >= 0.")
        return k

    if not H.E:
        raise ValueError("Cannot infer k from an empty hypergraph.")
    return len(next(iter(H.E)))


def _validate_k_uniform(H: HypergraphLike, k: int) -> None:
    """
    Validate that a hypergraph is k-uniform.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int
        Target edge size.

    Raises
    ------
    ValueError
        If some hyperedge has size different from ``k``.
    """
    if any(len(edge) != k for edge in H.E):
        raise ValueError("H is not k-uniform.")


@invariant_metadata(
    display_name="Degree sequence",
    notation=r"d(H)",
    category="degree-sequence invariants",
    aliases=("hypergraph degree sequence",),
    definition=(
        "The degree sequence of a hypergraph H is the list of vertex degrees of H, usually arranged in nonincreasing or nondecreasing order."
    ),
)
@require_hypergraph_like
def degree_sequence(H: HypergraphLike, *, nonincreasing: bool = True) -> list[int]:
    r"""
    Return the vertex degree sequence of a hypergraph.

    The **degree sequence** of a hypergraph is the list of vertex degrees,
    optionally sorted in nonincreasing or nondecreasing order.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    nonincreasing : bool, default=True
        If True, return the sequence sorted from largest to smallest.
        If False, return the sequence sorted from smallest to largest.

    Returns
    -------
    list of int
        The sorted degree sequence of ``H``.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> from graphcalc.hypergraphs.invariants.dsi import degree_sequence
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}, {2, 3, 4}])
    >>> degree_sequence(H)
    [3, 2, 1, 1]
    >>> degree_sequence(H, nonincreasing=False)
    [1, 1, 2, 3]
    """
    seq = list(H.degrees().values())
    return sorted(seq, reverse=nonincreasing)


@invariant_metadata(
    display_name="Reverse degree sequence",
    notation=r"d^{\\uparrow}(H)",
    category="degree-sequence invariants",
    aliases=("nondecreasing degree sequence",),
    definition=(
        "The reverse degree sequence of a hypergraph H is its degree sequence arranged in nondecreasing order."
    ),
)
@require_hypergraph_like
def reverse_degree_sequence(H: HypergraphLike) -> list[int]:
    r"""
    Return the nondecreasing degree sequence of a hypergraph.

    This is a convenience wrapper for ``degree_sequence(H,
    nonincreasing=False)``.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.

    Returns
    -------
    list of int
        The degree sequence sorted from smallest to largest.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> from graphcalc.hypergraphs.invariants.dsi import reverse_degree_sequence
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}, {2, 3, 4}])
    >>> reverse_degree_sequence(H)
    [1, 1, 2, 3]
    """
    return degree_sequence(H, nonincreasing=False)


@invariant_metadata(
    display_name="Havel--Hakimi residue",
    notation=r"R(d)",
    category="degree-sequence invariants",
    aliases=("graph residue", "HH residue"),
    definition=(
        "The Havel--Hakimi residue of a nonnegative integer sequence d is the number of entries remaining when the Havel--Hakimi reduction process terminates with all remaining entries equal to zero."
    ),
)
def hh_residue_graph_degree_sequence(
    deg_seq: list[int] | tuple[int, ...],
    *,
    trace: bool = False,
) -> int:
    r"""
    Return the classical Havel--Hakimi residue of a degree sequence.

    Let :math:`d = (d_1, \dots, d_n)` be a finite sequence of nonnegative
    integers. The **Havel--Hakimi reduction** repeatedly transforms the
    sequence as follows:

    1. Sort the current sequence in nonincreasing order.
    2. If the first entry is 0, stop.
    3. Remove the first entry :math:`s`.
    4. Subtract 1 from each of the next :math:`s` entries.

    If at any step :math:`s` exceeds the number of remaining entries, or a
    negative entry is produced, then the sequence is not graphical and the
    process is invalid.

    The **residue** is the number of entries remaining when the process stops,
    at which point all entries are zero.

    For a simple graph :math:`G`, the residue :math:`R(G)` is defined as the
    residue of its degree sequence.

    Parameters
    ----------
    deg_seq : list[int] or tuple[int, ...]
        A finite sequence of integer degrees.
    trace : bool, default=False
        If True, print the intermediate sequences after each reduction step.

    Returns
    -------
    int
        The Havel--Hakimi residue of ``deg_seq``.

    Raises
    ------
    ValueError
        If a negative entry occurs in the input sequence.
    ValueError
        If the sequence is not graphical under the Havel--Hakimi process.

    Notes
    -----
    This routine is purely sequence-based and does not require an explicit
    graph realization.

    """
    d = [int(x) for x in deg_seq]

    if any(x < 0 for x in d):
        raise ValueError("Degree sequence must be nonnegative.")

    while True:
        d.sort(reverse=True)

        if not d or d[0] == 0:
            return len(d)

        s = d.pop(0)
        if s > len(d):
            raise ValueError(
                "Non-graphic sequence encountered (s exceeds remaining length)."
            )

        for i in range(s):
            d[i] -= 1
            if d[i] < 0:
                raise ValueError(
                    "Non-graphic sequence encountered (negative entry produced)."
                )

        if trace:
            print(d)


@invariant_metadata(
    display_name="Generalized Havel--Hakimi residue",
    notation=r"R_k(H)",
    category="degree-sequence invariants",
    aliases=("generalized residue", "hypergraph Havel--Hakimi residue"),
    definition=(
        "The generalized Havel--Hakimi residue of a k-uniform hypergraph H is the residue obtained from its degree multiset by the fixed greedy reduction rule implemented here; for k=2 it coincides with the classical graph Havel--Hakimi residue."
    ),
)
@require_hypergraph_like
def generalized_havel_hakimi_residue(
    H: HypergraphLike,
    *,
    k: Optional[int] = None,
    trace: bool = False,
) -> int:
    r"""
    Return a Havel--Hakimi-type residue for a k-uniform hypergraph.

    This function defines a degree-sequence invariant for uniform
    hypergraphs that extends the classical Havel--Hakimi residue of graphs.

    Let :math:`H=(V,E)` be a finite ``k``-uniform hypergraph, and let
    :math:`d(v)` denote the degree of vertex :math:`v`. Write the multiset of
    vertex degrees as the degree sequence of ``H``.

    Case ``k = 2``
        The function returns the classical Havel--Hakimi residue of the graph
        degree sequence.

    Case ``k >= 3``
        The function applies the following fixed greedy reduction rule to the
        degree multiset:

        Repeat while the current maximum degree is positive:

        1. Remove one occurrence of the current maximum degree :math:`s`.
        2. Perform :math:`s` micro-steps. In each micro-step:
           select the ``k-1`` currently largest remaining degrees and subtract
           1 from each of them.

        If at any micro-step fewer than ``k-1`` positive degrees remain, the
        process is declared to fail and the function returns 0.

    When the process terminates successfully, all remaining entries are zero,
    and the residue is defined to be the number of remaining entries.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int or None, default=None
        Target uniformity. If None, infer from the hyperedges of ``H``.
    trace : bool, default=False
        If True, print intermediate sequences for the classical ``k=2`` case.
        For ``k>=3``, no intermediate trace is printed.

    Returns
    -------
    int
        The residue value defined by the above reduction process.

    Raises
    ------
    ValueError
        If ``H`` is not ``k``-uniform.
    ValueError
        If ``k`` is None and ``H`` has no hyperedges.

    Notes
    -----
    - If ``H`` has no hyperedges, the function returns ``|V(H)|``.
    - For ``k=2``, this exactly recovers the classical graph residue.
    - For ``k>=3``, this defines a consistent degree-sequence invariant via a
      fixed greedy rule, but no universal extremal interpretation is assumed
      unless separately proved.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> from graphcalc.hypergraphs.invariants.dsi import generalized_havel_hakimi_residue
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> generalized_havel_hakimi_residue(H, k=2)
    2
    """
    if not H.E:
        return len(H.V)

    k0 = _infer_k_uniform(H, k)
    _validate_k_uniform(H, k0)

    degs = list(H.degrees().values())

    if k0 == 2:
        return hh_residue_graph_degree_sequence(degs, trace=trace)

    heap = [-d for d in degs]
    heapq.heapify(heap)

    while heap and (-heap[0]) > 0:
        s = -heapq.heappop(heap)

        for _ in range(s):
            if len(heap) < (k0 - 1):
                return 0

            picked = []
            for _j in range(k0 - 1):
                val = -heapq.heappop(heap)
                if val <= 0:
                    return 0
                picked.append(val - 1)

            for val in picked:
                heapq.heappush(heap, -val)

    return len(heap)


@invariant_metadata(
    display_name="Generalized annihilation number",
    notation=r"a_k(H)",
    category="degree-sequence invariants",
    aliases=("scaled annihilation number",),
    definition=(
        "The generalized annihilation number of a k-uniform hypergraph H is the largest integer t such that the sum of the t smallest vertex degrees is at most (k-1)|E(H)|."
    ),
)
@require_hypergraph_like
def generalized_annihilation_number(
    H: HypergraphLike,
    k: Optional[int] = None,
) -> int:
    r"""
    Return the scaled annihilation number of a k-uniform hypergraph.

    Let :math:`H` be a ``k``-uniform hypergraph with ``m = |E(H)|`` and let

    .. math::
        d_1 \le d_2 \le \cdots \le d_n

    be the nondecreasing vertex degree sequence.

    The **scaled annihilation number** is defined by

    .. math::
        a_k(H) =
        \max \left\{
            t :
            \sum_{i=1}^{t} d_i \le (k-1)m
        \right\}.

    For ``k=2``, this reduces to the classical annihilation number of a graph,
    since ``(k-1)m = m``.

    Parameters
    ----------
    H : HypergraphLike
        A finite hypergraph.
    k : int or None, default=None
        Target uniformity. If None, infer from the hyperedges of ``H``.

    Returns
    -------
    int
        The scaled annihilation number :math:`a_k(H)`.

    Raises
    ------
    ValueError
        If ``H`` is not ``k``-uniform.
    ValueError
        If ``k`` is None and ``H`` has no hyperedges.

    Notes
    -----
    If ``H`` has no hyperedges, the function returns ``|V(H)|``.

    For a ``k``-uniform hypergraph, this parameter depends only on the degree
    multiset, since

    .. math::
        \sum_{v \in V(H)} d(v) = k|E(H)|.

    Examples
    --------
    >>> import graphcalc.hypergraphs as gc
    >>> from graphcalc.hypergraphs.invariants.dsi import generalized_annihilation_number
    >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
    >>> generalized_annihilation_number(H, k=2)
    2
    """
    if H.m == 0:
        return len(H.V)

    k0 = _infer_k_uniform(H, k)
    _validate_k_uniform(H, k0)

    degs = sorted(H.degrees().values())
    target = (k0 - 1) * H.m

    partial_sum = 0
    t = 0
    for d in degs:
        if partial_sum + d <= target:
            partial_sum += d
            t += 1
        else:
            break

    return t
