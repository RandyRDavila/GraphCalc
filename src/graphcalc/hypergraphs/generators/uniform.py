# graphcalc/hypergraphs/generators/uniform.py

from __future__ import annotations

from itertools import combinations, product
import math
import random
from typing import Hashable, Iterable, Sequence, TypeAlias, TypeVar

from graphcalc.hypergraphs.core.basics import Hypergraph


Vertex = TypeVar("Vertex", bound=Hashable)
VertexLike: TypeAlias = Hashable
EdgeLike: TypeAlias = Iterable[VertexLike]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _labels_or_range(n: int, labels: Sequence[Vertex] | None = None) -> list[Vertex | int]:
    """
    Return the supplied labels as a list, or ``list(range(n))`` if labels are omitted.

    Parameters
    ----------
    n : int
        Number of vertices.
    labels : sequence, optional
        Vertex labels. Must have length ``n`` if provided.

    Returns
    -------
    list
        Vertex labels.

    Raises
    ------
    ValueError
        If ``n < 0`` or ``labels`` has length different from ``n``.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if labels is None:
        return list(range(n))
    labels = list(labels)
    if len(labels) != n:
        raise ValueError("labels must have length n")
    return labels


def _inv_mod(a: int, p: int) -> int:
    """
    Compute the multiplicative inverse of ``a`` modulo ``p``.

    Parameters
    ----------
    a : int
        Integer to invert.
    p : int
        Modulus.

    Returns
    -------
    int
        Integer ``x`` such that ``a * x ≡ 1 (mod p)``.

    Raises
    ------
    ValueError
        If ``a ≡ 0 (mod p)``.
    """
    a %= p
    if a == 0:
        raise ValueError("0 has no inverse mod p")
    return pow(a, -1, p)


def _check_k(k: int) -> None:
    """
    Validate the uniformity parameter.

    Parameters
    ----------
    k : int
        Edge size.

    Raises
    ------
    ValueError
        If ``k < 2``.
    """
    if k < 2:
        raise ValueError("k must be >= 2 for k-uniform hypergraphs.")


def _check_prime(q: int) -> None:
    """
    Validate that ``q`` is prime.

    Parameters
    ----------
    q : int
        Candidate prime.

    Raises
    ------
    ValueError
        If ``q < 2`` or ``q`` is composite.
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    if q == 2:
        return
    if q % 2 == 0:
        raise ValueError("this implementation expects prime q")
    limit = math.isqrt(q)
    for p in range(3, limit + 1, 2):
        if q % p == 0:
            raise ValueError("this implementation expects prime q")


def _build_hypergraph(
    edges: Iterable[EdgeLike],
    vertices: Iterable[VertexLike],
    rank_max: int,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct a hypergraph with a default ``rank_max``.

    Parameters
    ----------
    edges : iterable of iterable
        Hyperedges.
    vertices : iterable
        Vertex set.
    rank_max : int
        Maximum rank to set if not already provided.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Constructed hypergraph.
    """
    hg_kwargs.setdefault("rank_max", rank_max)
    return Hypergraph.from_edges(edges, vertices=vertices, **hg_kwargs)


# ---------------------------------------------------------------------------
# Canonical k-uniform families
# ---------------------------------------------------------------------------
def complete_k_uniform(
    n: int,
    k: int,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct the complete ``k``-uniform hypergraph on ``n`` vertices.

    The hyperedges are all ``k``-subsets of the vertex set.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Complete ``k``-uniform hypergraph.

    Raises
    ------
    ValueError
        If ``k < 2`` or ``k > n``.
    """
    _check_k(k)
    if not (0 <= k <= n):
        raise ValueError("require 0 <= k <= n")
    vertices = _labels_or_range(n, labels)
    edges = (set(c) for c in combinations(vertices, k))
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def erdos_ko_rado_star(
    n: int,
    k: int,
    t: int = 1,
    core: Iterable[VertexLike] | None = None,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct an Erdős-Ko-Rado ``t``-star.

    This is the family of all ``k``-sets containing a fixed ``t``-set
    called the core.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    t : int, default=1
        Size of the fixed core.
    core : iterable, optional
        Chosen core. If omitted, the first ``t`` vertices are used.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        The EKR ``t``-star.

    Raises
    ------
    ValueError
        If the parameters are inconsistent or if ``core`` is not a valid
        ``t``-subset of the vertex set.
    """
    _check_k(k)
    if not (0 <= t <= k <= n):
        raise ValueError("require 0 <= t <= k <= n")

    vertices = _labels_or_range(n, labels)
    if core is None:
        core_tuple = tuple(vertices[:t])
    else:
        core_tuple = tuple(core)

    if len(core_tuple) != t or any(v not in vertices for v in core_tuple) or len(set(core_tuple)) != t:
        raise ValueError("core must be a t-subset of V")

    rest = [v for v in vertices if v not in core_tuple]
    edges = (set(core_tuple) | set(c) for c in combinations(rest, k - t))
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def hilton_milner(
    n: int,
    k: int,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct the Hilton-Milner intersecting family.

    Standard construction assumes ``n > 2k >= 4``. On the vertex set
    ``V = [v_0, ..., v_{n-1}]``, let ``S = {v_0, ..., v_{k-1}}`` and
    ``x = v_k``. The family is

    ``{S} ∪ {A : |A| = k, x in A, A ∩ S != empty}``.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Hilton-Milner family.

    Raises
    ------
    ValueError
        If ``k < 2`` or ``n <= 2k``.
    """
    _check_k(k)
    if n <= 2 * k:
        raise ValueError("Hilton-Milner is standard for n > 2k.")

    vertices = _labels_or_range(n, labels)
    S = set(vertices[:k])
    x = vertices[k]
    rest = [v for v in vertices if v != x]

    edges: list[set[Vertex | int]] = [set(S)]
    for A in combinations(rest, k - 1):
        edge = {x, *A}
        if edge & S:
            edges.append(set(edge))

    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def k_uniform_matching(
    n: int,
    k: int,
    m: int,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct a matching of ``m`` disjoint ``k``-edges.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    m : int
        Number of disjoint edges.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Hypergraph consisting of ``m`` pairwise disjoint edges.

    Raises
    ------
    ValueError
        If ``m < 0`` or ``n < m * k``.
    """
    _check_k(k)
    if m < 0:
        raise ValueError("m must be >= 0")
    if m * k > n:
        raise ValueError("need n >= m*k")

    vertices = _labels_or_range(n, labels)
    edges = [set(vertices[i * k : (i + 1) * k]) for i in range(m)]
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def k_uniform_star(
    n: int,
    k: int,
    center: VertexLike | None = None,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct the ``k``-uniform star centered at one vertex.

    The hyperedges are all ``k``-sets containing the chosen center.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    center : hashable, optional
        Distinguished center vertex. If omitted, the first vertex is used.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        ``k``-uniform star.

    Raises
    ------
    ValueError
        If ``n < k`` or ``center`` is not in the vertex set.
    """
    _check_k(k)
    if n < k:
        raise ValueError("need n >= k")

    vertices = _labels_or_range(n, labels)
    if center is None:
        center_value = vertices[0]
    else:
        center_value = center

    if center_value not in vertices:
        raise ValueError("center must be in V")

    leaves = [v for v in vertices if v != center_value]
    edges = (set((center_value, *c)) for c in combinations(leaves, k - 1))
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def sunflower(
    k: int,
    petals: int,
    core_size: int = 1,
    start_label: int = 0,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct a ``k``-uniform sunflower (delta-system).

    All edges share the same core of size ``core_size`` and are otherwise
    pairwise disjoint.

    Parameters
    ----------
    k : int
        Uniform edge size.
    petals : int
        Number of petals (edges).
    core_size : int, default=1
        Size of the common core.
    start_label : int, default=0
        Starting integer used for vertex labels.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Sunflower hypergraph.

    Raises
    ------
    ValueError
        If ``k < 2``, ``petals < 0``, or ``core_size`` is not in
        ``[0, k]``.
    """
    _check_k(k)
    if not (0 <= core_size <= k):
        raise ValueError("need 0 <= core_size <= k")
    if petals < 0:
        raise ValueError("petals must be >= 0")

    core = list(range(start_label, start_label + core_size))
    next_vertex = start_label + core_size
    edges: list[set[int]] = []

    for _ in range(petals):
        petal_vertices = list(range(next_vertex, next_vertex + (k - core_size)))
        next_vertex += (k - core_size)
        edges.append(set(core + petal_vertices))

    vertices = list(range(start_label, next_vertex))
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


# ---------------------------------------------------------------------------
# Cyclic / path / cycle constructions
# ---------------------------------------------------------------------------
def tight_cycle(
    n: int,
    k: int,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct the ``k``-uniform tight cycle on ``n`` vertices.

    The edges are the cyclic consecutive ``k``-windows modulo ``n``.
    There are exactly ``n`` edges.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Tight cycle.

    Raises
    ------
    ValueError
        If ``n < k``.
    """
    _check_k(k)
    if n < k:
        raise ValueError("need n >= k")

    vertices = _labels_or_range(n, labels)
    edges = [{vertices[(i + j) % n] for j in range(k)} for i in range(n)]
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def loose_cycle(
    r: int,
    k: int,
    start_label: int = 0,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct a ``k``-uniform loose cycle with ``r`` edges.

    Consecutive edges intersect in exactly one vertex, and nonconsecutive
    edges are disjoint. The vertex count is ``r * (k - 1)``.

    Parameters
    ----------
    r : int
        Number of edges.
    k : int
        Uniform edge size.
    start_label : int, default=0
        Starting integer used for vertex labels.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Loose cycle. If ``r == 0``, returns the empty hypergraph.

    Raises
    ------
    ValueError
        If ``k < 2`` or ``r < 0``.
    """
    _check_k(k)
    if r < 0:
        raise ValueError("r must be >= 0")
    if r == 0:
        return _build_hypergraph([], [], rank_max=k, **hg_kwargs)

    shared = list(range(start_label, start_label + r))
    next_vertex = start_label + r
    edges: list[set[int]] = []

    for i in range(r):
        edge = {shared[i - 1], shared[i]}
        private = list(range(next_vertex, next_vertex + (k - 2)))
        next_vertex += (k - 2)
        edge.update(private)
        edges.append(edge)

    vertices = list(range(start_label, next_vertex))
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def loose_path(
    r: int,
    k: int,
    start_label: int = 0,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct a ``k``-uniform loose path with ``r`` edges.

    Consecutive edges intersect in exactly one vertex, and nonconsecutive
    edges are disjoint. The vertex count is ``r * (k - 1) + 1``.

    Parameters
    ----------
    r : int
        Number of edges.
    k : int
        Uniform edge size.
    start_label : int, default=0
        Starting integer used for vertex labels.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Loose path. If ``r == 0``, returns the empty hypergraph.

    Raises
    ------
    ValueError
        If ``k < 2`` or ``r < 0``.
    """
    _check_k(k)
    if r < 0:
        raise ValueError("r must be >= 0")
    if r == 0:
        return _build_hypergraph([], [], rank_max=k, **hg_kwargs)

    shared = list(range(start_label, start_label + (r + 1)))
    next_vertex = start_label + (r + 1)
    edges: list[set[int]] = []

    for i in range(r):
        edge = {shared[i], shared[i + 1]}
        private = list(range(next_vertex, next_vertex + (k - 2)))
        next_vertex += (k - 2)
        edge.update(private)
        edges.append(edge)

    vertices = list(range(start_label, next_vertex))
    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


# ---------------------------------------------------------------------------
# Arithmetic / additive combinatorics
# ---------------------------------------------------------------------------
def arithmetic_progressions(
    n: int,
    k: int,
    cyclic: bool = True,
    include_reverse: bool = False,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct the ``k``-term arithmetic progression hypergraph on ``Z_n``.

    The vertices are ``0, 1, ..., n - 1`` and hyperedges are ``k``-term
    arithmetic progressions.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Progression length and edge size.
    cyclic : bool, default=True
        If True, progressions are taken modulo ``n``. If False, only
        non-wrapping progressions inside ``[0, n-1]`` are included.
    include_reverse : bool, default=False
        In the cyclic model, whether to include both steps ``d`` and ``-d``.
        When False, attempts to avoid double counting reverse progressions.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Arithmetic progression hypergraph.

    Raises
    ------
    ValueError
        If ``k < 2`` or ``k > n``.
    """
    _check_k(k)
    if k > n:
        raise ValueError("need k <= n")

    vertices = list(range(n))
    edges: set[frozenset[int]] = set()

    if cyclic:
        steps = range(1, n) if include_reverse else range(1, (n // 2) + 1)
        for a in range(n):
            for d in steps:
                pts = [(a + i * d) % n for i in range(k)]
                if len(set(pts)) == k:
                    edges.add(frozenset(pts))
    else:
        for a in range(n):
            for d in range(1, n):
                if a + (k - 1) * d >= n:
                    break
                pts = [a + i * d for i in range(k)]
                edges.add(frozenset(pts))

    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


# ---------------------------------------------------------------------------
# Finite geometry
# ---------------------------------------------------------------------------
def affine_plane(q: int, **hg_kwargs) -> Hypergraph:
    """
    Construct the affine plane ``AG(2, q)`` for prime ``q``.

    Vertices are points of ``F_q^2`` and edges are affine lines.

    Parameters
    ----------
    q : int
        Prime order of the field.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Affine plane hypergraph with ``q^2`` vertices, ``q^2 + q`` lines,
        and line size ``q``.

    Raises
    ------
    ValueError
        If ``q`` is not prime.
    """
    _check_prime(q)

    vertices = [(x, y) for x in range(q) for y in range(q)]
    edges: list[set[tuple[int, int]]] = []

    for m in range(q):
        for b in range(q):
            edges.append({(x, (m * x + b) % q) for x in range(q)})

    for c in range(q):
        edges.append({(c, y) for y in range(q)})

    return _build_hypergraph(edges, vertices, rank_max=q, **hg_kwargs)


def projective_plane(q: int, **hg_kwargs) -> Hypergraph:
    """
    Construct the projective plane ``PG(2, q)`` for prime ``q``.

    Vertices are projective points and hyperedges are projective lines.

    Parameters
    ----------
    q : int
        Prime order of the field.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Projective plane hypergraph with ``q^2 + q + 1`` vertices,
        ``q^2 + q + 1`` lines, and line size ``q + 1``.

    Raises
    ------
    ValueError
        If ``q`` is not prime.
    """
    _check_prime(q)

    def norm(triple: tuple[int, int, int]) -> tuple[int, int, int]:
        x, y, z = (t % q for t in triple)
        if x == y == z == 0:
            raise ValueError("cannot normalize 0 vector")
        if x != 0:
            inv = _inv_mod(x, q)
            return (1, (y * inv) % q, (z * inv) % q)
        if y != 0:
            inv = _inv_mod(y, q)
            return ((x * inv) % q, 1, (z * inv) % q)
        inv = _inv_mod(z, q)
        return ((x * inv) % q, (y * inv) % q, 1)

    points = {
        norm((x, y, z))
        for x, y, z in product(range(q), repeat=3)
        if not (x == y == z == 0)
    }
    sorted_points = sorted(points)

    lines = {
        norm((a, b, c))
        for a, b, c in product(range(q), repeat=3)
        if not (a == b == c == 0)
    }
    sorted_lines = sorted(lines)

    edges: list[set[tuple[int, int, int]]] = []
    for a, b, c in sorted_lines:
        edge: set[tuple[int, int, int]] = set()
        for x, y, z in sorted_points:
            if (a * x + b * y + c * z) % q == 0:
                edge.add((x, y, z))
        edges.append(edge)

    return _build_hypergraph(edges, sorted_points, rank_max=q + 1, **hg_kwargs)


def fano_plane(**hg_kwargs) -> Hypergraph:
    """
    Construct the Fano plane.

    This is the projective plane ``PG(2, 2)``, a 3-uniform hypergraph on
    7 vertices with 7 edges.

    Parameters
    ----------
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Fano plane.
    """
    return projective_plane(2, **hg_kwargs)


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------
def random_k_uniform(
    n: int,
    k: int,
    p: float | None = None,
    m: int | None = None,
    seed: object | None = None,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct a random ``k``-uniform hypergraph on ``n`` vertices.

    Exactly one of ``p`` or ``m`` must be supplied.

    - If ``p`` is provided, each ``k``-subset is included independently
      with probability ``p``.
    - If ``m`` is provided, exactly ``m`` distinct edges are chosen
      uniformly without replacement.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    p : float, optional
        Inclusion probability for the binomial model.
    m : int, optional
        Number of edges for the fixed-size model.
    seed : object, optional
        Seed passed to ``random.Random``.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        Random ``k``-uniform hypergraph.

    Raises
    ------
    ValueError
        If ``k > n``, if both or neither of ``p`` and ``m`` are supplied,
        if ``p`` is outside ``[0, 1]``, or if ``m`` is out of range.
    """
    _check_k(k)
    if not (0 <= k <= n):
        raise ValueError("require 0 <= k <= n")
    if (p is None) == (m is None):
        raise ValueError("specify exactly one of p or m")

    vertices = _labels_or_range(n, labels)
    rng = random.Random(seed)
    all_edges = list(combinations(vertices, k))

    if p is not None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0,1]")
        edges = [set(e) for e in all_edges if rng.random() < p]
    else:
        if not (0 <= m <= len(all_edges)):
            raise ValueError("m out of range")
        edges = [set(e) for e in rng.sample(all_edges, m)]

    return _build_hypergraph(edges, vertices, rank_max=k, **hg_kwargs)


def random_k_regular_configuration(
    n: int,
    k: int,
    d: int,
    seed: object | None = None,
    max_tries: int = 2000,
    labels: Sequence[Vertex] | None = None,
    **hg_kwargs,
) -> Hypergraph:
    """
    Construct a random simple ``d``-regular ``k``-uniform hypergraph
    using a configuration-style model.

    This attempts to partition vertex stubs into ``k``-blocks. Candidate
    blocks containing repeated vertices are rejected. Duplicate edges are
    deduplicated, so exact regularity may fail if too many rejections occur.
    The best attempt found within ``max_tries`` is returned.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Uniform edge size.
    d : int
        Target vertex degree.
    seed : object, optional
        Seed passed to ``random.Random``.
    max_tries : int, default=2000
        Maximum number of configuration attempts.
    labels : sequence, optional
        Vertex labels. If omitted, uses ``range(n)``.
    **hg_kwargs
        Additional keyword arguments passed to ``Hypergraph.from_edges``.

    Returns
    -------
    Hypergraph
        A simple ``k``-uniform hypergraph, often but not always exactly
        ``d``-regular.

    Raises
    ------
    ValueError
        If ``n * d`` is not divisible by ``k``.
    """
    _check_k(k)
    if d < 0:
        raise ValueError("d must be >= 0")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")
    if n * d % k != 0:
        raise ValueError("need n*d divisible by k")

    vertices = _labels_or_range(n, labels)
    rng = random.Random(seed)
    target_m = (n * d) // k
    index_vertices = list(range(n))

    best_edges: set[frozenset[Vertex | int]] = set()

    for _ in range(max_tries):
        stubs: list[int] = []
        for i in index_vertices:
            stubs.extend([i] * d)
        rng.shuffle(stubs)

        edges: list[frozenset[Vertex | int]] = []
        valid = True
        for i in range(0, len(stubs), k):
            block = stubs[i : i + k]
            if len(set(block)) != k:
                valid = False
                break
            edges.append(frozenset(vertices[j] for j in block))

        if not valid:
            continue

        edge_set = set(edges)
        if len(edge_set) > len(best_edges):
            best_edges = edge_set
        if len(edge_set) == target_m:
            return _build_hypergraph(edge_set, vertices, rank_max=k, **hg_kwargs)

    return _build_hypergraph(best_edges, vertices, rank_max=k, **hg_kwargs)
