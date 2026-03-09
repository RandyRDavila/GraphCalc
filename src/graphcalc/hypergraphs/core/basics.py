from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import combinations
from math import sqrt
from typing import Dict, FrozenSet, Hashable, Optional, Set

__all__ = ["Vertex", "Hyperedge", "Hypergraph"]

Vertex = Hashable
Hyperedge = FrozenSet[Vertex]
EdgeLike = Iterable[Vertex]


class Hypergraph:
    """
    Finite simple undirected hypergraph represented as a set system.

    The hypergraph is stored as a pair ``(V, E)`` where ``V`` is a set of
    vertices and ``E`` is a set of hyperedges, each represented as a
    ``frozenset`` of vertices.

    Conventions
    -----------
    - simple: duplicate hyperedges are ignored
    - undirected: hyperedges are sets
    - empty hyperedges are disallowed by default
    - singleton hyperedges are disallowed by default
    - an optional upper bound on hyperedge size may be enforced via
      ``rank_max``

    Parameters
    ----------
    rank_max : int | None, default=None
        Optional maximum allowed hyperedge size.
    allow_empty : bool, default=False
        Whether the empty hyperedge is permitted.
    allow_singletons : bool, default=False
        Whether singleton hyperedges are permitted.
    auto_add_vertices : bool, default=True
        If True, vertices appearing in added hyperedges are automatically
        inserted into ``V``. If False, each hyperedge must already be a
        subset of ``V``.
    V : iterable, optional
        Initial vertex set.
    E : iterable of iterables, optional
        Initial hyperedge set.

    Notes
    -----
    The internal sets are stored privately as ``_V`` and ``_E``. Public
    access is provided through read-only properties.
    """

    def __init__(
        self,
        *,
        rank_max: Optional[int] = None,
        allow_empty: bool = False,
        allow_singletons: bool = False,
        auto_add_vertices: bool = True,
        V: Optional[Iterable[Vertex]] = None,
        E: Optional[Iterable[EdgeLike]] = None,
    ) -> None:
        self.rank_max = rank_max
        self.allow_empty = allow_empty
        self.allow_singletons = allow_singletons
        self.auto_add_vertices = auto_add_vertices

        self._V: Set[Vertex] = set(V or ())
        self._E: Set[Hyperedge] = set()

        self._validate_parameters()

        if E is not None:
            for edge in E:
                self.add_edge(edge)

    def __repr__(self) -> str:
        return (
            f"Hypergraph(n={self.n}, m={self.m}, rank={self.rank}, "
            f"rank_max={self.rank_max}, allow_empty={self.allow_empty}, "
            f"allow_singletons={self.allow_singletons}, "
            f"auto_add_vertices={self.auto_add_vertices})"
        )

    def __iter__(self) -> Iterator[Hyperedge]:
        return iter(self._E)

    def __len__(self) -> int:
        return len(self._E)

    def __contains__(self, edge: object) -> bool:
        if not isinstance(edge, frozenset):
            try:
                edge = frozenset(edge)  # type: ignore[arg-type]
            except TypeError:
                return False
        return edge in self._E

    @property
    def V(self) -> FrozenSet[Vertex]:
        """Return the vertex set as a read-only frozenset."""
        return frozenset(self._V)

    @property
    def E(self) -> FrozenSet[Hyperedge]:
        """Return the hyperedge set as a read-only frozenset."""
        return frozenset(self._E)

    @property
    def n(self) -> int:
        """Number of vertices."""
        return len(self._V)

    @property
    def m(self) -> int:
        """Number of hyperedges."""
        return len(self._E)

    @property
    def rank(self) -> int:
        """Maximum hyperedge size present, or 0 if there are no hyperedges."""
        return max((len(edge) for edge in self._E), default=0)

    @property
    def is_empty(self) -> bool:
        """Return True iff the hypergraph has no hyperedges."""
        return not self._E

    @staticmethod
    def _to_edge(edge: EdgeLike) -> Hyperedge:
        """Normalize an edge-like iterable into a frozenset hyperedge."""
        return frozenset(edge)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[EdgeLike],
        *,
        vertices: Optional[Iterable[Vertex]] = None,
        rank_max: Optional[int] = None,
        allow_empty: bool = False,
        allow_singletons: bool = False,
        auto_add_vertices: bool = True,
    ) -> "Hypergraph":
        """Construct a hypergraph from an iterable of hyperedges."""
        return cls(
            rank_max=rank_max,
            allow_empty=allow_empty,
            allow_singletons=allow_singletons,
            auto_add_vertices=auto_add_vertices,
            V=vertices,
            E=edges,
        )

    def copy(self) -> "Hypergraph":
        """Return a shallow copy of the hypergraph."""
        return Hypergraph(
            rank_max=self.rank_max,
            allow_empty=self.allow_empty,
            allow_singletons=self.allow_singletons,
            auto_add_vertices=self.auto_add_vertices,
            V=self._V,
            E=self._E,
        )

    def _min_edge_size(self) -> int:
        if self.allow_empty:
            return 0
        if self.allow_singletons:
            return 1
        return 2

    def _validate_parameters(self) -> None:
        min_size = self._min_edge_size()
        if self.rank_max is not None and self.rank_max < 0:
            raise ValueError("rank_max must be >= 0 or None.")
        if self.rank_max is not None and self.rank_max < min_size:
            raise ValueError(
                f"rank_max={self.rank_max} is incompatible with "
                f"allow_empty={self.allow_empty} and "
                f"allow_singletons={self.allow_singletons}."
            )

    def _validate_edge(self, edge: Hyperedge) -> None:
        size = len(edge)

        if not self.allow_empty and size == 0:
            raise ValueError("Empty hyperedge not allowed.")
        if not self.allow_singletons and size == 1:
            raise ValueError("Singleton hyperedge not allowed.")
        if self.rank_max is not None and size > self.rank_max:
            raise ValueError(f"Hyperedge size {size} exceeds rank_max={self.rank_max}.")
        if not self.auto_add_vertices and not edge.issubset(self._V):
            missing = edge.difference(self._V)
            raise ValueError(f"Hyperedge contains vertices not in V: {missing}")

    def validate(self) -> None:
        """Validate all stored hyperedges against the current configuration."""
        self._validate_parameters()
        for edge in self._E:
            self._validate_edge(edge)

    def has_vertex(self, v: Vertex) -> bool:
        return v in self._V

    def has_edge(self, edge: EdgeLike) -> bool:
        return self._to_edge(edge) in self._E

    def add_vertex(self, v: Vertex) -> None:
        self._V.add(v)

    def add_vertices(self, vertices: Iterable[Vertex]) -> None:
        self._V.update(vertices)

    def add_edge(self, edge: EdgeLike) -> Hyperedge:
        normalized = self._to_edge(edge)
        self._validate_edge(normalized)

        if self.auto_add_vertices:
            self._V.update(normalized)

        self._E.add(normalized)
        return normalized

    def add_edges(self, edges: Iterable[EdgeLike]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def remove_edge(self, edge: EdgeLike) -> None:
        self._E.discard(self._to_edge(edge))

    def remove_vertex(self, v: Vertex, *, drop_incident_edges: bool = True) -> None:
        """
        Remove a vertex.

        Parameters
        ----------
        v : hashable
            Vertex to remove.
        drop_incident_edges : bool, default=True
            If True, remove every incident hyperedge. If False, remove the
            vertex from each incident hyperedge and re-validate the result.
        """
        if v not in self._V:
            return

        incident = [edge for edge in self._E if v in edge]

        if drop_incident_edges:
            for edge in incident:
                self._E.discard(edge)
        else:
            new_edges: Set[Hyperedge] = set()
            for edge in self._E:
                if v not in edge:
                    new_edges.add(edge)
                    continue
                new_edge = frozenset(x for x in edge if x != v)
                self._validate_edge(new_edge)
                new_edges.add(new_edge)
            self._E = new_edges

        self._V.discard(v)

    def clear(self) -> None:
        self._V.clear()
        self._E.clear()

    def vertices(self) -> Iterator[Vertex]:
        return iter(self._V)

    def edges(self) -> Iterator[Hyperedge]:
        return iter(self._E)

    def edge_sizes(self) -> Dict[Hyperedge, int]:
        return {edge: len(edge) for edge in self._E}

    def edge_size_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for edge in self._E:
            size = len(edge)
            counts[size] = counts.get(size, 0) + 1
        return counts

    def degrees(self) -> Dict[Vertex, int]:
        deg = {v: 0 for v in self._V}
        for edge in self._E:
            for v in edge:
                deg[v] += 1
        return deg

    def degree(self, v: Vertex) -> int:
        if v not in self._V:
            raise ValueError(f"Vertex {v!r} is not in V.")
        return sum(1 for edge in self._E if v in edge)

    def subset_degree(self, vertices: Iterable[Vertex]) -> int:
        """
        Return the number of hyperedges containing the given vertex subset.

        This is also commonly called the codegree of the subset.
        """
        subset = frozenset(vertices)
        if not subset.issubset(self._V):
            missing = subset.difference(self._V)
            raise ValueError(f"Subset contains vertices not in V: {missing}")
        return sum(1 for edge in self._E if subset.issubset(edge))

    def d_degree(self, S: Iterable[Vertex]) -> int:
        """Backward-compatible alias for subset_degree()."""
        return self.subset_degree(S)

    def incident_edges(self, v: Vertex) -> Set[Hyperedge]:
        if v not in self._V:
            raise ValueError(f"Vertex {v!r} is not in V.")
        return {edge for edge in self._E if v in edge}

    def neighbors(self, v: Vertex) -> Set[Vertex]:
        """Return the neighbors of ``v`` in the 2-section graph."""
        if v not in self._V:
            raise ValueError(f"Vertex {v!r} is not in V.")
        out: Set[Vertex] = set()
        for edge in self._E:
            if v in edge:
                out.update(edge)
        out.discard(v)
        return out

    def degree_stats(self) -> Dict[str, float]:
        values = list(self.degrees().values())
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0}

        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(mean),
            "stdev": float(sqrt(var)),
        }

    def is_uniform(self, k: Optional[int] = None) -> bool:
        """
        Return whether the hypergraph is uniform.

        If ``k`` is provided, test whether every hyperedge has size ``k``.
        Otherwise, test whether all hyperedges have the same size.
        """
        if not self._E:
            return True
        sizes = {len(edge) for edge in self._E}
        if k is None:
            return len(sizes) == 1
        return sizes == {k}

    def two_section_edges(self) -> Set[FrozenSet[Vertex]]:
        """Return the edge set of the 2-section graph."""
        out: Set[FrozenSet[Vertex]] = set()
        for edge in self._E:
            for u, v in combinations(edge, 2):
                out.add(frozenset((u, v)))
        return out

    def _vertex_to_edges(self) -> Dict[Vertex, Set[Hyperedge]]:
        v2e: Dict[Vertex, Set[Hyperedge]] = {v: set() for v in self._V}
        for edge in self._E:
            for v in edge:
                v2e[v].add(edge)
        return v2e

    def is_incidence_connected(self) -> bool:
        """
        Test connectivity of the incidence bipartite graph.

        Isolated vertices count: if ``V`` contains an isolated vertex, then
        the incidence graph is disconnected unless the whole hypergraph is
        empty.
        """
        if not self._V and not self._E:
            return True

        start: tuple[str, object]
        if self._V:
            start = ("v", next(iter(self._V)))
        else:
            start = ("e", next(iter(self._E)))

        v2e = self._vertex_to_edges()
        seen = {start}
        stack = [start]

        while stack:
            kind, obj = stack.pop()
            if kind == "v":
                for edge in v2e[obj]:  # type: ignore[index]
                    node = ("e", edge)
                    if node not in seen:
                        seen.add(node)
                        stack.append(node)
            else:
                for v in obj:  # type: ignore[assignment]
                    node = ("v", v)
                    if node not in seen:
                        seen.add(node)
                        stack.append(node)

        return len(seen) == self.n + self.m

    def induced_subhypergraph(self, vertices: Iterable[Vertex]) -> "Hypergraph":
        """
        Return the induced subhypergraph on a vertex subset.

        Hyperedges are intersected with the subset and only valid resulting
        hyperedges are retained.
        """
        subset = set(vertices)
        if not subset.issubset(self._V):
            missing = subset.difference(self._V)
            raise ValueError(f"Subset contains vertices not in V: {missing}")

        new_edges = []
        for edge in self._E:
            restricted = frozenset(edge & subset)
            try:
                self._validate_edge(restricted)
            except ValueError:
                continue
            new_edges.append(restricted)

        return Hypergraph(
            rank_max=self.rank_max,
            allow_empty=self.allow_empty,
            allow_singletons=self.allow_singletons,
            auto_add_vertices=False,
            V=subset,
            E=new_edges,
        )


    def dual(self) -> "Hypergraph":
        """
        Return the dual hypergraph.

        The vertices of the dual are the hyperedges of the original, indexed
        as integers ``0, 1, ..., m-1`` in the iteration order of ``self._E``.

        Notes
        -----
        The dual may contain singleton or empty hyperedges even if the original
        hypergraph disallows them, so the dual is constructed with
        ``allow_empty=True`` and ``allow_singletons=True``.
        """
        edge_list = list(self._E)
        dual_vertices = set(range(len(edge_list)))
        dual_edges = []

        for v in self._V:
            incidence = frozenset(i for i, edge in enumerate(edge_list) if v in edge)
            dual_edges.append(incidence)

        return Hypergraph(
            allow_empty=True,
            allow_singletons=True,
            auto_add_vertices=False,
            V=dual_vertices,
            E=dual_edges,
        )
