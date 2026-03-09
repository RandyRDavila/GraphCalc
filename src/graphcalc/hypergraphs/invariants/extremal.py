# # src/graphcalc/hypergraphs/invariants/extremal.py
# """
# Extremal, set-theoretic, and design-adjacent hypergraph parameters.

# This module contains hypergraph invariants and derived families that are
# central in extremal set theory and hypergraph theory, including codegrees,
# Sperner-ness, and lower/upper shadows.

# Currently included
# ------------------
# codegree
#     Number of hyperedges containing a given vertex subset.
# maximum_codegree
#     Maximum codegree over all subsets of a fixed size.
# minimum_codegree
#     Minimum codegree over all subsets of a fixed size.
# average_codegree
#     Average codegree over all subsets of a fixed size.
# is_sperner
#     Test whether no hyperedge properly contains another.
# lower_shadow
#     Return the lower shadow of the hyperedge family.
# lower_shadow_size
#     Return the size of the lower shadow.
# upper_shadow
#     Return the upper shadow of the hyperedge family.
# upper_shadow_size
#     Return the size of the upper shadow.
# """

# from __future__ import annotations

# from itertools import combinations
# from typing import FrozenSet, Hashable, Iterable, Set

# from graphcalc.hypergraphs.utils import HypergraphLike, require_hypergraph_like
# import pulp

# from graphcalc.solvers import with_solver


# __all__ = [
#     "codegree",
#     "maximum_codegree",
#     "minimum_codegree",
#     "average_codegree",
#     "is_sperner",
#     "lower_shadow",
#     "lower_shadow_size",
#     "upper_shadow",
#     "upper_shadow_size",
#     "fractional_matching_number",
#     "fractional_transversal_number",
# ]


# @require_hypergraph_like
# def codegree(H: HypergraphLike, S: Iterable[Hashable]) -> int:
#     r"""
#     Return the codegree of a vertex subset in a hypergraph.

#     The **codegree** of a subset :math:`S \subseteq V(H)` is the number of
#     hyperedges containing :math:`S`:

#     .. math::
#         d_H(S) = |\{ e \in E(H) : S \subseteq e \}|.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     S : iterable of hashable
#         Vertex subset whose codegree is to be computed.

#     Returns
#     -------
#     int
#         The number of hyperedges containing ``S``.

#     Notes
#     -----
#     The empty subset is contained in every hyperedge, so its codegree is
#     :math:`|E(H)|`.

#     Examples
#     --------
#     >>> import graphcalc.graphs as gc
#     >>> H = gc.Hypergraph(E=[{1, 2, 3}, {2, 3, 4}])
#     >>> gc.codegree(H, [2, 3])
#     2
#     >>> gc.codegree(H, [1, 4])
#     0
#     """
#     subset = frozenset(S)
#     return sum(1 for edge in H.E if subset.issubset(edge))


# @require_hypergraph_like
# def maximum_codegree(H: HypergraphLike, t: int = 2) -> int:
#     r"""
#     Return the maximum ``t``-codegree of a hypergraph.

#     The **maximum t-codegree** is

#     .. math::
#         \Delta_t(H) = \max\{ d_H(S) : S \subseteq V(H),\ |S|=t \}.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     t : int, default=2
#         Subset size.

#     Returns
#     -------
#     int
#         Maximum codegree over all ``t``-subsets of the vertex set.

#     Raises
#     ------
#     ValueError
#         If ``t < 0``.

#     Notes
#     -----
#     If ``t > |V(H)|``, the result is 0.
#     """
#     if t < 0:
#         raise ValueError("t must be >= 0")

#     vertices = list(H.V)
#     if t > len(vertices):
#         return 0

#     return max((codegree(H, subset) for subset in combinations(vertices, t)), default=0)


# @require_hypergraph_like
# def minimum_codegree(H: HypergraphLike, t: int = 2) -> int:
#     r"""
#     Return the minimum ``t``-codegree of a hypergraph.

#     The **minimum t-codegree** is

#     .. math::
#         \delta_t(H) = \min\{ d_H(S) : S \subseteq V(H),\ |S|=t \}.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     t : int, default=2
#         Subset size.

#     Returns
#     -------
#     int
#         Minimum codegree over all ``t``-subsets of the vertex set.

#     Raises
#     ------
#     ValueError
#         If ``t < 0``.

#     Notes
#     -----
#     If ``t > |V(H)|``, the result is 0.
#     """
#     if t < 0:
#         raise ValueError("t must be >= 0")

#     vertices = list(H.V)
#     if t > len(vertices):
#         return 0

#     return min((codegree(H, subset) for subset in combinations(vertices, t)), default=0)


# @require_hypergraph_like
# def average_codegree(H: HypergraphLike, t: int = 2) -> float:
#     r"""
#     Return the average ``t``-codegree of a hypergraph.

#     The **average t-codegree** is the mean of :math:`d_H(S)` over all
#     ``t``-subsets :math:`S \subseteq V(H)`.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     t : int, default=2
#         Subset size.

#     Returns
#     -------
#     float
#         Average codegree over all ``t``-subsets of the vertex set.

#     Raises
#     ------
#     ValueError
#         If ``t < 0``.

#     Notes
#     -----
#     If ``t > |V(H)|``, the result is 0.0.
#     """
#     if t < 0:
#         raise ValueError("t must be >= 0")

#     vertices = list(H.V)
#     if t > len(vertices):
#         return 0.0

#     subsets = list(combinations(vertices, t))
#     if not subsets:
#         return 0.0

#     total = sum(codegree(H, subset) for subset in subsets)
#     return float(total / len(subsets))


# @require_hypergraph_like
# def is_sperner(H: HypergraphLike) -> bool:
#     r"""
#     Return whether a hypergraph is Sperner.

#     A hypergraph is **Sperner** if no hyperedge properly contains another:

#     .. math::
#         e, f \in E(H),\ e \subseteq f \implies e = f.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.

#     Returns
#     -------
#     bool
#         True if the hypergraph is Sperner, otherwise False.

#     Notes
#     -----
#     Since the core hypergraph representation is simple, equality of hyperedges
#     need not be checked separately here.
#     """
#     edges = list(H.E)
#     for i in range(len(edges)):
#         for j in range(len(edges)):
#             if i == j:
#                 continue
#             if edges[i] < edges[j]:
#                 return False
#     return True


# @require_hypergraph_like
# def lower_shadow(H: HypergraphLike) -> Set[FrozenSet[Hashable]]:
#     r"""
#     Return the lower shadow of a hypergraph.

#     The **lower shadow** :math:`\partial H` is the family of all
#     ``(|e|-1)``-subsets obtained by deleting one vertex from each hyperedge
#     :math:`e \in E(H)`:

#     .. math::
#         \partial H = \{ e \setminus \{v\} : e \in E(H),\ v \in e \}.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.

#     Returns
#     -------
#     set of frozenset
#         The lower shadow of the hyperedge family.

#     Notes
#     -----
#     Hyperedges of size 0 contribute nothing to the lower shadow.
#     """
#     shadow: Set[FrozenSet[Hashable]] = set()
#     for edge in H.E:
#         for v in edge:
#             shadow.add(frozenset(edge.difference({v})))
#     return shadow


# @require_hypergraph_like
# def lower_shadow_size(H: HypergraphLike) -> int:
#     r"""
#     Return the size of the lower shadow of a hypergraph.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.

#     Returns
#     -------
#     int
#         Number of sets in the lower shadow.
#     """
#     return len(lower_shadow(H))


# @require_hypergraph_like
# def upper_shadow(
#     H: HypergraphLike,
#     *,
#     ground_set: Iterable[Hashable] | None = None,
# ) -> Set[FrozenSet[Hashable]]:
#     r"""
#     Return the upper shadow of a hypergraph.

#     Relative to a ground set :math:`X`, the **upper shadow** of a family
#     :math:`H` is the collection of sets obtained by adding one new element
#     from :math:`X` to a hyperedge:

#     .. math::
#         \partial^{+} H
#         = \{ e \cup \{v\} : e \in E(H),\ v \in X \setminus e \}.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     ground_set : iterable of hashable, optional
#         Ground set relative to which the upper shadow is formed. If omitted,
#         ``H.V`` is used.

#     Returns
#     -------
#     set of frozenset
#         The upper shadow of the hyperedge family relative to ``ground_set``.

#     Notes
#     -----
#     If ``ground_set`` is omitted, the upper shadow stays inside the current
#     vertex set of the hypergraph.
#     """
#     universe = set(H.V if ground_set is None else ground_set)
#     shadow: Set[FrozenSet[Hashable]] = set()

#     for edge in H.E:
#         for v in universe.difference(edge):
#             shadow.add(frozenset(set(edge) | {v}))

#     return shadow


# @require_hypergraph_like
# def upper_shadow_size(
#     H: HypergraphLike,
#     *,
#     ground_set: Iterable[Hashable] | None = None,
# ) -> int:
#     r"""
#     Return the size of the upper shadow of a hypergraph.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     ground_set : iterable of hashable, optional
#         Ground set relative to which the upper shadow is formed. If omitted,
#         ``H.V`` is used.

#     Returns
#     -------
#     int
#         Number of sets in the upper shadow.
#     """
#     return len(upper_shadow(H, ground_set=ground_set))

# @require_hypergraph_like
# @with_solver
# def fractional_matching_number(
#     H: HypergraphLike,
#     *,
#     verbose: bool = False,
#     solve=None,  # injected by @with_solver
# ) -> float:
#     r"""
#     Return the fractional matching number of a hypergraph.

#     A **fractional matching** of a hypergraph :math:`H=(V,E)` is a function
#     :math:`w : E \to [0,1]` such that for every vertex :math:`v \in V`,

#     .. math::
#         \sum_{e \ni v} w(e) \le 1.

#     The **fractional matching number** :math:`\nu^*(H)` is the optimum value of

#     .. math::
#         \max \sum_{e \in E} w(e)

#     subject to

#     .. math::
#         \sum_{e \ni v} w(e) \le 1 \quad \text{for all } v \in V,

#     and

#     .. math::
#         0 \le w(e) \le 1 \quad \text{for all } e \in E.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     verbose : bool, default=False
#         If True, print basic solver information.

#     Other Parameters
#     ----------------
#     solver : str or dict or pulp.LpSolver or type or callable or None, optional
#         Flexible solver specification handled by
#         :func:`graphcalc.solvers.resolve_solver`.
#     solver_options : dict, optional
#         Extra keyword arguments used when constructing the solver.

#     Returns
#     -------
#     float
#         The fractional matching number :math:`\nu^*(H)`.

#     Raises
#     ------
#     ValueError
#         If no optimal solution is found by the solver.

#     Notes
#     -----
#     If ``H`` has no hyperedges, the value is 0.0.

#     Examples
#     --------
#     >>> import graphcalc.graphs as gc
#     >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
#     >>> gc.fractional_matching_number(H)
#     1.0
#     """
#     if not H.E:
#         return 0.0

#     edges = list(H.E)

#     prob = pulp.LpProblem("FractionalMatchingNumberHypergraph", pulp.LpMaximize)
#     w = {
#         i: pulp.LpVariable(f"w_{i}", lowBound=0.0, upBound=1.0, cat="Continuous")
#         for i in range(len(edges))
#     }

#     prob += pulp.lpSum(w[i] for i in range(len(edges)))

#     for v in H.V:
#         incident = [i for i, edge in enumerate(edges) if v in edge]
#         if incident:
#             prob += pulp.lpSum(w[i] for i in incident) <= 1, f"vertex_{repr(v)}"

#     solve(prob)

#     value = pulp.value(prob.objective)
#     if verbose:
#         status = pulp.LpStatus.get(prob.status, str(prob.status))
#         print(f"Solver status : {status}")
#         print(f"Objective     : {value}")

#     return float(value if value is not None else 0.0)


# @require_hypergraph_like
# @with_solver
# def fractional_transversal_number(
#     H: HypergraphLike,
#     *,
#     verbose: bool = False,
#     solve=None,  # injected by @with_solver
# ) -> float:
#     r"""
#     Return the fractional transversal number of a hypergraph.

#     A **fractional transversal** of a hypergraph :math:`H=(V,E)` is a function
#     :math:`x : V \to [0,1]` such that for every hyperedge :math:`e \in E`,

#     .. math::
#         \sum_{v \in e} x(v) \ge 1.

#     The **fractional transversal number** :math:`\tau^*(H)` is the optimum value
#     of

#     .. math::
#         \min \sum_{v \in V} x(v)

#     subject to

#     .. math::
#         \sum_{v \in e} x(v) \ge 1 \quad \text{for all } e \in E,

#     and

#     .. math::
#         0 \le x(v) \le 1 \quad \text{for all } v \in V.

#     Parameters
#     ----------
#     H : HypergraphLike
#         A finite hypergraph.
#     verbose : bool, default=False
#         If True, print basic solver information.

#     Other Parameters
#     ----------------
#     solver : str or dict or pulp.LpSolver or type or callable or None, optional
#         Flexible solver specification handled by
#         :func:`graphcalc.solvers.resolve_solver`.
#     solver_options : dict, optional
#         Extra keyword arguments used when constructing the solver.

#     Returns
#     -------
#     float
#         The fractional transversal number :math:`\tau^*(H)`.

#     Raises
#     ------
#     ValueError
#         If ``H`` contains an empty hyperedge, in which case no fractional
#         transversal exists.
#     ValueError
#         If no optimal solution is found by the solver.

#     Notes
#     -----
#     If ``H`` has no hyperedges, the value is 0.0.

#     Examples
#     --------
#     >>> import graphcalc.graphs as gc
#     >>> H = gc.Hypergraph(E=[{1, 2}, {2, 3}])
#     >>> gc.fractional_transversal_number(H)
#     1.0
#     """
#     for edge in H.E:
#         if len(edge) == 0:
#             raise ValueError(
#                 "Hypergraph contains an empty hyperedge; no fractional transversal exists."
#             )

#     if not H.E:
#         return 0.0

#     vertices = list(H.V)

#     prob = pulp.LpProblem("FractionalTransversalNumberHypergraph", pulp.LpMinimize)
#     x = {
#         v: pulp.LpVariable(f"x_{i}", lowBound=0.0, upBound=1.0, cat="Continuous")
#         for i, v in enumerate(vertices)
#     }

#     prob += pulp.lpSum(x[v] for v in vertices)

#     for i, edge in enumerate(H.E):
#         prob += pulp.lpSum(x[v] for v in edge) >= 1, f"hit_edge_{i}"

#     solve(prob)

#     value = pulp.value(prob.objective)
#     if verbose:
#         status = pulp.LpStatus.get(prob.status, str(prob.status))
#         print(f"Solver status : {status}")
#         print(f"Objective     : {value}")

#     return float(value if value is not None else 0.0)
