from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from graphcalc.additive_combinatorics.ambient_groups import Element, FiniteAbelianGroup
from graphcalc.additive_combinatorics.sets import AdditiveSet

__all__ = [
    "empty_set",
    "singleton",
    "interval",
    "arithmetic_progression",
    "random_subset",
    "subgroup_from_generators",
    "coset",
    "whole_group",
    "union_of_cosets",
    "random_symmetric_subset",
    "all_subsets_of_group",
    "sample_random_subsets",
]


def empty_set(group: FiniteAbelianGroup) -> AdditiveSet:
    """
    Return the empty subset of a finite abelian group.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.

    Returns
    -------
    AdditiveSet
        The empty subset of ``group``.
    """
    return AdditiveSet([], group=group)


def singleton(group: FiniteAbelianGroup, x: Sequence[int]) -> AdditiveSet:
    """
    Return the singleton subset containing one ambient group element.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.
    x : sequence of int
        Element of the ambient group.

    Returns
    -------
    AdditiveSet
        The singleton set ``{x}``, with ``x`` reduced canonically modulo the
        ambient group moduli.

    Raises
    ------
    ValueError
        If ``x`` does not have the correct coordinate length.
    """
    return AdditiveSet([x], group=group)


def interval(modulus: int, length: int, *, start: int = 0) -> AdditiveSet:
    r"""
    Return an interval in the cyclic group :math:`\mathbb{Z}/n\mathbb{Z}`.

    Parameters
    ----------
    modulus : int
        Modulus ``n`` defining the cyclic group :math:`\mathbb{Z}/n\mathbb{Z}`.
    length : int
        Number of consecutive residues to include.
    start : int, default=0
        Initial residue class.

    Returns
    -------
    AdditiveSet
        The subset

        .. math::

            \{start, start+1, \dots, start+length-1\} \pmod{n}

        viewed as a subset of :math:`\mathbb{Z}/n\mathbb{Z}`.

    Raises
    ------
    ValueError
        If ``modulus`` is not positive, or if ``length`` is negative.

    Notes
    -----
    If ``length`` exceeds ``modulus``, the resulting subset is the whole cyclic
    group, since duplicate residues are removed canonically.
    """
    if modulus <= 0:
        raise ValueError("modulus must be positive.")
    if length < 0:
        raise ValueError("length must be nonnegative.")

    G = FiniteAbelianGroup((modulus,))
    elems = [((start + i),) for i in range(length)]
    return AdditiveSet(elems, group=G)


def arithmetic_progression(
    group: FiniteAbelianGroup,
    start: Sequence[int],
    step: Sequence[int],
    length: int,
) -> AdditiveSet:
    r"""
    Return a finite arithmetic progression in an ambient finite abelian group.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.
    start : sequence of int
        Initial element of the progression.
    step : sequence of int
        Common difference.
    length : int
        Number of terms.

    Returns
    -------
    AdditiveSet
        The set

        .. math::

            \{start + j \cdot step : 0 \le j < length\}

        in the ambient group.

    Raises
    ------
    ValueError
        If ``length`` is negative, or if ``start`` or ``step`` does not have
        the correct coordinate length.

    Notes
    -----
    In torsion groups, different progression terms may coincide, so the
    resulting set may have cardinality smaller than ``length``.
    """
    if length < 0:
        raise ValueError("length must be nonnegative.")

    elems = [group.add(start, group.scalar_mul(j, step)) for j in range(length)]
    return AdditiveSet(elems, group=group)


def random_subset(
    group: FiniteAbelianGroup,
    *,
    size: int | None = None,
    density: float | None = None,
    rng=None,
) -> AdditiveSet:
    """
    Return a random subset of a finite abelian group.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.
    size : int, optional
        Desired subset size. If provided, a subset of exactly this cardinality
        is sampled uniformly from all subsets of that size.
    density : float, optional
        Bernoulli inclusion probability for each ambient group element. If
        provided, each element is included independently with this probability.
    rng : numpy.random.Generator, optional
        Random number generator to use. If omitted, a new default generator is
        created.

    Returns
    -------
    AdditiveSet
        A random subset of the ambient group.

    Raises
    ------
    ValueError
        If neither or both of ``size`` and ``density`` are provided.
    ValueError
        If ``size`` is negative or exceeds the ambient group order.
    ValueError
        If ``density`` does not lie in the interval ``[0, 1]``.

    Notes
    -----
    Exactly one of ``size`` and ``density`` must be specified.
    """
    if (size is None) == (density is None):
        raise ValueError("Exactly one of size and density must be specified.")

    if rng is None:
        rng = np.random.default_rng()

    elems = group.list_elements()

    if size is not None:
        if size < 0 or size > group.order:
            raise ValueError("size must satisfy 0 <= size <= |G|.")
        if size == 0:
            return AdditiveSet([], group=group)

        indices = rng.choice(len(elems), size=size, replace=False)
        chosen = [elems[int(i)] for i in indices]
        return AdditiveSet(chosen, group=group)

    assert density is not None
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must satisfy 0 <= density <= 1.")

    chosen = [x for x in elems if rng.random() < density]
    return AdditiveSet(chosen, group=group)


def subgroup_from_generators(
    group: FiniteAbelianGroup,
    generators: Iterable[Sequence[int]],
) -> AdditiveSet:
    r"""
    Return the subgroup generated by a family of elements.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.
    generators : iterable of sequence of int
        Generating family.

    Returns
    -------
    AdditiveSet
        The smallest subgroup of the ambient group containing all of the given
        generators.

    Notes
    -----
    Since the ambient group is finite, the subgroup is computed by iterative
    closure under addition and negation starting from the generators and the
    identity element.

    The empty generating family yields the trivial subgroup ``{0}``.
    """
    current = {group.zero()}
    frontier = {group.normalize(g) for g in generators}

    current |= frontier

    changed = True
    while changed:
        changed = False
        new_elems = set(current)

        for a in current:
            new_elems.add(group.neg(a))

        current_list = list(current)
        for a in current_list:
            for b in current_list:
                new_elems.add(group.add(a, b))

        if new_elems != current:
            current = new_elems
            changed = True

    return AdditiveSet(current, group=group)


def coset(subgroup: AdditiveSet, x: Sequence[int]) -> AdditiveSet:
    r"""
    Return a translate of a subgroup, viewed as a coset.

    Parameters
    ----------
    subgroup : AdditiveSet
        A subgroup of some finite abelian ambient group.
    x : sequence of int
        Translation element.

    Returns
    -------
    AdditiveSet
        The translate

        .. math::

            x + H = \{x + h : h \in H\}

        in the same ambient group as ``subgroup``.

    Raises
    ------
    ValueError
        If ``x`` does not have the correct coordinate length.

    Notes
    -----
    This function does not itself verify that ``subgroup`` is actually a
    subgroup; it simply returns the translate of the given additive set.
    """
    return subgroup.translate(x)


def whole_group(group: FiniteAbelianGroup) -> AdditiveSet:
    """
    Return the whole ambient group as an additive set.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.

    Returns
    -------
    AdditiveSet
        The full set of all elements of the ambient group.
    """
    return AdditiveSet(group.elements(), group=group)


def union_of_cosets(subgroup: AdditiveSet, translates: Iterable[Sequence[int]]) -> AdditiveSet:
    r"""
    Return a union of translates of a given additive set.

    Parameters
    ----------
    subgroup : AdditiveSet
        Base additive set, typically a subgroup.
    translates : iterable of sequence of int
        Translation elements.

    Returns
    -------
    AdditiveSet
        The union of the translates :math:`t + H` over the given translation
        elements :math:`t`.

    Notes
    -----
    This function does not verify that ``subgroup`` is actually a subgroup.
    """
    elems = set()
    for t in translates:
        elems.update(subgroup.translate(t).elements)
    return AdditiveSet(elems, group=subgroup.group)


def random_symmetric_subset(
    group: FiniteAbelianGroup,
    *,
    density: float = 0.5,
    include_zero: bool = True,
    rng=None,
) -> AdditiveSet:
    """
    Return a random symmetric subset of a finite abelian group.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.
    density : float, default=0.5
        Sampling density applied to inversion orbits.
    include_zero : bool, default=True
        Whether to force inclusion of the identity element.
    rng : numpy.random.Generator, optional
        Random number generator to use.

    Returns
    -------
    AdditiveSet
        A random subset A satisfying A = -A.

    Raises
    ------
    ValueError
        If density is not in the interval [0, 1].
    """
    import numpy as np

    if not (0.0 <= density <= 1.0):
        raise ValueError("density must satisfy 0 <= density <= 1.")

    if rng is None:
        rng = np.random.default_rng()

    chosen = set()
    seen = set()

    for x in group.elements():
        if x in seen:
            continue
        nx = group.neg(x)
        orbit = {x, nx}
        seen.update(orbit)

        if x == group.zero():
            if include_zero or rng.random() < density:
                chosen.add(x)
        else:
            if rng.random() < density:
                chosen.update(orbit)

    return AdditiveSet(chosen, group=group)


def all_subsets_of_group(group: FiniteAbelianGroup, *, max_order: int = 10) -> list[AdditiveSet]:
    """
    Return all subsets of a small finite abelian group.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.
    max_order : int, default=10
        Maximum ambient group order for which exhaustive subset generation is
        allowed.

    Returns
    -------
    list of AdditiveSet
        All subsets of the ambient group.

    Raises
    ------
    ValueError
        If the ambient group order exceeds ``max_order``.
    """
    if group.order > max_order:
        raise ValueError("Exhaustive subset generation is only allowed for very small groups.")

    elems = list(group.elements())
    out: list[AdditiveSet] = []
    for mask in range(1 << len(elems)):
        subset = [elems[i] for i in range(len(elems)) if (mask >> i) & 1]
        out.append(AdditiveSet(subset, group=group))
    return out


def sample_random_subsets(
    group: FiniteAbelianGroup,
    *,
    num_samples: int,
    size: int | None = None,
    density: float | None = None,
    seed: int | None = None,
) -> list[AdditiveSet]:
    """
    Sample multiple random subsets of an ambient finite abelian group.

    Parameters
    ----------
    group : FiniteAbelianGroup
        Ambient finite abelian group.
    num_samples : int
        Number of samples to draw.
    size : int, optional
        Fixed size of each sample.
    density : float, optional
        Bernoulli density of each sample.
    seed : int, optional
        Seed for reproducible sampling.

    Returns
    -------
    list of AdditiveSet
        List of random samples.

    Raises
    ------
    ValueError
        If ``num_samples`` is negative.
    """
    import numpy as np

    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative.")

    rng = np.random.default_rng(seed)
    return [
        random_subset(group, size=size, density=density, rng=rng)
        for _ in range(num_samples)
    ]
