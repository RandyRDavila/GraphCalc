from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
from graphcalc.additive_combinatorics.generators import (
    all_subsets_of_group,
    arithmetic_progression,
    coset,
    empty_set,
    interval,
    random_subset,
    random_symmetric_subset,
    sample_random_subsets,
    singleton,
    subgroup_from_generators,
    union_of_cosets,
    whole_group,
)
from graphcalc.additive_combinatorics.invariants import (
    additive_energy,
    cardinality,
    diffset_defect,
    diffset_density,
    diffset_size,
    difference_constant,
    doubling_constant,
    doubling_minus_difference,
    energy_over_sumset_square,
    max_difference_representation_count,
    max_sum_representation_count,
    normalized_additive_energy,
    stabilizer_index,
    stabilizer_size,
    stabilizer_size_of_sumset,
    sumset_defect,
    sumset_density,
    sumset_size,
    sumset_stabilizer_index,
    tripling_constant,
)
from graphcalc.additive_combinatorics.properties import (
    contains_zero,
    has_full_diffset,
    has_full_sumset,
    has_small_doubling,
    is_aperiodic,
    is_coset,
    is_difference_larger_than_sumset,
    is_doubling_at_most_2,
    is_doubling_at_most_3_over_2,
    is_empty_set,
    is_periodic,
    is_sidon,
    is_subgroup,
    is_sum_free,
    is_symmetric,
    is_trivial_set,
    is_whole_group,
    sumset_is_periodic,
)
from graphcalc.additive_combinatorics.sets import AdditiveSet

__all__ = [
    "additive_dataset_column_definitions",
    "additive_set_to_record",
    "generate_additive_set_dataset",
    "small_additive_snapshot",
    "medium_additive_snapshot",
    "large_additive_snapshot",
]


def additive_dataset_column_definitions() -> dict[str, dict]:
    r"""
    Return the schema for additive-combinatorics conjecturing datasets.

    Returns
    -------
    dict of str to dict
        Dictionary keyed by column name. Each value is a metadata record
        describing the column type and intended meaning.

    Notes
    -----
    The schema is designed for conjecturing workflows in which each row
    describes one finite additive set together with ambient-group descriptors,
    numerical invariants, Boolean predicates, and optional provenance
    information.

    The returned dictionary is intended to be stable enough for downstream
    export helpers and dataframe construction.
    """
    return {
        "ambient_moduli": {
            "type": "object",
            "description": "Tuple of cyclic moduli defining the ambient finite abelian group.",
        },
        "ambient_rank": {
            "type": "int",
            "description": "Number of cyclic factors in the ambient group.",
        },
        "ambient_order": {
            "type": "int",
            "description": "Cardinality of the ambient finite abelian group.",
        },
        "elements": {
            "type": "object",
            "description": "Canonical tuple of elements of the additive set.",
        },
        "cardinality": {
            "type": "int",
            "description": "Cardinality |A| of the additive set.",
        },
        "sumset_size": {
            "type": "int",
            "description": "Cardinality |A+A| of the self-sumset.",
        },
        "diffset_size": {
            "type": "int",
            "description": "Cardinality |A-A| of the self-difference set.",
        },
        "doubling_constant": {
            "type": "float",
            "description": "Ratio |A+A| / |A| for nonempty sets, else None.",
        },
        "difference_constant": {
            "type": "float",
            "description": "Ratio |A-A| / |A| for nonempty sets, else None.",
        },
        "tripling_constant": {
            "type": "float",
            "description": "Ratio |A+A+A| / |A| for nonempty sets, else None.",
        },
        "sumset_defect": {
            "type": "int",
            "description": "Difference |A+A| - |A|.",
        },
        "diffset_defect": {
            "type": "int",
            "description": "Difference |A-A| - |A|.",
        },
        "doubling_minus_difference": {
            "type": "int",
            "description": "Difference |A+A| - |A-A|.",
        },
        "sumset_density": {
            "type": "float",
            "description": "Relative sumset size |A+A| / |G|.",
        },
        "diffset_density": {
            "type": "float",
            "description": "Relative difference-set size |A-A| / |G|.",
        },
        "additive_energy": {
            "type": "int",
            "description": "Additive energy E(A).",
        },
        "normalized_additive_energy": {
            "type": "float",
            "description": "Ratio E(A) / |A|^3 for nonempty sets, else None.",
        },
        "energy_over_sumset_square": {
            "type": "float",
            "description": "Ratio E(A) / |A+A|^2 when |A+A| > 0, else None.",
        },
        "max_sum_representation_count": {
            "type": "int",
            "description": "Maximum ordered representation count in A+A.",
        },
        "max_difference_representation_count": {
            "type": "int",
            "description": "Maximum ordered representation count in A-A.",
        },
        "stabilizer_size": {
            "type": "int",
            "description": "Cardinality of Stab(A).",
        },
        "stabilizer_size_of_sumset": {
            "type": "int",
            "description": "Cardinality of Stab(A+A).",
        },
        "stabilizer_index": {
            "type": "int",
            "description": "Index |G| / |Stab(A)|.",
        },
        "sumset_stabilizer_index": {
            "type": "int",
            "description": "Index |G| / |Stab(A+A)|.",
        },
        "contains_zero": {
            "type": "bool",
            "description": "Whether the ambient identity belongs to A.",
        },
        "is_empty_set": {
            "type": "bool",
            "description": "Whether A is empty.",
        },
        "is_trivial_set": {
            "type": "bool",
            "description": "Whether |A| <= 1.",
        },
        "is_whole_group": {
            "type": "bool",
            "description": "Whether A equals the whole ambient group.",
        },
        "is_symmetric": {
            "type": "bool",
            "description": "Whether A = -A.",
        },
        "is_subgroup": {
            "type": "bool",
            "description": "Whether A is a subgroup of the ambient group.",
        },
        "is_coset": {
            "type": "bool",
            "description": "Whether A is a coset of a subgroup.",
        },
        "is_sum_free": {
            "type": "bool",
            "description": "Whether (A+A) ∩ A is empty.",
        },
        "is_sidon": {
            "type": "bool",
            "description": "Whether A is a Sidon set.",
        },
        "has_small_doubling_3_over_2": {
            "type": "bool",
            "description": "Whether |A+A| <= (3/2)|A|.",
        },
        "has_small_doubling_2": {
            "type": "bool",
            "description": "Whether |A+A| <= 2|A|.",
        },
        "has_full_sumset": {
            "type": "bool",
            "description": "Whether A+A equals the ambient group.",
        },
        "has_full_diffset": {
            "type": "bool",
            "description": "Whether A-A equals the ambient group.",
        },
        "is_difference_larger_than_sumset": {
            "type": "bool",
            "description": "Whether |A-A| > |A+A|.",
        },
        "is_periodic": {
            "type": "bool",
            "description": "Whether A has nontrivial stabilizer.",
        },
        "is_aperiodic": {
            "type": "bool",
            "description": "Whether A has trivial stabilizer.",
        },
        "sumset_is_periodic": {
            "type": "bool",
            "description": "Whether A+A has nontrivial stabilizer.",
        },
        "construction": {
            "type": "object",
            "description": "Optional label describing how the set was generated.",
        },
        "construction_parameters": {
            "type": "object",
            "description": "Optional parameter dictionary describing the construction.",
        },
    }


def additive_set_to_record(
    A: AdditiveSet,
    *,
    construction: str | None = None,
    construction_parameters: dict | None = None,
) -> dict:
    r"""
    Convert an additive set into a conjecturing-oriented record.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.
    construction : str, optional
        Optional label describing how the set was generated.
    construction_parameters : dict, optional
        Optional parameter dictionary recording the construction choices.

    Returns
    -------
    dict
        Dictionary containing ambient descriptors, numerical invariants,
        Boolean predicates, and optional provenance metadata.

    Notes
    -----
    Quantities that are undefined on the empty set, such as
    :math:`|A+A|/|A|`, are stored as ``None`` instead of raising an exception.
    This makes the resulting records easier to place into a pandas dataframe.
    """
    nonempty = not A.is_empty
    sumset_sz = A.sumset().size

    return {
        "ambient_moduli": A.group.moduli,
        "ambient_rank": A.group.rank,
        "ambient_order": A.group.order,
        "elements": A.elements,
        "cardinality": cardinality(A),
        "sumset_size": sumset_size(A),
        "diffset_size": diffset_size(A),
        "doubling_constant": doubling_constant(A) if nonempty else None,
        "difference_constant": difference_constant(A) if nonempty else None,
        "tripling_constant": tripling_constant(A) if nonempty else None,
        "sumset_defect": sumset_defect(A),
        "diffset_defect": diffset_defect(A),
        "doubling_minus_difference": doubling_minus_difference(A),
        "sumset_density": sumset_density(A),
        "diffset_density": diffset_density(A),
        "additive_energy": additive_energy(A),
        "normalized_additive_energy": normalized_additive_energy(A) if nonempty else None,
        "energy_over_sumset_square": energy_over_sumset_square(A) if sumset_sz > 0 else None,
        "max_sum_representation_count": max_sum_representation_count(A),
        "max_difference_representation_count": max_difference_representation_count(A),
        "stabilizer_size": stabilizer_size(A),
        "stabilizer_size_of_sumset": stabilizer_size_of_sumset(A),
        "stabilizer_index": stabilizer_index(A),
        "sumset_stabilizer_index": sumset_stabilizer_index(A),
        "contains_zero": contains_zero(A),
        "is_empty_set": is_empty_set(A),
        "is_trivial_set": is_trivial_set(A),
        "is_whole_group": is_whole_group(A),
        "is_symmetric": is_symmetric(A),
        "is_subgroup": is_subgroup(A),
        "is_coset": is_coset(A),
        "is_sum_free": is_sum_free(A),
        "is_sidon": is_sidon(A),
        "has_small_doubling_3_over_2": is_doubling_at_most_3_over_2(A),
        "has_small_doubling_2": is_doubling_at_most_2(A),
        "has_full_sumset": has_full_sumset(A),
        "has_full_diffset": has_full_diffset(A),
        "is_difference_larger_than_sumset": is_difference_larger_than_sumset(A),
        "is_periodic": is_periodic(A),
        "is_aperiodic": is_aperiodic(A),
        "sumset_is_periodic": sumset_is_periodic(A),
        "construction": construction,
        "construction_parameters": construction_parameters,
    }


def generate_additive_set_dataset(
    instances: Iterable[
        AdditiveSet
        | tuple[AdditiveSet, str]
        | tuple[AdditiveSet, str, dict]
    ],
) -> list[dict]:
    r"""
    Convert a collection of additive sets into dataset rows.

    Parameters
    ----------
    instances : iterable
        Iterable whose entries are one of:

        - an :class:`AdditiveSet`,
        - a pair ``(A, construction)``, or
        - a triple ``(A, construction, construction_parameters)``.

    Returns
    -------
    list of dict
        Dataset rows obtained by applying :func:`additive_set_to_record`.

    Raises
    ------
    TypeError
        If an entry is not in one of the accepted formats.
    """
    rows: list[dict] = []

    for item in instances:
        if isinstance(item, AdditiveSet):
            rows.append(additive_set_to_record(item))
        elif isinstance(item, tuple) and len(item) == 2:
            A, construction = item
            if not isinstance(A, AdditiveSet):
                raise TypeError("Dataset tuple entries must begin with an AdditiveSet.")
            rows.append(additive_set_to_record(A, construction=construction))
        elif isinstance(item, tuple) and len(item) == 3:
            A, construction, construction_parameters = item
            if not isinstance(A, AdditiveSet):
                raise TypeError("Dataset tuple entries must begin with an AdditiveSet.")
            rows.append(
                additive_set_to_record(
                    A,
                    construction=construction,
                    construction_parameters=construction_parameters,
                )
            )
        else:
            raise TypeError(
                "Each dataset entry must be an AdditiveSet, "
                "(AdditiveSet, construction), or "
                "(AdditiveSet, construction, construction_parameters)."
            )

    return rows


def _small_snapshot_instances() -> list[tuple[AdditiveSet, str, dict]]:
    r"""
    Build a small deterministic collection of additive-set examples.

    Returns
    -------
    list of tuple
        Triples ``(A, construction, construction_parameters)`` suitable for
        :func:`generate_additive_set_dataset`.

    Notes
    -----
    This helper is internal. It keeps the public snapshot function concise while
    making the construction logic easier to expand and test.
    """
    instances: list[tuple[AdditiveSet, str, dict]] = []

    for n in (4, 5, 6, 7, 8):
        G = FiniteAbelianGroup((n,))

        instances.append((empty_set(G), "empty_set", {"moduli": (n,)}))
        instances.append((whole_group(G), "whole_group", {"moduli": (n,)}))
        instances.append((singleton(G, G.zero()), "singleton", {"moduli": (n,), "element": G.zero()}))

        instances.append(
            (
                interval(n, min(3, n), start=0),
                "interval",
                {"modulus": n, "length": min(3, n), "start": 0},
            )
        )

        instances.append(
            (
                arithmetic_progression(G, start=(0,), step=(1,), length=min(4, n)),
                "arithmetic_progression",
                {"moduli": (n,), "start": (0,), "step": (1,), "length": min(4, n)},
            )
        )

        trivial_subgroup = subgroup_from_generators(G, [(0,)])
        instances.append(
            (
                trivial_subgroup,
                "subgroup_from_generators",
                {"moduli": (n,), "generators": [(0,)]},
            )
        )

        if n % 2 == 0:
            even_subgroup = subgroup_from_generators(G, [(2,)])
            instances.append(
                (
                    even_subgroup,
                    "subgroup_from_generators",
                    {"moduli": (n,), "generators": [(2,)]},
                )
            )
            instances.append(
                (
                    coset(even_subgroup, (1,)),
                    "coset",
                    {"moduli": (n,), "base_generators": [(2,)], "translate": (1,)},
                )
            )
            instances.append(
                (
                    union_of_cosets(even_subgroup, [(0,), (1,)]),
                    "union_of_cosets",
                    {"moduli": (n,), "base_generators": [(2,)], "translates": [(0,), (1,)]},
                )
            )

        rng_random = np.random.default_rng(1000 + n)
        instances.append(
            (
                random_subset(G, size=min(3, n), rng=rng_random),
                "random_subset_size",
                {"moduli": (n,), "size": min(3, n), "seed": 1000 + n},
            )
        )

        rng_sym = np.random.default_rng(6000 + n)
        instances.append(
            (
                random_symmetric_subset(G, density=0.5, rng=rng_sym),
                "random_symmetric_subset",
                {"moduli": (n,), "density": 0.5, "seed": 6000 + n},
            )
        )

        if G.order <= 5:
            for idx, subset in enumerate(all_subsets_of_group(G)):
                instances.append(
                    (
                        subset,
                        "all_subsets_of_group",
                        {"moduli": (n,), "index": idx},
                    )
                )

    return instances


def small_additive_snapshot() -> list[dict]:
    r"""
    Return a small deterministic additive-combinatorics dataset snapshot.

    Returns
    -------
    list of dict
        Reproducible dataset rows spanning several structured and random
        families in small cyclic groups.

    Notes
    -----
    This snapshot is intended for quick experimentation, smoke tests, and
    documentation examples.
    """
    return generate_additive_set_dataset(_small_snapshot_instances())


def _medium_snapshot_instances() -> list[tuple[AdditiveSet, str, dict]]:
    r"""
    Build a medium deterministic collection of additive-set examples.

    Returns
    -------
    list of tuple
        Triples ``(A, construction, construction_parameters)`` suitable for
        :func:`generate_additive_set_dataset`.
    """
    instances: list[tuple[AdditiveSet, str, dict]] = []

    cyclic_moduli = (5, 6, 7, 8, 9, 10)
    for n in cyclic_moduli:
        G = FiniteAbelianGroup((n,))

        instances.append((empty_set(G), "empty_set", {"moduli": (n,)}))
        instances.append((whole_group(G), "whole_group", {"moduli": (n,)}))
        instances.append((singleton(G, G.zero()), "singleton", {"moduli": (n,), "element": G.zero()}))
        instances.append((singleton(G, (1,)), "singleton", {"moduli": (n,), "element": (1,)}))

        for start in (0, 1):
            for length in (0, 1, min(3, n), min(4, n)):
                instances.append(
                    (
                        interval(n, length, start=start),
                        "interval",
                        {"modulus": n, "length": length, "start": start},
                    )
                )

        for step in (1, 2):
            instances.append(
                (
                    arithmetic_progression(G, start=(0,), step=(step,), length=min(4, n)),
                    "arithmetic_progression",
                    {"moduli": (n,), "start": (0,), "step": (step,), "length": min(4, n)},
                )
            )

        for gen in ((0,), (1,), (2,)):
            H = subgroup_from_generators(G, [gen])
            instances.append(
                (
                    H,
                    "subgroup_from_generators",
                    {"moduli": (n,), "generators": [gen]},
                )
            )
            instances.append(
                (
                    coset(H, (1,)),
                    "coset",
                    {"moduli": (n,), "base_generators": [gen], "translate": (1,)},
                )
            )

        if n % 2 == 0:
            H = subgroup_from_generators(G, [(2,)])
            instances.append(
                (
                    union_of_cosets(H, [(0,), (1,)]),
                    "union_of_cosets",
                    {"moduli": (n,), "base_generators": [(2,)], "translates": [(0,), (1,)]},
                )
            )

        rng_size = np.random.default_rng(2000 + n)
        rng_density = np.random.default_rng(3000 + n)
        rng_sym = np.random.default_rng(3500 + n)

        instances.append(
            (
                random_subset(G, size=min(4, n), rng=rng_size),
                "random_subset_size",
                {"moduli": (n,), "size": min(4, n), "seed": 2000 + n},
            )
        )
        instances.append(
            (
                random_subset(G, density=0.4, rng=rng_density),
                "random_subset_density",
                {"moduli": (n,), "density": 0.4, "seed": 3000 + n},
            )
        )
        instances.append(
            (
                random_symmetric_subset(G, density=0.5, rng=rng_sym),
                "random_symmetric_subset",
                {"moduli": (n,), "density": 0.5, "seed": 3500 + n},
            )
        )

    for moduli in ((2, 2), (2, 3), (3, 3), (2, 2, 2)):
        G = FiniteAbelianGroup(moduli)

        instances.append((empty_set(G), "empty_set", {"moduli": moduli}))
        instances.append((whole_group(G), "whole_group", {"moduli": moduli}))
        instances.append((singleton(G, G.zero()), "singleton", {"moduli": moduli, "element": G.zero()}))

        instances.append(
            (
                subgroup_from_generators(G, []),
                "subgroup_from_generators",
                {"moduli": moduli, "generators": []},
            )
        )

        step = tuple(1 for _ in moduli)
        instances.append(
            (
                arithmetic_progression(G, start=G.zero(), step=step, length=3),
                "arithmetic_progression",
                {"moduli": moduli, "start": G.zero(), "step": step, "length": 3},
            )
        )

        gens = [tuple(1 if i == j else 0 for i in range(len(moduli))) for j in range(len(moduli))]
        H = subgroup_from_generators(G, gens[:1])
        instances.append(
            (
                H,
                "subgroup_from_generators",
                {"moduli": moduli, "generators": gens[:1]},
            )
        )
        instances.append(
            (
                coset(H, tuple(1 for _ in moduli)),
                "coset",
                {"moduli": moduli, "base_generators": gens[:1], "translate": tuple(1 for _ in moduli)},
            )
        )
        instances.append(
            (
                union_of_cosets(H, [G.zero(), tuple(1 for _ in moduli)]),
                "union_of_cosets",
                {"moduli": moduli, "base_generators": gens[:1], "translates": [G.zero(), tuple(1 for _ in moduli)]},
            )
        )

        rng_size = np.random.default_rng(4000 + sum(moduli))
        rng_density = np.random.default_rng(5000 + sum(moduli))
        rng_sym = np.random.default_rng(5500 + sum(moduli))

        instances.append(
            (
                random_subset(G, size=min(4, G.order), rng=rng_size),
                "random_subset_size",
                {"moduli": moduli, "size": min(4, G.order), "seed": 4000 + sum(moduli)},
            )
        )
        instances.append(
            (
                random_subset(G, density=0.5, rng=rng_density),
                "random_subset_density",
                {"moduli": moduli, "density": 0.5, "seed": 5000 + sum(moduli)},
            )
        )
        instances.append(
            (
                random_symmetric_subset(G, density=0.5, rng=rng_sym),
                "random_symmetric_subset",
                {"moduli": moduli, "density": 0.5, "seed": 5500 + sum(moduli)},
            )
        )

    return instances


def medium_additive_snapshot() -> list[dict]:
    r"""
    Return a medium deterministic additive-combinatorics dataset snapshot.

    Returns
    -------
    list of dict
        Reproducible dataset rows spanning cyclic and product groups together
        with structured and seeded random families.

    Notes
    -----
    This snapshot is intended for richer conjecturing experiments while
    remaining small enough for routine unit testing and documentation use.
    """
    return generate_additive_set_dataset(_medium_snapshot_instances())


def large_additive_snapshot(
    *,
    cyclic_moduli: Sequence[int] = tuple(range(4, 21)),
    product_moduli: Sequence[tuple[int, ...]] = ((2, 2), (2, 3), (3, 3), (2, 2, 2)),
    random_samples_per_group: int = 25,
) -> list[dict]:
    r"""
    Return a larger deterministic additive-combinatorics dataset snapshot.

    Parameters
    ----------
    cyclic_moduli : sequence of int, default=range(4, 21)
        Cyclic moduli to include as ambient groups :math:`\mathbb{Z}/n\mathbb{Z}`.
    product_moduli : sequence of tuple of int, optional
        Product-group ambient moduli to include.
    random_samples_per_group : int, default=25
        Number of fixed-size and density-based random samples to generate per
        ambient group.

    Returns
    -------
    list of dict
        Deterministic dataset rows with many examples and descriptors.

    Raises
    ------
    ValueError
        If ``random_samples_per_group`` is negative.

    Notes
    -----
    This snapshot is designed for conjecturing workflows that benefit from many
    examples. All randomness is seeded deterministically from the ambient-group
    parameters.
    """
    if random_samples_per_group < 0:
        raise ValueError("random_samples_per_group must be nonnegative.")

    instances: list[tuple[AdditiveSet, str, dict]] = []
    instances.extend(_medium_snapshot_instances())

    for n in cyclic_moduli:
        G = FiniteAbelianGroup((n,))

        if G.order <= 6:
            for idx, subset in enumerate(all_subsets_of_group(G, max_order=6)):
                instances.append(
                    (
                        subset,
                        "all_subsets_of_group",
                        {"moduli": (n,), "index": idx},
                    )
                )

        for step in (1, 2, 3):
            instances.append(
                (
                    arithmetic_progression(G, start=(0,), step=(step,), length=min(5, n)),
                    "arithmetic_progression",
                    {"moduli": (n,), "start": (0,), "step": (step,), "length": min(5, n)},
                )
            )

        H = subgroup_from_generators(G, [(2,)])
        instances.append(
            (
                H,
                "subgroup_from_generators",
                {"moduli": (n,), "generators": [(2,)]},
            )
        )
        instances.append(
            (
                coset(H, (1,)),
                "coset",
                {"moduli": (n,), "base_generators": [(2,)], "translate": (1,)},
            )
        )
        instances.append(
            (
                union_of_cosets(H, [(0,), (1,)]),
                "union_of_cosets",
                {"moduli": (n,), "base_generators": [(2,)], "translates": [(0,), (1,)]},
            )
        )

        random_by_size = sample_random_subsets(
            G,
            num_samples=random_samples_per_group,
            size=min(4, n),
            seed=7000 + n,
        )
        for idx, A in enumerate(random_by_size):
            instances.append(
                (
                    A,
                    "sample_random_subsets_size",
                    {"moduli": (n,), "size": min(4, n), "seed": 7000 + n, "index": idx},
                )
            )

        random_by_density = sample_random_subsets(
            G,
            num_samples=random_samples_per_group,
            density=0.4,
            seed=8000 + n,
        )
        for idx, A in enumerate(random_by_density):
            instances.append(
                (
                    A,
                    "sample_random_subsets_density",
                    {"moduli": (n,), "density": 0.4, "seed": 8000 + n, "index": idx},
                )
            )

        for idx in range(random_samples_per_group):
            rng_sym = np.random.default_rng(9000 + 100 * n + idx)
            A = random_symmetric_subset(G, density=0.5, rng=rng_sym)
            instances.append(
                (
                    A,
                    "random_symmetric_subset",
                    {"moduli": (n,), "density": 0.5, "seed": 9000 + 100 * n + idx},
                )
            )

    for moduli in product_moduli:
        G = FiniteAbelianGroup(moduli)

        if G.order <= 6:
            for idx, subset in enumerate(all_subsets_of_group(G, max_order=6)):
                instances.append(
                    (
                        subset,
                        "all_subsets_of_group",
                        {"moduli": moduli, "index": idx},
                    )
                )

        basis = [tuple(1 if i == j else 0 for i in range(len(moduli))) for j in range(len(moduli))]
        for gens in ([], basis[:1], basis[:2]):
            H = subgroup_from_generators(G, gens)
            instances.append(
                (
                    H,
                    "subgroup_from_generators",
                    {"moduli": moduli, "generators": gens},
                )
            )
            instances.append(
                (
                    coset(H, tuple(1 for _ in moduli)),
                    "coset",
                    {"moduli": moduli, "base_generators": gens, "translate": tuple(1 for _ in moduli)},
                )
            )

        random_by_size = sample_random_subsets(
            G,
            num_samples=random_samples_per_group,
            size=min(4, G.order),
            seed=10000 + sum(moduli),
        )
        for idx, A in enumerate(random_by_size):
            instances.append(
                (
                    A,
                    "sample_random_subsets_size",
                    {"moduli": moduli, "size": min(4, G.order), "seed": 10000 + sum(moduli), "index": idx},
                )
            )

        random_by_density = sample_random_subsets(
            G,
            num_samples=random_samples_per_group,
            density=0.5,
            seed=11000 + sum(moduli),
        )
        for idx, A in enumerate(random_by_density):
            instances.append(
                (
                    A,
                    "sample_random_subsets_density",
                    {"moduli": moduli, "density": 0.5, "seed": 11000 + sum(moduli), "index": idx},
                )
            )

        for idx in range(random_samples_per_group):
            rng_sym = np.random.default_rng(12000 + 100 * sum(moduli) + idx)
            A = random_symmetric_subset(G, density=0.5, rng=rng_sym)
            instances.append(
                (
                    A,
                    "random_symmetric_subset",
                    {"moduli": moduli, "density": 0.5, "seed": 12000 + 100 * sum(moduli) + idx},
                )
            )

    return generate_additive_set_dataset(instances)
