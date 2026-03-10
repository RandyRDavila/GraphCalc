from __future__ import annotations

from fractions import Fraction

from graphcalc.metadata import invariant_metadata
from graphcalc.additive_combinatorics.sets import AdditiveSet

__all__ = [
    "cardinality",
    "sumset_size",
    "diffset_size",
    "doubling_constant",
    "difference_constant",
    "tripling_constant",
    "additive_energy",
    "max_sum_representation_count",
    "max_difference_representation_count",
    "stabilizer_size",
    "stabilizer_size_of_sumset",
    "sumset_defect",
    "diffset_defect",
    "doubling_minus_difference",
    "sumset_density",
    "diffset_density",
    "stabilizer_index",
    "sumset_stabilizer_index",
    "normalized_additive_energy",
    "energy_over_sumset_square",
]


@invariant_metadata(
    display_name="Cardinality",
    notation=r"|A|",
    category="additive combinatorics invariants",
    aliases=("set size", "subset cardinality"),
    definition="The cardinality of a finite additive set A is the number of elements in A.",
)
def cardinality(A: AdditiveSet) -> int:
    r"""
    Return the cardinality of a finite additive set. For a finite subset :math:`A` of an abelian group, the cardinality is :math:`|A|`.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The number of elements of :math:`A`.

    """
    return A.size


@invariant_metadata(
    display_name="Sumset size",
    notation=r"|A+A|",
    category="additive combinatorics invariants",
    aliases=("self-sumset size",),
    definition="The sumset size of A is the cardinality of A + A.",
)
def sumset_size(A: AdditiveSet) -> int:
    r"""
    Return the size of the self-sumset.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The cardinality of :math:`A + A`.
    """
    return A.sumset().size


@invariant_metadata(
    display_name="Difference set size",
    notation=r"|A-A|",
    category="additive combinatorics invariants",
    aliases=("self-difference-set size",),
    definition="The difference-set size of A is the cardinality of A - A.",
)
def diffset_size(A: AdditiveSet) -> int:
    r"""
    Return the size of the self-difference set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The cardinality of :math:`A - A`.
    """
    return A.diffset().size


@invariant_metadata(
    display_name="Doubling constant",
    notation=r"\sigma[A] = \frac{|A+A|}{|A|}",
    category="additive combinatorics invariants",
    aliases=("doubling ratio",),
    definition="The doubling constant of a nonempty finite additive set A is |A+A| / |A|.",
)
def doubling_constant(A: AdditiveSet) -> float:
    r"""
    Return the doubling constant of a finite additive set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    float
        The ratio :math:`|A+A| / |A|`.

    Raises
    ------
    ValueError
        If ``A`` is empty.
    """
    if A.is_empty:
        raise ValueError("doubling_constant is undefined for the empty set.")
    return float(A.sumset().size / A.size)


@invariant_metadata(
    display_name="Difference constant",
    notation=r"\delta[A] = \frac{|A-A|}{|A|}",
    category="additive combinatorics invariants",
    aliases=("difference ratio",),
    definition="The difference constant of a nonempty finite additive set A is |A-A| / |A|.",
)
def difference_constant(A: AdditiveSet) -> float:
    r"""
    Return the difference constant of a finite additive set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    float
        The ratio :math:`|A-A| / |A|`.

    Raises
    ------
    ValueError
        If ``A`` is empty.
    """
    if A.is_empty:
        raise ValueError("difference_constant is undefined for the empty set.")
    return float(A.diffset().size / A.size)


@invariant_metadata(
    display_name="Tripling constant",
    notation=r"\frac{|A+A+A|}{|A|}",
    category="additive combinatorics invariants",
    aliases=("tripling ratio",),
    definition="The tripling constant of a nonempty finite additive set A is |A+A+A| / |A|.",
)
def tripling_constant(A: AdditiveSet) -> float:
    r"""
    Return the tripling constant of a finite additive set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    float
        The ratio :math:`|A+A+A| / |A|`.

    Raises
    ------
    ValueError
        If ``A`` is empty.
    """
    if A.is_empty:
        raise ValueError("tripling_constant is undefined for the empty set.")
    return float(A.k_fold_sum(3).size / A.size)


@invariant_metadata(
    display_name="Additive energy",
    notation=r"E(A)",
    category="additive combinatorics invariants",
    aliases=("self-additive energy",),
    definition=(
        "The additive energy of a finite additive set A is the number of quadruples "
        "(a1, a2, a3, a4) in A^4 satisfying a1 + a2 = a3 + a4."
    ),
)
def additive_energy(A: AdditiveSet) -> int:
    r"""
    Return the additive energy of a finite additive set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The number of additive quadruples in :math:`A`.
    """
    reps = A.representation_function()
    return int(sum(v * v for v in reps.values()))


@invariant_metadata(
    display_name="Maximum sum representation count",
    notation=r"\max_x r_{A,A}(x)",
    category="additive combinatorics invariants",
    aliases=("max sum multiplicity", "maximum sum multiplicity"),
    definition="The maximum sum representation count of A is the largest number of ordered representations of an element x as a1 + a2 with a1, a2 in A.",
)
def max_sum_representation_count(A: AdditiveSet) -> int:
    r"""
    Return the maximum self-sum representation count.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The maximum value of :math:`r_{A,A}(x)` over all ambient group elements
        :math:`x`.
    """
    return A.max_sum_representation_count()


@invariant_metadata(
    display_name="Maximum difference representation count",
    notation=r"\max_x r_{A,-A}(x)",
    category="additive combinatorics invariants",
    aliases=("max difference multiplicity", "maximum difference multiplicity"),
    definition="The maximum difference representation count of A is the largest number of ordered representations of an element x as a1 - a2 with a1, a2 in A.",
)
def max_difference_representation_count(A: AdditiveSet) -> int:
    r"""
    Return the maximum self-difference representation count.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The maximum number of ordered representations of an element as a
        difference of two elements of :math:`A`.
    """
    diff_counts: dict = {}
    for a1 in A.elements:
        for a2 in A.elements:
            x = A.group.sub(a1, a2)
            diff_counts[x] = diff_counts.get(x, 0) + 1
    return max(diff_counts.values(), default=0)


@invariant_metadata(
    display_name="Stabilizer size",
    notation=r"|\operatorname{Stab}(A)|",
    category="additive combinatorics invariants",
    aliases=("period size",),
    definition="The stabilizer size of A is the cardinality of the subgroup of all ambient group elements g satisfying A + g = A.",
)
def stabilizer_size(A: AdditiveSet) -> int:
    r"""
    Return the size of the additive stabilizer of a set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The cardinality of the stabilizer subgroup of :math:`A`.
    """
    return A.stabilizer().size


@invariant_metadata(
    display_name="Stabilizer size of the sumset",
    notation=r"|\operatorname{Stab}(A+A)|",
    category="additive combinatorics invariants",
    aliases=("sumset period size",),
    definition="The stabilizer size of the sumset of A is the cardinality of the subgroup of all ambient group elements g satisfying (A + A) + g = A + A.",
)
def stabilizer_size_of_sumset(A: AdditiveSet) -> int:
    r"""
    Return the size of the stabilizer of the self-sumset.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The cardinality of the stabilizer subgroup of :math:`A + A`.
    """
    return A.sumset().stabilizer().size


@invariant_metadata(
    display_name="Sumset defect",
    notation=r"|A+A| - |A|",
    category="additive combinatorics invariants",
    aliases=("doubling defect",),
    definition="The sumset defect of A is |A+A| - |A|.",
)
def sumset_defect(A: AdditiveSet) -> int:
    r"""
    Return the sumset defect of a finite additive set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The quantity :math:`|A+A| - |A|`.
    """
    return A.sumset().size - A.size


@invariant_metadata(
    display_name="Difference-set defect",
    notation=r"|A-A| - |A|",
    category="additive combinatorics invariants",
    aliases=("difference defect",),
    definition="The difference-set defect of A is |A-A| - |A|.",
)
def diffset_defect(A: AdditiveSet) -> int:
    r"""
    Return the difference-set defect of a finite additive set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The quantity :math:`|A-A| - |A|`.
    """
    return A.diffset().size - A.size


@invariant_metadata(
    display_name="Doubling minus difference",
    notation=r"|A+A| - |A-A|",
    category="additive combinatorics invariants",
    aliases=("sum-difference gap",),
    definition="The doubling-minus-difference invariant of A is |A+A| - |A-A|.",
)
def doubling_minus_difference(A: AdditiveSet) -> int:
    r"""
    Return the difference between the self-sumset size and the self-difference-set size.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The quantity :math:`|A+A| - |A-A|`.
    """
    return A.sumset().size - A.diffset().size


@invariant_metadata(
    display_name="Sumset density",
    notation=r"\frac{|A+A|}{|G|}",
    category="additive combinatorics invariants",
    aliases=("relative sumset size",),
    definition="The sumset density of A is |A+A| / |G|, where G is the ambient group.",
)
def sumset_density(A: AdditiveSet) -> float:
    r"""
    Return the density of the self-sumset in the ambient group.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    float
        The ratio :math:`|A+A| / |G|`.
    """
    return float(A.sumset().size / A.group.order)


@invariant_metadata(
    display_name="Difference-set density",
    notation=r"\frac{|A-A|}{|G|}",
    category="additive combinatorics invariants",
    aliases=("relative difference-set size",),
    definition="The difference-set density of A is |A-A| / |G|, where G is the ambient group.",
)
def diffset_density(A: AdditiveSet) -> float:
    r"""
    Return the density of the self-difference set in the ambient group.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    float
        The ratio :math:`|A-A| / |G|`.
    """
    return float(A.diffset().size / A.group.order)


@invariant_metadata(
    display_name="Stabilizer index",
    notation=r"[G : \operatorname{Stab}(A)]",
    category="additive combinatorics invariants",
    aliases=("period index",),
    definition="The stabilizer index of A is the index of the stabilizer subgroup of A in the ambient group G.",
)
def stabilizer_index(A: AdditiveSet) -> int:
    r"""
    Return the index of the stabilizer subgroup of a finite additive set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The quotient :math:`|G| / |\operatorname{Stab}(A)|`.
    """
    return A.group.order // A.stabilizer().size


@invariant_metadata(
    display_name="Sumset stabilizer index",
    notation=r"[G : \operatorname{Stab}(A+A)]",
    category="additive combinatorics invariants",
    aliases=("sumset period index",),
    definition="The sumset stabilizer index of A is the index of the stabilizer subgroup of A + A in the ambient group G.",
)
def sumset_stabilizer_index(A: AdditiveSet) -> int:
    r"""
    Return the index of the stabilizer subgroup of the self-sumset.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    int
        The quotient :math:`|G| / |\operatorname{Stab}(A+A)|`.
    """
    return A.group.order // A.sumset().stabilizer().size


@invariant_metadata(
    display_name="Normalized additive energy",
    notation=r"\frac{E(A)}{|A|^3}",
    category="additive combinatorics invariants",
    aliases=("relative additive energy",),
    definition="The normalized additive energy of a nonempty set A is E(A) / |A|^3.",
)
def normalized_additive_energy(A: AdditiveSet) -> float:
    r"""
    Return the additive energy normalized by :math:`|A|^3`.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    float
        The ratio :math:`E(A) / |A|^3`.

    Raises
    ------
    ValueError
        If ``A`` is empty.
    """
    if A.is_empty:
        raise ValueError("normalized_additive_energy is undefined for the empty set.")
    return float(additive_energy(A) / (A.size ** 3))


@invariant_metadata(
    display_name="Energy over sumset square",
    notation=r"\frac{E(A)}{|A+A|^2}",
    category="additive combinatorics invariants",
    aliases=("sumset-normalized energy",),
    definition="The sumset-normalized additive energy of A is E(A) / |A+A|^2 when the sumset is nonempty.",
)
def energy_over_sumset_square(A: AdditiveSet) -> float:
    r"""
    Return the additive energy normalized by the square of the sumset size.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    float
        The ratio :math:`E(A) / |A+A|^2`.

    Raises
    ------
    ValueError
        If :math:`A+A` is empty.
    """
    s = A.sumset().size
    if s == 0:
        raise ValueError("energy_over_sumset_square is undefined when A+A is empty.")
    return float(additive_energy(A) / (s ** 2))
