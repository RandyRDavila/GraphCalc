from __future__ import annotations

from graphcalc.metadata import invariant_metadata
from graphcalc.additive_combinatorics.sets import AdditiveSet

__all__ = [
    "contains_zero",
    "is_symmetric",
    "is_subgroup",
    "is_coset",
    "is_sum_free",
    "is_sidon",
    "has_small_doubling",
    "is_periodic",
    "is_aperiodic",
    "is_empty_set",
    "is_trivial_set",
    "is_whole_group",
    "has_full_sumset",
    "has_full_diffset",
    "is_doubling_at_most_3_over_2",
    "is_doubling_at_most_2",
    "is_difference_larger_than_sumset",
    "sumset_is_periodic",
]


@invariant_metadata(
    display_name="Contains zero",
    notation=r"0 \in A",
    category="additive combinatorics predicates",
    aliases=("contains identity",),
    definition="A finite additive set A contains zero if the additive identity of the ambient group belongs to A.",
)
def contains_zero(A: AdditiveSet) -> bool:
    r"""
    Return whether the additive identity belongs to the set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if the ambient group identity belongs to :math:`A`, and
        ``False`` otherwise.
    """
    return A.contains_zero()


@invariant_metadata(
    display_name="Is symmetric",
    notation=r"A = -A",
    category="additive combinatorics predicates",
    aliases=("centrally symmetric",),
    definition="A finite additive set A is symmetric if A = -A.",
)
def is_symmetric(A: AdditiveSet) -> bool:
    r"""
    Return whether the set is symmetric under additive inversion.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A = -A`, and ``False`` otherwise.
    """
    return A.equal(A.negate())


@invariant_metadata(
    display_name="Is subgroup",
    notation=r"A \le G",
    category="additive combinatorics predicates",
    aliases=("is additive subgroup",),
    definition="A finite additive set A is a subgroup if it contains zero and is closed under subtraction.",
)
def is_subgroup(A: AdditiveSet) -> bool:
    r"""
    Return whether the set is an additive subgroup of its ambient group.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A` is a subgroup of its ambient group, and
        ``False`` otherwise.

    Notes
    -----
    In an abelian group, closure under subtraction and containment of the
    identity are equivalent to being a subgroup.
    """
    if not A.contains_zero():
        return False

    Aset = set(A.elements)
    for a in A.elements:
        for b in A.elements:
            if A.group.sub(a, b) not in Aset:
                return False
    return True


@invariant_metadata(
    display_name="Is coset",
    notation=r"A = x + H",
    category="additive combinatorics predicates",
    aliases=("is affine subgroup translate",),
    definition="A finite additive set A is a coset if it is empty, or if for some element x the translate A - x is a subgroup of the ambient group.",
)
def is_coset(A: AdditiveSet) -> bool:
    r"""
    Return whether the set is a coset of a subgroup.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A` is a coset of some subgroup of the ambient group,
        and ``False`` otherwise.

    Notes
    -----
    The empty set is not considered a coset here.
    """
    if A.is_empty:
        return False

    anchor = A.elements[0]
    translated = A.translate(A.group.neg(anchor))
    return is_subgroup(translated)


@invariant_metadata(
    display_name="Is sum-free",
    notation=r"(A+A)\cap A = \varnothing",
    category="additive combinatorics predicates",
    aliases=("sum free",),
    definition="A finite additive set A is sum-free if no sum of two elements of A belongs to A.",
)
def is_sum_free(A: AdditiveSet) -> bool:
    r"""
    Return whether the set is sum-free.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`(A+A)\cap A = \varnothing`, and ``False`` otherwise.
    """
    Aset = set(A.elements)
    for a in A.elements:
        for b in A.elements:
            if A.group.add(a, b) in Aset:
                return False
    return True


@invariant_metadata(
    display_name="Is Sidon",
    notation=r"r_{A,A}(x) \le 2 \text{ for } x \ne 2a",
    category="additive combinatorics predicates",
    aliases=("is B2 set",),
    definition=(
        "A finite additive set A is Sidon if whenever a1 + a2 = a3 + a4 with ai in A, "
        "the unordered pairs {a1, a2} and {a3, a4} coincide."
    ),
)
def is_sidon(A: AdditiveSet) -> bool:
    r"""
    Return whether the set is a Sidon set.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A` is a Sidon set, and ``False`` otherwise.

    Notes
    -----
    This implementation checks uniqueness of unordered pair sums.
    """
    seen: dict = {}
    elems = list(A.elements)

    for i, a in enumerate(elems):
        for j in range(i, len(elems)):
            b = elems[j]
            s = A.group.add(a, b)
            pair = (a, b)
            if s in seen and seen[s] != pair:
                return False
            seen[s] = pair
    return True


@invariant_metadata(
    display_name="Has small doubling",
    notation=r"|A+A| \le K|A|",
    category="additive combinatorics predicates",
    aliases=("small doubling",),
    definition="A finite additive set A has small doubling with parameter K if |A+A| <= K|A|.",
)
def has_small_doubling(A: AdditiveSet, K: float) -> bool:
    r"""
    Return whether the set has doubling at most ``K``.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.
    K : float
        Doubling threshold.

    Returns
    -------
    bool
        ``True`` if :math:`|A+A| \le K|A|`, and ``False`` otherwise.

    Raises
    ------
    ValueError
        If ``K`` is negative.

    Notes
    -----
    For the empty set, this predicate returns ``True`` for every nonnegative
    ``K``, since both sides of the inequality are zero.
    """
    if K < 0:
        raise ValueError("K must be nonnegative.")
    return A.sumset().size <= K * A.size


@invariant_metadata(
    display_name="Is periodic",
    notation=r"|\operatorname{Stab}(A)| > 1",
    category="additive combinatorics predicates",
    aliases=("has nontrivial period",),
    definition="A finite additive set A is periodic if its stabilizer is nontrivial.",
)
def is_periodic(A: AdditiveSet) -> bool:
    r"""
    Return whether the set has nontrivial additive stabilizer.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`\operatorname{Stab}(A)` has more than one element,
        and ``False`` otherwise.
    """
    return A.stabilizer().size > 1


@invariant_metadata(
    display_name="Is aperiodic",
    notation=r"|\operatorname{Stab}(A)| = 1",
    category="additive combinatorics predicates",
    aliases=("has trivial period",),
    definition="A finite additive set A is aperiodic if its stabilizer is trivial.",
)
def is_aperiodic(A: AdditiveSet) -> bool:
    r"""
    Return whether the set has trivial additive stabilizer.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`\operatorname{Stab}(A)` is the trivial subgroup, and
        ``False`` otherwise.
    """
    return A.stabilizer().size == 1


@invariant_metadata(
    display_name="Is empty set",
    notation=r"A = \varnothing",
    category="additive combinatorics predicates",
    aliases=("is empty",),
    definition="A finite additive set A is empty if it has no elements.",
)
def is_empty_set(A: AdditiveSet) -> bool:
    r"""
    Return whether the additive set is empty.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A = \varnothing`, and ``False`` otherwise.
    """
    return A.is_empty


@invariant_metadata(
    display_name="Is trivial set",
    notation=r"|A| \le 1",
    category="additive combinatorics predicates",
    aliases=("has at most one element",),
    definition="A finite additive set A is trivial if it has cardinality at most one.",
)
def is_trivial_set(A: AdditiveSet) -> bool:
    r"""
    Return whether the additive set has cardinality at most one.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`|A| \le 1`, and ``False`` otherwise.
    """
    return A.size <= 1


@invariant_metadata(
    display_name="Is whole group",
    notation=r"A = G",
    category="additive combinatorics predicates",
    aliases=("is ambient group",),
    definition="A finite additive set A is the whole group if it contains every element of the ambient group.",
)
def is_whole_group(A: AdditiveSet) -> bool:
    r"""
    Return whether the additive set equals its ambient group.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A = G`, and ``False`` otherwise.
    """
    return A.size == A.group.order


@invariant_metadata(
    display_name="Has full sumset",
    notation=r"A+A = G",
    category="additive combinatorics predicates",
    aliases=("sumset is full",),
    definition="A finite additive set A has full sumset if A + A equals the ambient group.",
)
def has_full_sumset(A: AdditiveSet) -> bool:
    r"""
    Return whether the self-sumset fills the ambient group.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A+A = G`, and ``False`` otherwise.
    """
    return A.sumset().size == A.group.order


@invariant_metadata(
    display_name="Has full difference set",
    notation=r"A-A = G",
    category="additive combinatorics predicates",
    aliases=("difference set is full",),
    definition="A finite additive set A has full difference set if A - A equals the ambient group.",
)
def has_full_diffset(A: AdditiveSet) -> bool:
    r"""
    Return whether the self-difference set fills the ambient group.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A-A = G`, and ``False`` otherwise.
    """
    return A.diffset().size == A.group.order


@invariant_metadata(
    display_name="Doubling at most 3/2",
    notation=r"|A+A| \le \frac{3}{2}|A|",
    category="additive combinatorics predicates",
    aliases=("very small doubling",),
    definition="A finite additive set A has doubling at most 3/2 if |A+A| <= (3/2)|A|.",
)
def is_doubling_at_most_3_over_2(A: AdditiveSet) -> bool:
    r"""
    Return whether the set has doubling at most :math:`3/2`.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`|A+A| \le \frac{3}{2}|A|`, and ``False`` otherwise.
    """
    return has_small_doubling(A, 1.5)


@invariant_metadata(
    display_name="Doubling at most 2",
    notation=r"|A+A| \le 2|A|",
    category="additive combinatorics predicates",
    aliases=("small doubling at most two",),
    definition="A finite additive set A has doubling at most 2 if |A+A| <= 2|A|.",
)
def is_doubling_at_most_2(A: AdditiveSet) -> bool:
    r"""
    Return whether the set has doubling at most 2.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`|A+A| \le 2|A|`, and ``False`` otherwise.
    """
    return has_small_doubling(A, 2.0)


@invariant_metadata(
    display_name="Difference larger than sumset",
    notation=r"|A-A| > |A+A|",
    category="additive combinatorics predicates",
    aliases=("MSTD complement",),
    definition="A finite additive set A has larger difference set than sumset if |A-A| > |A+A|.",
)
def is_difference_larger_than_sumset(A: AdditiveSet) -> bool:
    r"""
    Return whether the self-difference set is larger than the self-sumset.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`|A-A| > |A+A|`, and ``False`` otherwise.
    """
    return A.diffset().size > A.sumset().size


@invariant_metadata(
    display_name="Sumset is periodic",
    notation=r"|\operatorname{Stab}(A+A)| > 1",
    category="additive combinatorics predicates",
    aliases=("sumset has nontrivial period",),
    definition="A finite additive set A has periodic sumset if the stabilizer of A + A is nontrivial.",
)
def sumset_is_periodic(A: AdditiveSet) -> bool:
    r"""
    Return whether the self-sumset has nontrivial stabilizer.

    Parameters
    ----------
    A : AdditiveSet
        Input additive set.

    Returns
    -------
    bool
        ``True`` if :math:`A+A` is periodic, and ``False`` otherwise.
    """
    return A.sumset().stabilizer().size > 1
