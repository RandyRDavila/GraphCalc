from __future__ import annotations

from collections import Counter
from typing import Iterable, Iterator, Sequence, Tuple

from graphcalc.additive_combinatorics.ambient_groups import Element, FiniteAbelianGroup

__all__ = ["AdditiveSet"]


class AdditiveSet:
    r"""
    Finite subset of a finite abelian ambient group.

    This class represents a subset :math:`A \subseteq G`, where
    :math:`G = \mathbb{Z}/n_1\mathbb{Z} \times \cdots \times \mathbb{Z}/n_k\mathbb{Z}`
    is modeled by :class:`graphcalc.additive_combinatorics.ambient_groups.FiniteAbelianGroup`.

    Elements are stored canonically as reduced tuples in the ambient group, and
    duplicates are removed automatically. The class is intended to serve as the
    core object for additive-combinatorial computations such as sumsets,
    difference sets, translations, dilations, stabilizers, and representation
    functions.

    Parameters
    ----------
    elements : iterable of sequence of int
        Elements of the subset. Each element must have the correct coordinate
        length for the ambient group. Representatives are reduced modulo the
        ambient moduli and deduplicated.
    group : FiniteAbelianGroup
        Ambient finite abelian group containing the set.
    validate : bool, default=True
        Whether to validate the inputs on construction.

    Notes
    -----
    The empty subset is allowed.

    The internal representation is canonical: if two input tuples represent the
    same ambient group element, they are identified. For example, in
    :math:`\mathbb{Z}/4\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z}`, the elements
    ``(5, -1)`` and ``(1, 4)`` are treated as the same element.

    Examples
    --------
    >>> from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
    >>> G = FiniteAbelianGroup((5,))
    >>> A = AdditiveSet([(0,), (1,), (6,)], group=G)
    >>> A.elements
    ((0,), (1,))
    >>> A.size
    2
    >>> A.sumset().elements
    ((0,), (1,), (2,))
    """

    def __init__(
        self,
        elements: Iterable[Sequence[int]],
        *,
        group: FiniteAbelianGroup,
        validate: bool = True,
    ) -> None:
        self._group = group

        if validate:
            self._validate_parameters()

        canonical = tuple(sorted({self._group.normalize(x) for x in elements}))
        self._elements = canonical

        if validate:
            self._validate_parameters()

    def __repr__(self) -> str:
        """
        Return a concise string representation of the additive set.

        Returns
        -------
        str
            Summary including the set size and the ambient group.
        """
        return f"AdditiveSet(size={self.size}, group={self.group!r})"

    @property
    def group(self) -> FiniteAbelianGroup:
        """
        Return the ambient finite abelian group.

        Returns
        -------
        FiniteAbelianGroup
            The ambient group in which the set is viewed as a subset.
        """
        return self._group

    @property
    def elements(self) -> Tuple[Element, ...]:
        """
        Return the canonical elements of the set.

        Returns
        -------
        tuple of Element
            The elements of the subset in sorted canonical form.
        """
        return self._elements

    @property
    def size(self) -> int:
        """
        Return the cardinality of the set.

        Returns
        -------
        int
            The number of distinct elements of the subset.
        """
        return len(self._elements)

    @property
    def is_empty(self) -> bool:
        """
        Return whether the subset is empty.

        Returns
        -------
        bool
            ``True`` if the set has no elements and ``False`` otherwise.
        """
        return self.size == 0

    def _validate_parameters(self) -> None:
        """
        Validate the internal data.

        Raises
        ------
        TypeError
            If the ambient group is not a :class:`FiniteAbelianGroup`.
        """
        if not isinstance(self._group, FiniteAbelianGroup):
            raise TypeError("group must be a FiniteAbelianGroup instance.")

    def copy(self) -> "AdditiveSet":
        """
        Return a copy of the additive set.

        Returns
        -------
        AdditiveSet
            A new additive set with the same elements and ambient group.
        """
        return AdditiveSet(self._elements, group=self._group, validate=False)

    def __contains__(self, x: Sequence[int]) -> bool:
        """
        Test membership in the subset.

        Parameters
        ----------
        x : sequence of int
            Candidate element.

        Returns
        -------
        bool
            ``True`` if ``x`` represents an element of the subset, and
            ``False`` otherwise.

        Raises
        ------
        ValueError
            If ``x`` does not have the correct coordinate length.
        """
        return self._group.normalize(x) in set(self._elements)

    def __iter__(self) -> Iterator[Element]:
        """
        Iterate over the elements of the subset.

        Returns
        -------
        iterator of Element
            Iterator over the canonical elements of the set.
        """
        return iter(self._elements)

    def __len__(self) -> int:
        """
        Return the cardinality of the subset.

        Returns
        -------
        int
            The number of distinct elements.
        """
        return self.size

    def equal(self, other: "AdditiveSet") -> bool:
        """
        Test equality as subsets of the same ambient group.

        Parameters
        ----------
        other : AdditiveSet
            Another additive set.

        Returns
        -------
        bool
            ``True`` if the two sets have the same ambient moduli and the same
            canonical elements, and ``False`` otherwise.
        """
        return (
            isinstance(other, AdditiveSet)
            and self.group.moduli == other.group.moduli
            and self.elements == other.elements
        )

    def contains_zero(self) -> bool:
        """
        Return whether the set contains the additive identity.

        Returns
        -------
        bool
            ``True`` if the ambient zero element belongs to the subset, and
            ``False`` otherwise.
        """
        return self.group.zero() in self._elements

    def negate(self) -> "AdditiveSet":
        r"""
        Return the reflected set :math:`-A`.

        Returns
        -------
        AdditiveSet
            The set :math:`-A = \{-a : a \in A\}` in the same ambient group.
        """
        return AdditiveSet((self.group.neg(x) for x in self._elements), group=self.group)

    def translate(self, x: Sequence[int]) -> "AdditiveSet":
        r"""
        Return the translate :math:`A + x`.

        Parameters
        ----------
        x : sequence of int
            Translation element in the ambient group.

        Returns
        -------
        AdditiveSet
            The translated set :math:`A + x = \{a + x : a \in A\}`.

        Raises
        ------
        ValueError
            If ``x`` does not have the correct coordinate length.
        """
        return AdditiveSet((self.group.add(a, x) for a in self._elements), group=self.group)

    def dilate(self, m: int) -> "AdditiveSet":
        r"""
        Return the dilate :math:`mA`.

        Parameters
        ----------
        m : int
            Integer scalar.

        Returns
        -------
        AdditiveSet
            The set :math:`mA = \{ma : a \in A\}` formed by scalar
            multiplication in the ambient group.
        """
        return AdditiveSet(
            (self.group.scalar_mul(m, a) for a in self._elements),
            group=self.group,
        )

    def sumset(self, other: "AdditiveSet | None" = None) -> "AdditiveSet":
        r"""
        Return a sumset.

        Parameters
        ----------
        other : AdditiveSet or None, default=None
            Second set. If omitted, computes the self-sumset :math:`A + A`.

        Returns
        -------
        AdditiveSet
            The sumset :math:`A + B = \{a+b : a \in A,\, b \in B\}` if
            ``other`` is provided, and :math:`A + A` otherwise.

        Raises
        ------
        ValueError
            If the ambient groups do not agree.
        """
        if other is None:
            other = self
        self._check_same_group(other)
        return AdditiveSet(
            (self.group.add(a, b) for a in self._elements for b in other._elements),
            group=self.group,
        )

    def diffset(self, other: "AdditiveSet | None" = None) -> "AdditiveSet":
        r"""
        Return a difference set.

        Parameters
        ----------
        other : AdditiveSet or None, default=None
            Second set. If omitted, computes :math:`A - A`.

        Returns
        -------
        AdditiveSet
            The difference set :math:`A - B = \{a-b : a \in A,\, b \in B\}` if
            ``other`` is provided, and :math:`A - A` otherwise.

        Raises
        ------
        ValueError
            If the ambient groups do not agree.
        """
        if other is None:
            other = self
        self._check_same_group(other)
        return AdditiveSet(
            (self.group.sub(a, b) for a in self._elements for b in other._elements),
            group=self.group,
        )

    def k_fold_sum(self, k: int) -> "AdditiveSet":
        r"""
        Return the iterated sumset :math:`kA`.

        Parameters
        ----------
        k : int
            Number of summands.

        Returns
        -------
        AdditiveSet
            The set of all sums of ``k`` elements of :math:`A`.

        Raises
        ------
        ValueError
            If ``k`` is negative.

        Notes
        -----
        By convention, :math:`0A = \{0\}`, where :math:`0` is the additive
        identity of the ambient group.
        """
        if k < 0:
            raise ValueError("k must be nonnegative.")

        if k == 0:
            return AdditiveSet((self.group.zero(),), group=self.group)

        out = self.copy()
        for _ in range(k - 1):
            out = out.sumset(self)
        return out

    def representation_function(self, other: "AdditiveSet | None" = None) -> dict[Element, int]:
        r"""
        Return the additive representation function.

        Parameters
        ----------
        other : AdditiveSet or None, default=None
            Second set. If omitted, computes the representation counts for
            :math:`A + A`.

        Returns
        -------
        dict of Element to int
            Dictionary mapping each group element :math:`x` appearing in the
            sumset to the number of pairs with sum :math:`x`. That is, the
            returned value is the function

            .. math::

                r_{A,B}(x) = |\{(a,b) \in A \times B : a+b=x\}|.

        Raises
        ------
        ValueError
            If the ambient groups do not agree.
        """
        if other is None:
            other = self
        self._check_same_group(other)

        counts = Counter(
            self.group.add(a, b)
            for a in self._elements
            for b in other._elements
        )
        return dict(counts)

    def max_sum_representation_count(self, other: "AdditiveSet | None" = None) -> int:
        """
        Return the maximum value of the additive representation function.

        Parameters
        ----------
        other : AdditiveSet or None, default=None
            Second set. If omitted, computes the maximum over the self-sumset.

        Returns
        -------
        int
            The maximum number of representations of a group element as a sum
            of one element from each set. Returns ``0`` if the relevant product
            set is empty.
        """
        reps = self.representation_function(other)
        return max(reps.values(), default=0)

    def is_subset_of(self, other: "AdditiveSet") -> bool:
        """
        Return whether this set is a subset of another additive set.

        Parameters
        ----------
        other : AdditiveSet
            Candidate superset.

        Returns
        -------
        bool
            ``True`` if every element of this set belongs to ``other`` and the
            ambient groups agree, and ``False`` otherwise.
        """
        if self.group.moduli != other.group.moduli:
            return False
        other_elements = set(other._elements)
        return all(x in other_elements for x in self._elements)

    def stabilizer(self) -> "AdditiveSet":
        r"""
        Return the additive stabilizer of the set.

        Returns
        -------
        AdditiveSet
            The subgroup

            .. math::

                \operatorname{Stab}(A) = \{g \in G : A + g = A\}.

        Notes
        -----
        This is computed by exhaustive search over the ambient group, so it is
        intended for small groups.
        """
        stabilizing = []
        for g in self.group.elements():
            if self.translate(g).elements == self.elements:
                stabilizing.append(g)
        return AdditiveSet(stabilizing, group=self.group)

    def _check_same_group(self, other: "AdditiveSet") -> None:
        """
        Validate that two additive sets live in the same ambient group.

        Parameters
        ----------
        other : AdditiveSet
            Another additive set.

        Raises
        ------
        ValueError
            If the ambient group moduli do not agree.
        """
        if self.group.moduli != other.group.moduli:
            raise ValueError("Additive sets must live in the same ambient group.")
