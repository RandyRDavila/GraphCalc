from __future__ import annotations

from itertools import product
from math import prod
from typing import Iterator, Sequence, Tuple

Element = Tuple[int, ...]

__all__ = ["Element", "FiniteAbelianGroup"]


class FiniteAbelianGroup:
    r"""
    Finite abelian group of the form

    .. math::

        \mathbb{Z}/n_1\mathbb{Z} \times \cdots \times \mathbb{Z}/n_k\mathbb{Z}.

    This class provides a lightweight explicit model for finite additive groups
    used in additive combinatorics. Group elements are represented canonically
    as tuples of integers, one coordinate for each cyclic factor, with all
    arithmetic performed coordinatewise modulo the corresponding modulus.

    The main intended use is as the ambient group for
    :class:`graphcalc.additive_combinatorics.sets.AdditiveSet`, where one wants
    a mathematically transparent representation of finite subsets together with
    exact additive operations such as sumsets, difference sets, translations,
    and stabilizers.

    Parameters
    ----------
    moduli : sequence of int
        Positive integers specifying the cyclic factors. For example,
        ``moduli=(5,)`` represents :math:`\mathbb{Z}/5\mathbb{Z}`, while
        ``moduli=(2, 3)`` represents
        :math:`\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/3\mathbb{Z}`.
    validate : bool, default=True
        Whether to validate the moduli on construction.

    Notes
    -----
    Elements are stored and returned in canonical reduced form. For instance,
    in ``FiniteAbelianGroup((4, 5))``, the tuples ``(5, -1)`` and ``(1, 4)``
    represent the same group element.

    This class deliberately models only nontrivial finite products of cyclic
    groups, so ``moduli`` must be nonempty and all entries must be positive.

    Examples
    --------
    Construct the cyclic group :math:`\mathbb{Z}/7\mathbb{Z}`:

    >>> G = FiniteAbelianGroup((7,))
    >>> G.order
    7
    >>> G.add((3,), (6,))
    (2,)

    Construct a product group and perform arithmetic:

    >>> G = FiniteAbelianGroup((4, 5))
    >>> G.normalize((5, -1))
    (1, 4)
    >>> G.add((3, 4), (2, 3))
    (1, 2)
    >>> G.neg((1, 2))
    (3, 3)
    """

    def __init__(
        self,
        moduli: Sequence[int],
        *,
        validate: bool = True,
    ) -> None:
        self._moduli = tuple(int(n) for n in moduli)
        if validate:
            self._validate_parameters()

    def __repr__(self) -> str:
        """
        Return a concise string representation of the group.

        Returns
        -------
        str
            Summary including the rank, order, and cyclic moduli.
        """
        return (
            f"FiniteAbelianGroup(rank={self.rank}, "
            f"order={self.order}, moduli={self.moduli})"
        )

    @property
    def moduli(self) -> Tuple[int, ...]:
        """
        Return the cyclic moduli defining the ambient group.

        Returns
        -------
        tuple of int
            The tuple ``(n_1, ..., n_k)`` such that the group is
            :math: `\\mathbb{Z}/n_1\\mathbb{Z} \\times \\cdots \times \\mathbb{Z}/n_k\\mathbb{Z}`.
        """
        return self._moduli

    @property
    def rank(self) -> int:
        """
        Return the number of cyclic factors.

        Returns
        -------
        int
            The number of coordinates in the ambient product group.

        Notes
        -----
        This is the number of cyclic components in the chosen presentation,
        not a minimal generating rank invariant.
        """
        return len(self._moduli)

    @property
    def order(self) -> int:
        """
        Return the cardinality of the group.

        Returns
        -------
        int
            The product of the cyclic moduli.
        """
        return prod(self._moduli)

    def _validate_parameters(self) -> None:
        """
        Validate the defining moduli.

        Raises
        ------
        ValueError
            If ``moduli`` is empty or contains a nonpositive entry.
        """
        if not self._moduli:
            raise ValueError("moduli must be a nonempty sequence of positive integers.")
        if any(n <= 0 for n in self._moduli):
            raise ValueError("Every modulus must be positive.")

    def _validate_element_shape(self, x: Sequence[int]) -> None:
        """
        Validate that an element has the correct number of coordinates.

        Parameters
        ----------
        x : sequence of int
            Candidate element representation.

        Raises
        ------
        ValueError
            If the length of ``x`` does not match :attr:`rank`.
        """
        if len(x) != self.rank:
            raise ValueError(
                f"Element must have length {self.rank} to match moduli={self.moduli}."
            )

    def normalize(self, x: Sequence[int]) -> Element:
        """
        Return the canonical representative of a group element.

        Parameters
        ----------
        x : sequence of int
            Tuple-like representation of a group element.

        Returns
        -------
        Element
            The coordinatewise reduced tuple, with the ``i``-th coordinate taken
            modulo ``moduli[i]``.

        Raises
        ------
        ValueError
            If ``x`` does not have the correct number of coordinates.

        Examples
        --------
        >>> G = FiniteAbelianGroup((4, 5))
        >>> G.normalize((5, -1))
        (1, 4)
        """
        self._validate_element_shape(x)
        return tuple(int(a) % n for a, n in zip(x, self._moduli))

    def zero(self) -> Element:
        """
        Return the additive identity.

        Returns
        -------
        Element
            The zero element ``(0, ..., 0)`` of the ambient group.
        """
        return tuple(0 for _ in self._moduli)

    def add(self, x: Sequence[int], y: Sequence[int]) -> Element:
        """
        Add two group elements.

        Parameters
        ----------
        x, y : sequence of int
            Elements of the ambient group.

        Returns
        -------
        Element
            The sum ``x + y`` computed coordinatewise modulo the corresponding
            cyclic moduli.

        Raises
        ------
        ValueError
            If either input does not have the correct number of coordinates.

        Examples
        --------
        >>> G = FiniteAbelianGroup((4, 5))
        >>> G.add((3, 4), (2, 3))
        (1, 2)
        """
        self._validate_element_shape(x)
        self._validate_element_shape(y)
        return tuple((int(a) + int(b)) % n for a, b, n in zip(x, y, self._moduli))

    def neg(self, x: Sequence[int]) -> Element:
        """
        Return the additive inverse of an element.

        Parameters
        ----------
        x : sequence of int
            Element of the ambient group.

        Returns
        -------
        Element
            The inverse ``-x`` computed coordinatewise modulo the corresponding
            cyclic moduli.

        Raises
        ------
        ValueError
            If ``x`` does not have the correct number of coordinates.

        Examples
        --------
        >>> G = FiniteAbelianGroup((4, 5))
        >>> G.neg((1, 2))
        (3, 3)
        """
        self._validate_element_shape(x)
        return tuple((-int(a)) % n for a, n in zip(x, self._moduli))

    def sub(self, x: Sequence[int], y: Sequence[int]) -> Element:
        """
        Subtract one group element from another.

        Parameters
        ----------
        x, y : sequence of int
            Elements of the ambient group.

        Returns
        -------
        Element
            The difference ``x - y`` computed coordinatewise modulo the
            corresponding cyclic moduli.

        Raises
        ------
        ValueError
            If either input does not have the correct number of coordinates.
        """
        self._validate_element_shape(x)
        self._validate_element_shape(y)
        return tuple((int(a) - int(b)) % n for a, b, n in zip(x, y, self._moduli))

    def contains(self, x: Sequence[int]) -> bool:
        """
        Test whether a tuple can be interpreted as an element of the group.

        Parameters
        ----------
        x : sequence of int
            Candidate element representation.

        Returns
        -------
        bool
            ``True`` if ``x`` has the correct number of coordinates and
            therefore determines an element of the ambient group, and ``False``
            otherwise.

        Notes
        -----
        Since element representatives are interpreted modulo the defining
        moduli, any integer tuple of the correct length is accepted.
        """
        return len(tuple(x)) == self.rank

    def elements(self) -> Iterator[Element]:
        """
        Iterate over all elements of the group.

        Returns
        -------
        iterator of Element
            Iterator over the canonical representatives of all group elements in
            lexicographic order.

        Notes
        -----
        This method is suitable for small groups. For larger ambient groups, the
        total number of elements is :attr:`order`, which may be too large for
        exhaustive enumeration.
        """
        return product(*(range(n) for n in self._moduli))

    def list_elements(self) -> list[Element]:
        """
        Return all group elements as a list.

        Returns
        -------
        list of Element
            The full list of canonical representatives in lexicographic order.

        See Also
        --------
        elements
            Lazy iterator over all elements.
        """
        return list(self.elements())

    def random_element(self, rng=None) -> Element:
        """
        Return a random group element.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator to use. If omitted, a new default generator
            is created.

        Returns
        -------
        Element
            A uniformly sampled element of the ambient group, represented in
            canonical coordinates.

        Notes
        -----
        Sampling is independent across coordinates, with the ``i``-th
        coordinate chosen uniformly from ``{0, ..., moduli[i]-1}``.
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        return tuple(int(rng.integers(0, n)) for n in self._moduli)

    def scalar_mul(self, m: int, x: Sequence[int]) -> Element:
        """
        Return the repeated sum :math:`m x`.

        Parameters
        ----------
        m : int
            Integer scalar.
        x : sequence of int
            Element of the ambient group.

        Returns
        -------
        Element
            The element obtained by coordinatewise multiplication of ``x`` by
            ``m``, reduced modulo the corresponding cyclic moduli.

        Raises
        ------
        ValueError
            If ``x`` does not have the correct number of coordinates.
        """
        self._validate_element_shape(x)
        return tuple((int(m) * int(a)) % n for a, n in zip(x, self._moduli))

    def equal(self, x: Sequence[int], y: Sequence[int]) -> bool:
        """
        Test whether two tuples represent the same group element.

        Parameters
        ----------
        x, y : sequence of int
            Candidate element representations.

        Returns
        -------
        bool
            ``True`` if ``x`` and ``y`` agree after canonical reduction modulo
            the defining moduli, and ``False`` otherwise.

        Examples
        --------
        >>> G = FiniteAbelianGroup((4, 5))
        >>> G.equal((5, -1), (1, 4))
        True
        >>> G.equal((1, 4), (1, 3))
        False
        """
        return self.normalize(x) == self.normalize(y)
