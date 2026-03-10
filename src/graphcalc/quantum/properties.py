# src/graphcalc/quantum/properties.py

from __future__ import annotations

from itertools import combinations
from math import isclose
from typing import Iterable, Sequence

import numpy as np

from graphcalc.metadata import invariant_metadata
from graphcalc.quantum.states import QuantumState

__all__ = [
    "is_valid_state",
    "is_pure",
    "is_mixed",
    "has_positive_partial_transpose",
    "is_product_state",
    "is_entangled",
]


@invariant_metadata(
    display_name="Valid state property",
    notation=r"\mathrm{valid}(\rho)",
    category="quantum state properties",
    aliases=("is valid state", "is valid density operator"),
    definition=(
        "A quantum state ρ is valid if it is a density operator: Hermitian, positive semidefinite, and of trace one."
    ),
)
def is_valid_state(
    state: QuantumState,
    *,
    tol: float | None = None,
) -> bool:
    """
    Return whether the input is a valid density operator.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    tol : float | None, default=None
        Optional tolerance to use in the validation. If omitted, ``state.tol``
        is used.

    Notes
    -----
    A valid density operator is Hermitian, positive semidefinite, and has
    trace one.

    This function is primarily useful because some operators produced from
    valid states, such as partial transposes, are not necessarily valid
    density operators.
    """
    use_tol = state.tol if tol is None else float(tol)

    rho = state.rho
    if rho.shape != (state.dimension, state.dimension):
        return False

    if not np.allclose(rho, rho.conj().T, atol=use_tol, rtol=0.0):
        return False

    tr = np.trace(rho)
    if not isclose(float(tr.real), 1.0, rel_tol=0.0, abs_tol=use_tol):
        return False
    if abs(tr.imag) > use_tol:
        return False

    evals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    return bool(np.all(evals >= -use_tol))


@invariant_metadata(
    display_name="Purity property",
    notation=r"\mathrm{pure}(\rho)",
    category="quantum state properties",
    aliases=("is pure",),
    definition=(
        "A quantum state ρ is pure if Tr(ρ^2)=1; equivalently, if its density operator has rank one."
    ),
)
def is_pure(state: QuantumState, *, tol: float | None = None) -> bool:
    """
    Return whether the state is pure.

    Notes
    -----
    A state ``rho`` is pure if and only if ``Tr(rho^2) = 1``. Equivalently,
    ``rho`` has rank one.
    """
    use_tol = state.tol if tol is None else float(tol)
    return isclose(state.purity(), 1.0, rel_tol=0.0, abs_tol=use_tol)


@invariant_metadata(
    display_name="Mixed-state property",
    notation=r"\mathrm{mixed}(\rho)",
    category="quantum state properties",
    aliases=("is mixed",),
    definition=(
        "A quantum state ρ is mixed if it is not pure."
    ),
)
def is_mixed(state: QuantumState, *, tol: float | None = None) -> bool:
    """
    Return whether the state is mixed.

    Notes
    -----
    A state is mixed if it is not pure.
    """
    return not is_pure(state, tol=tol)


@invariant_metadata(
    display_name="Positive partial transpose property",
    notation=r"\mathrm{PPT}(\rho)",
    category="quantum state properties",
    aliases=("has positive partial transpose", "is PPT"),
    definition=(
        "A quantum state ρ has positive partial transpose with respect to a chosen subsystem set if the corresponding partial transpose ρ^Γ is positive semidefinite."
    ),
)
def has_positive_partial_transpose(
    state: QuantumState,
    *,
    subsystems: Iterable[int],
    tol: float | None = None,
) -> bool:
    """
    Return whether the partial transpose is positive semidefinite.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    subsystems : iterable of int
        Indices of subsystems on which the partial transpose is taken.
    tol : float | None, default=None
        Numerical tolerance for positive semidefiniteness.

    Notes
    -----
    This is the PPT property. In bipartite systems of dimensions
    ``2 x 2`` and ``2 x 3``, PPT is equivalent to separability. In higher
    dimensions, PPT is only a necessary condition for separability in general.
    """
    use_tol = state.tol if tol is None else float(tol)
    pt = state.partial_transpose(subsystems)
    evals = np.linalg.eigvalsh(0.5 * (pt.rho + pt.rho.conj().T))
    return bool(np.all(evals >= -use_tol))


def _sorted_unique_subsystems(subsystems: Iterable[int], n: int) -> tuple[int, ...]:
    out = tuple(sorted(set(int(i) for i in subsystems)))
    if any(i < 0 or i >= n for i in out):
        raise ValueError("Subsystem index out of range.")
    return out


def _complement_subsystems(subsystems: Sequence[int], n: int) -> tuple[int, ...]:
    selected = set(subsystems)
    return tuple(i for i in range(n) if i not in selected)


def _tensor_close(
    left: QuantumState,
    right: QuantumState,
    target: QuantumState,
    *,
    tol: float,
) -> bool:
    candidate = left.tensor(right)
    return np.allclose(candidate.rho, target.rho, atol=tol, rtol=0.0)


@invariant_metadata(
    display_name="Product-state property",
    notation=r"\mathrm{product}(\rho)",
    category="quantum state properties",
    aliases=("is product state",),
    definition=(
        "A quantum state ρ is product with respect to a chosen bipartition A|B if ρ = ρ_A \\otimes ρ_B."
    ),
)
def is_product_state(
    state: QuantumState,
    *,
    partition: Iterable[int] | None = None,
    tol: float | None = None,
) -> bool:
    """
    Return whether the state is product with respect to a bipartition.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    partition : iterable of int | None, default=None
        Subsystems forming one side of the bipartition. The complementary
        subsystems form the other side. If omitted, and the state has exactly
        two subsystems, the partition ``[0] | [1]`` is used. If omitted for a
        multipartite state with more than two subsystems, this function tests
        whether the state is product across *some* nontrivial bipartition.
    tol : float | None, default=None
        Numerical tolerance for comparisons.

    Notes
    -----
    Definitions of “product state” vary with context. This function uses the
    following convention.

    - For a specified bipartition ``A | B``, the function tests whether
      ``rho = rho_A ⊗ rho_B``.
    - If no partition is supplied and the state has more than two subsystems,
      the function returns True if the state is product across at least one
      nontrivial bipartition.

    This is exact for the implemented criterion. In particular, for a pure
    bipartite state, productness is equivalent to purity of either reduced
    state.
    """
    use_tol = state.tol if tol is None else float(tol)
    n = state.num_subsystems

    if n == 1:
        return True

    if partition is not None:
        left = _sorted_unique_subsystems(partition, n)
        if not left:
            raise ValueError("partition must specify a nonempty proper subset.")
        right = _complement_subsystems(left, n)
        if not right:
            raise ValueError("partition must specify a nonempty proper subset.")

        left_state = state.reduced_state(left)
        right_state = state.reduced_state(right)
        return _tensor_close(left_state, right_state, state, tol=use_tol)

    if n == 2:
        left_state = state.reduced_state([0])
        right_state = state.reduced_state([1])
        return _tensor_close(left_state, right_state, state, tol=use_tol)

    subsystems = tuple(range(n))
    for r in range(1, n):
        for left in combinations(subsystems, r):
            if 0 not in left:
                continue
            right = _complement_subsystems(left, n)
            left_state = state.reduced_state(left)
            right_state = state.reduced_state(right)
            if _tensor_close(left_state, right_state, state, tol=use_tol):
                return True
    return False


@invariant_metadata(
    display_name="Entanglement property",
    notation=r"\mathrm{entangled}(\rho)",
    category="quantum state properties",
    aliases=("is entangled",),
    definition=(
        "A quantum state ρ is entangled with respect to a chosen bipartition if it is not a product state across that bipartition."
    ),
)
def is_entangled(
    state: QuantumState,
    *,
    partition: Iterable[int] | None = None,
    tol: float | None = None,
) -> bool:
    """
    Return whether the state is entangled with respect to a bipartition.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    partition : iterable of int | None, default=None
        Subsystems forming one side of the bipartition. The complementary
        subsystems form the other side. If omitted, the function uses the
        default convention described below.
    tol : float | None, default=None
        Numerical tolerance for comparisons.

    Notes
    -----
    This function is defined as the negation of ``is_product_state`` with
    respect to the chosen bipartition convention.

    Therefore:

    - if ``partition`` is specified, this tests entanglement across that
      bipartition;
    - if the state has exactly two subsystems and ``partition`` is omitted,
      this tests bipartite entanglement across ``[0] | [1]``;
    - if the state has more than two subsystems and ``partition`` is omitted,
      this returns True iff the state is not product across any nontrivial
      bipartition.

    In the multipartite case, this corresponds to a “genuinely non-product
    across every bipartition” convention, which is stronger than merely
    “not fully separable.”
    """
    return not is_product_state(state, partition=partition, tol=tol)
