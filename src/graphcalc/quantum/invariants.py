# src/graphcalc/quantum/invariants.py

from __future__ import annotations

from math import log2
from typing import Iterable

import numpy as np

from graphcalc.metadata import invariant_metadata
from graphcalc.quantum.states import QuantumState

__all__ = [
    "purity",
    "rank",
    "linear_entropy",
    "von_neumann_entropy",
    "negativity",
    "logarithmic_negativity",
    "fidelity",
    "entanglement_entropy",
    "mutual_information",
]


@invariant_metadata(
    display_name="Purity",
    notation=r"\operatorname{Tr}(\rho^2)",
    category="quantum state invariants",
    aliases=("state purity",),
    definition=(
        "The purity of a quantum state ρ is Tr(ρ^2)."
    ),
)
def purity(state: QuantumState) -> float:
    """
    Return the purity ``Tr(rho^2)`` of a quantum state.
    """
    return state.purity()


@invariant_metadata(
    display_name="Rank",
    notation=r"\operatorname{rank}(\rho)",
    category="quantum state invariants",
    aliases=("state rank",),
    definition=(
        "The rank of a quantum state ρ is the rank of its density operator."
    ),
)
def rank(state: QuantumState) -> int:
    """
    Return the rank of the density operator.
    """
    return state.rank


@invariant_metadata(
    display_name="Linear entropy",
    notation=r"1 - \operatorname{Tr}(\rho^2)",
    category="quantum state invariants",
    aliases=("state linear entropy",),
    definition=(
        "The linear entropy of a quantum state ρ is 1 - Tr(ρ^2)."
    ),
)
def linear_entropy(state: QuantumState) -> float:
    """
    Return the linear entropy ``1 - Tr(rho^2)``.
    """
    return float(1.0 - purity(state))


@invariant_metadata(
    display_name="von Neumann entropy",
    notation=r"S(\rho)",
    category="quantum state invariants",
    aliases=("state entropy",),
    definition=(
        "The von Neumann entropy of a quantum state ρ is S(ρ) = -Tr(ρ log ρ)."
    ),
)
def von_neumann_entropy(state: QuantumState, *, base: float = 2.0) -> float:
    """
    Return the von Neumann entropy of a quantum state.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    base : float, default=2.0
        Logarithm base. Must be positive and not equal to 1.

    Notes
    -----
    The entropy is computed from the eigenvalues of the density operator:
    ``S(rho) = -Tr(rho log rho)``.
    """
    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and not equal to 1.")

    evals = state.eigenvalues()
    positive = evals[evals > state.tol]

    if positive.size == 0:
        return 0.0

    if base == 2.0:
        logs = np.log2(positive)
    else:
        logs = np.log(positive) / np.log(base)

    return float(-np.sum(positive * logs))


@invariant_metadata(
    display_name="Negativity",
    notation=r"N(\rho)",
    category="quantum state invariants",
    aliases=("entanglement negativity",),
    definition=(
        "The negativity of a quantum state ρ with respect to a chosen partial transpose is N(ρ) = (||ρ^Γ||_1 - 1)/2, equivalently the sum of absolute values of the negative eigenvalues of ρ^Γ."
    ),
)
def negativity(state: QuantumState, *, subsystems: Iterable[int]) -> float:
    """
    Return the entanglement negativity with respect to a partial transpose.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    subsystems : iterable of int
        Indices of subsystems on which to take the partial transpose.

    Notes
    -----
    If ``rho^Gamma`` denotes the partial transpose, then the negativity is

    ``N(rho) = (||rho^Gamma||_1 - 1) / 2``,

    equivalently the sum of absolute values of the negative eigenvalues of
    ``rho^Gamma``.
    """
    pt = state.partial_transpose(subsystems)
    evals = np.linalg.eigvalsh(0.5 * (pt.rho + pt.rho.conj().T))
    neg = -evals[evals < -state.tol].sum()
    return float(neg)


@invariant_metadata(
    display_name="Logarithmic negativity",
    notation=r"E_N(\rho)",
    category="quantum state invariants",
    aliases=("state logarithmic negativity",),
    definition=(
        "The logarithmic negativity of a quantum state ρ is E_N(ρ) = log_2 ||ρ^Γ||_1 = log_2(2N(ρ)+1)."
    ),
)
def logarithmic_negativity(state: QuantumState, *, subsystems: Iterable[int]) -> float:
    """
    Return the logarithmic negativity of a quantum state.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    subsystems : iterable of int
        Indices of subsystems on which to take the partial transpose.

    Notes
    -----
    The logarithmic negativity is

    ``E_N(rho) = log_2 ||rho^Gamma||_1 = log_2(2 N(rho) + 1)``.
    """
    n = negativity(state, subsystems=subsystems)
    return float(log2(2.0 * n + 1.0))

@invariant_metadata(
    display_name="Fidelity",
    notation=r"F(\rho,\sigma)",
    category="quantum state invariants",
    aliases=("Uhlmann fidelity",),
    definition=(
        "The fidelity between quantum states ρ and σ is F(ρ,σ) = (Tr(sqrt(sqrt(ρ) σ sqrt(ρ))))^2."
    ),
)
def fidelity(state1: QuantumState, state2: QuantumState) -> float:
    r"""
    Return the Uhlmann fidelity between two quantum states.

    Parameters
    ----------
    state1 : QuantumState
        First input state.
    state2 : QuantumState
        Second input state.

    Notes
    -----
    The fidelity is defined by

    ``F(rho, sigma) = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2``.

    This function returns a value in ``[0, 1]`` up to numerical tolerance.
    The input states must have the same total dimension.
    """
    if state1.dimension != state2.dimension:
        raise ValueError("States must have the same total dimension.")

    rho = 0.5 * (state1.rho + state1.rho.conj().T)
    sigma = 0.5 * (state2.rho + state2.rho.conj().T)

    evals, evecs = np.linalg.eigh(rho)
    evals[evals < 0.0] = 0.0
    sqrt_rho = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T

    middle = sqrt_rho @ sigma @ sqrt_rho
    middle = 0.5 * (middle + middle.conj().T)

    vals = np.linalg.eigvalsh(middle)
    vals[vals < 0.0] = 0.0

    out = float(np.sum(np.sqrt(vals)) ** 2)
    if out < 0.0 and abs(out) < max(state1.tol, state2.tol):
        return 0.0
    if out > 1.0 and abs(out - 1.0) < max(state1.tol, state2.tol):
        return 1.0
    return out


@invariant_metadata(
    display_name="Entanglement entropy",
    notation=r"S_{\mathrm{ent}}(\rho)",
    category="quantum state invariants",
    aliases=("bipartite entanglement entropy",),
    definition=(
        "For a pure quantum state ρ and a bipartition, the entanglement entropy is the von Neumann entropy of either reduced state across that bipartition."
    ),
)
def entanglement_entropy(
    state: QuantumState,
    *,
    subsystems: Iterable[int],
    base: float = 2.0,
) -> float:
    r"""
    Return the entanglement entropy across a bipartition for a pure state.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    subsystems : iterable of int
        Subsystems on one side of the bipartition.
    base : float, default=2.0
        Logarithm base used in the entropy.

    Notes
    -----
    For a pure state, the entanglement entropy across a bipartition
    ``A | A^c`` is the von Neumann entropy of the reduced state on either side.

    This function is only defined here for pure states. For mixed states,
    several inequivalent notions of entanglement measure exist, so this
    function raises ``ValueError``.
    """
    if not state.is_pure:
        raise ValueError("entanglement_entropy is only defined here for pure states.")

    reduced = state.reduced_state(subsystems)
    return von_neumann_entropy(reduced, base=base)


@invariant_metadata(
    display_name="Mutual information",
    notation=r"I(A:B)",
    category="quantum state invariants",
    aliases=("quantum mutual information",),
    definition=(
        "The quantum mutual information between subsystem collections A and B is I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_{AB})."
    ),
)
def mutual_information(
    state: QuantumState,
    *,
    subsystems_a: Iterable[int],
    subsystems_b: Iterable[int],
    base: float = 2.0,
) -> float:
    r"""
    Return the quantum mutual information between two subsystem collections.

    Parameters
    ----------
    state : QuantumState
        Input quantum state.
    subsystems_a : iterable of int
        First subsystem collection.
    subsystems_b : iterable of int
        Second subsystem collection.
    base : float, default=2.0
        Logarithm base used in the entropy.

    Notes
    -----
    The quantum mutual information is defined by

    ``I(A : B) = S(rho_A) + S(rho_B) - S(rho_AB)``,

    where ``rho_A``, ``rho_B``, and ``rho_AB`` are the reduced states on the
    specified subsystem sets.

    The subsystem sets must be disjoint.
    """
    a = tuple(sorted(set(int(i) for i in subsystems_a)))
    b = tuple(sorted(set(int(i) for i in subsystems_b)))

    if not a:
        raise ValueError("subsystems_a must be nonempty.")
    if not b:
        raise ValueError("subsystems_b must be nonempty.")
    if set(a) & set(b):
        raise ValueError("subsystems_a and subsystems_b must be disjoint.")

    n = state.num_subsystems
    if any(i < 0 or i >= n for i in a + b):
        raise ValueError("Subsystem index out of range.")

    ab = tuple(sorted(set(a) | set(b)))

    rho_a = state.reduced_state(a)
    rho_b = state.reduced_state(b)
    rho_ab = state.reduced_state(ab)

    return (
        von_neumann_entropy(rho_a, base=base)
        + von_neumann_entropy(rho_b, base=base)
        - von_neumann_entropy(rho_ab, base=base)
    )
