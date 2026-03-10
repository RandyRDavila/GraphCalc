from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np

from graphcalc.quantum.states import QuantumState

__all__ = [
    "basis_state",
    "computational_basis_state",
    "plus_state",
    "minus_state",
    "bell_state",
    "ghz_state",
    "w_state",
    "maximally_mixed_state",
    "werner_state",
]


def basis_state(index: int, *, dim: int = 2, tol: float = 1e-9) -> QuantumState:
    """
    Return the computational basis state ``|index><index|`` in dimension ``dim``.
    """
    return QuantumState.basis_state(index, dim=dim, tol=tol)


def computational_basis_state(bits: Sequence[int], *, tol: float = 1e-9) -> QuantumState:
    """
    Return the computational basis state indexed by a bit string.

    Parameters
    ----------
    bits : sequence of int
        Sequence of 0/1 values specifying a basis vector in ``(C^2)^{⊗ n}``.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumState``.

    Examples
    --------
    ``bits=(0, 1, 1)`` returns ``|011><011|``.
    """
    if not bits:
        raise ValueError("bits must be a nonempty sequence.")

    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Every entry of bits must be 0 or 1.")

    n = len(bits)
    dim = 2**n
    index = 0
    for bit in bits:
        index = 2 * index + bit

    ket = np.zeros(dim, dtype=complex)
    ket[index] = 1.0
    return QuantumState.from_ket(ket, dims=(2,) * n, tol=tol)


def plus_state(*, tol: float = 1e-9) -> QuantumState:
    r"""
    Return the single-qubit ``|+>`` state.

    ``|+> = (|0> + |1>) / sqrt(2)``.
    """
    ket = np.array([1.0, 1.0], dtype=complex) / sqrt(2)
    return QuantumState.from_ket(ket, dims=(2,), tol=tol)


def minus_state(*, tol: float = 1e-9) -> QuantumState:
    r"""
    Return the single-qubit ``|->`` state.

    ``|-> = (|0> - |1>) / sqrt(2)``.
    """
    ket = np.array([1.0, -1.0], dtype=complex) / sqrt(2)
    return QuantumState.from_ket(ket, dims=(2,), tol=tol)


def bell_state(which: int = 0, *, tol: float = 1e-9) -> QuantumState:
    r"""
    Return one of the four Bell states on two qubits.

    Parameters
    ----------
    which : int, default=0
        Index selecting the Bell state:
        - 0 : ``(|00> + |11>) / sqrt(2)``
        - 1 : ``(|00> - |11>) / sqrt(2)``
        - 2 : ``(|01> + |10>) / sqrt(2)``
        - 3 : ``(|01> - |10>) / sqrt(2)``
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumState``.
    """
    if which == 0:
        ket = np.array([1, 0, 0, 1], dtype=complex) / sqrt(2)
    elif which == 1:
        ket = np.array([1, 0, 0, -1], dtype=complex) / sqrt(2)
    elif which == 2:
        ket = np.array([0, 1, 1, 0], dtype=complex) / sqrt(2)
    elif which == 3:
        ket = np.array([0, 1, -1, 0], dtype=complex) / sqrt(2)
    else:
        raise ValueError("which must be one of 0, 1, 2, 3.")

    return QuantumState.from_ket(ket, dims=(2, 2), tol=tol)


def ghz_state(n: int, *, tol: float = 1e-9) -> QuantumState:
    r"""
    Return the ``n``-qubit GHZ state.

    ``|GHZ_n> = (|0...0> + |1...1>) / sqrt(2)``.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return plus_state(tol=tol)

    dim = 2**n
    ket = np.zeros(dim, dtype=complex)
    ket[0] = 1.0 / sqrt(2)
    ket[-1] = 1.0 / sqrt(2)
    return QuantumState.from_ket(ket, dims=(2,) * n, tol=tol)


def w_state(n: int, *, tol: float = 1e-9) -> QuantumState:
    r"""
    Return the ``n``-qubit W state.

    ``|W_n>`` is the equal superposition of all computational basis states
    with Hamming weight 1.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    dim = 2**n
    ket = np.zeros(dim, dtype=complex)

    for i in range(n):
        index = 1 << (n - 1 - i)
        ket[index] = 1.0

    ket /= np.linalg.norm(ket)
    return QuantumState.from_ket(ket, dims=(2,) * n, tol=tol)


def maximally_mixed_state(dim: int, *, tol: float = 1e-9) -> QuantumState:
    """
    Return the maximally mixed state ``I / dim`` on a single ``dim``-level system.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")

    rho = np.eye(dim, dtype=complex) / dim
    return QuantumState.from_density(rho, dims=(dim,), tol=tol)


def werner_state(p: float, *, tol: float = 1e-9) -> QuantumState:
    r"""
    Return the 2-qubit Werner state

    ``rho = p |Psi^-><Psi^-| + (1-p) I / 4``.

    Parameters
    ----------
    p : float
        Mixing parameter. Must satisfy ``0 <= p <= 1``.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumState``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1.")

    singlet = bell_state(3, tol=tol).rho
    mixed = np.eye(4, dtype=complex) / 4.0
    rho = p * singlet + (1.0 - p) * mixed
    return QuantumState.from_density(rho, dims=(2, 2), tol=tol)
