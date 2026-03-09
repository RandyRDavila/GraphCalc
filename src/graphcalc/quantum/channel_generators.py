from __future__ import annotations

from typing import Sequence

import numpy as np

from graphcalc.quantum.channels import QuantumChannel

__all__ = [
    "identity_channel",
    "depolarizing_channel",
    "bit_flip_channel",
    "phase_flip_channel",
    "bit_phase_flip_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
]


def identity_channel(dim: int, *, tol: float = 1e-9) -> QuantumChannel:
    """
    Return the identity channel on a ``dim``-dimensional system.
    """
    return QuantumChannel.identity(dim, tol=tol)


def depolarizing_channel(
    p: float,
    *,
    dim: int = 2,
    tol: float = 1e-9,
) -> QuantumChannel:
    r"""
    Return the depolarizing channel on a ``dim``-dimensional system.

    Parameters
    ----------
    p : float
        Mixing parameter. Must satisfy ``0 <= p <= 1``.
    dim : int, default=2
        Hilbert-space dimension.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumChannel``.

    Notes
    -----
    This implementation uses the convention

    ``Phi(rho) = (1 - p) rho + p * Tr(rho) * I / dim``.

    For density operators with ``Tr(rho)=1``, this is

    ``Phi(rho) = (1 - p) rho + p I / dim``.

    A Kraus representation is given by

    - ``K0 = sqrt(1-p) I``
    - ``K_{ij} = sqrt(p/dim) E_{ij}`` for all ``0 <= i,j < dim``,

    where ``E_{ij}`` is the matrix unit with a 1 in position ``(i,j)`` and
    zeros elsewhere.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    kraus = [np.sqrt(1.0 - p) * np.eye(dim, dtype=complex)]

    coeff = np.sqrt(p / dim)
    for i in range(dim):
        for j in range(dim):
            eij = np.zeros((dim, dim), dtype=complex)
            eij[i, j] = 1.0
            kraus.append(coeff * eij)

    return QuantumChannel.from_kraus(kraus, input_dim=dim, output_dim=dim, tol=tol)

def bit_flip_channel(p: float, *, tol: float = 1e-9) -> QuantumChannel:
    r"""
    Return the qubit bit-flip channel.

    Parameters
    ----------
    p : float
        Flip probability. Must satisfy ``0 <= p <= 1``.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumChannel``.

    Notes
    -----
    This channel is defined by

    ``Phi(rho) = (1 - p) rho + p X rho X``,

    where ``X`` is the Pauli X operator.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1.")

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    kraus = [
        np.sqrt(1.0 - p) * I,
        np.sqrt(p) * X,
    ]
    return QuantumChannel.from_kraus(kraus, tol=tol)


def phase_flip_channel(p: float, *, tol: float = 1e-9) -> QuantumChannel:
    r"""
    Return the qubit phase-flip channel.

    Parameters
    ----------
    p : float
        Flip probability. Must satisfy ``0 <= p <= 1``.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumChannel``.

    Notes
    -----
    This channel is defined by

    ``Phi(rho) = (1 - p) rho + p Z rho Z``,

    where ``Z`` is the Pauli Z operator.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1.")

    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    kraus = [
        np.sqrt(1.0 - p) * I,
        np.sqrt(p) * Z,
    ]
    return QuantumChannel.from_kraus(kraus, tol=tol)


def bit_phase_flip_channel(p: float, *, tol: float = 1e-9) -> QuantumChannel:
    r"""
    Return the qubit bit-phase-flip channel.

    Parameters
    ----------
    p : float
        Flip probability. Must satisfy ``0 <= p <= 1``.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumChannel``.

    Notes
    -----
    This channel is defined by

    ``Phi(rho) = (1 - p) rho + p Y rho Y``,

    where ``Y`` is the Pauli Y operator.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1.")

    I = np.eye(2, dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    kraus = [
        np.sqrt(1.0 - p) * I,
        np.sqrt(p) * Y,
    ]
    return QuantumChannel.from_kraus(kraus, tol=tol)


def phase_damping_channel(gamma: float, *, tol: float = 1e-9) -> QuantumChannel:
    r"""
    Return the qubit phase-damping channel.

    Parameters
    ----------
    gamma : float
        Dephasing parameter. Must satisfy ``0 <= gamma <= 1``.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumChannel``.

    Notes
    -----
    This channel preserves populations and damps coherences:

    ``[[rho00, rho01], [rho10, rho11]] -> [[rho00, sqrt(1-gamma) rho01],
    [sqrt(1-gamma) rho10, rho11]]``.

    A Kraus representation is

    ``K0 = diag(1, sqrt(1-gamma))``,
    ``K1 = diag(0, sqrt(gamma))``.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma <= 1.")

    k0 = np.array(
        [[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]],
        dtype=complex,
    )
    k1 = np.array(
        [[0.0, 0.0], [0.0, np.sqrt(gamma)]],
        dtype=complex,
    )

    return QuantumChannel.from_kraus([k0, k1], tol=tol)


def amplitude_damping_channel(gamma: float, *, tol: float = 1e-9) -> QuantumChannel:
    r"""
    Return the qubit amplitude-damping channel.

    Parameters
    ----------
    gamma : float
        Damping parameter. Must satisfy ``0 <= gamma <= 1``.
    tol : float, default=1e-9
        Numerical tolerance passed to ``QuantumChannel``.

    Notes
    -----
    This channel models relaxation from ``|1>`` to ``|0>`` with probability
    ``gamma``.

    A Kraus representation is

    ``K0 = [[1, 0], [0, sqrt(1-gamma)]]``,
    ``K1 = [[0, sqrt(gamma)], [0, 0]]``.

    In particular,

    - ``|0><0|`` is fixed,
    - ``|1><1|`` is mapped to
      ``gamma |0><0| + (1-gamma) |1><1|``.

    This channel is trace preserving and completely positive, but in general
    it is not unital.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma <= 1.")

    k0 = np.array(
        [[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]],
        dtype=complex,
    )
    k1 = np.array(
        [[0.0, np.sqrt(gamma)], [0.0, 0.0]],
        dtype=complex,
    )

    return QuantumChannel.from_kraus([k0, k1], tol=tol)
