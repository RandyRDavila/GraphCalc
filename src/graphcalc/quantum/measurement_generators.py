from __future__ import annotations

import numpy as np

from graphcalc.quantum.measurements import QuantumMeasurement

__all__ = [
    "computational_basis_measurement",
    "pauli_x_measurement",
    "pauli_y_measurement",
    "pauli_z_measurement",
    "bell_basis_measurement",
]


def computational_basis_measurement(*, dim: int = 2, tol: float = 1e-9) -> QuantumMeasurement:
    """
    Return the computational-basis projective measurement in dimension ``dim``.
    """
    return QuantumMeasurement.computational_basis(dim=dim, tol=tol)


def pauli_z_measurement(*, tol: float = 1e-9) -> QuantumMeasurement:
    """
    Return the qubit Pauli-Z basis measurement.

    Notes
    -----
    This is the computational-basis projective measurement with outcomes
    corresponding to ``|0><0|`` and ``|1><1|``.
    """
    return QuantumMeasurement.computational_basis(dim=2, tol=tol)


def pauli_x_measurement(*, tol: float = 1e-9) -> QuantumMeasurement:
    r"""
    Return the qubit Pauli-X basis measurement.

    Notes
    -----
    The projectors are onto the states

    ``|+> = (|0> + |1>) / sqrt(2)``
    and
    ``|-> = (|0> - |1>) / sqrt(2)``.
    """
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)

    p_plus = np.outer(plus, plus.conj())
    p_minus = np.outer(minus, minus.conj())

    return QuantumMeasurement.from_projectors([p_plus, p_minus], dim=2, tol=tol)


def pauli_y_measurement(*, tol: float = 1e-9) -> QuantumMeasurement:
    r"""
    Return the qubit Pauli-Y basis measurement.

    Notes
    -----
    The projectors are onto the states

    ``|y_+> = (|0> + i|1>) / sqrt(2)``
    and
    ``|y_-> = (|0> - i|1>) / sqrt(2)``.
    """
    y_plus = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
    y_minus = np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2.0)

    p_plus = np.outer(y_plus, y_plus.conj())
    p_minus = np.outer(y_minus, y_minus.conj())

    return QuantumMeasurement.from_projectors([p_plus, p_minus], dim=2, tol=tol)


def bell_basis_measurement(*, tol: float = 1e-9) -> QuantumMeasurement:
    r"""
    Return the two-qubit Bell-basis projective measurement.

    Notes
    -----
    The projectors are onto the four Bell states:

    - ``(|00> + |11>) / sqrt(2)``
    - ``(|00> - |11>) / sqrt(2)``
    - ``(|01> + |10>) / sqrt(2)``
    - ``(|01> - |10>) / sqrt(2)``
    """
    bell_kets = [
        np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0),
        np.array([1.0, 0.0, 0.0, -1.0], dtype=complex) / np.sqrt(2.0),
        np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0),
        np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0),
    ]
    projectors = [np.outer(ket, ket.conj()) for ket in bell_kets]
    return QuantumMeasurement.from_projectors(projectors, dim=4, tol=tol)
