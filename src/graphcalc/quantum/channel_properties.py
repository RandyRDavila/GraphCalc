# src/graphcalc/quantum/channel_properties.py

from __future__ import annotations

from math import isclose

import numpy as np

from graphcalc.metadata import invariant_metadata
from graphcalc.quantum.channels import QuantumChannel

__all__ = [
    "is_completely_positive",
    "is_trace_preserving",
    "is_unital",
    "is_quantum_channel",
    "is_unitary_channel",
]


@invariant_metadata(
    display_name="Complete positivity",
    notation=r"\mathrm{CP}(\Phi)",
    category="quantum channel properties",
    aliases=("is completely positive",),
    definition=(
        "A linear map Φ is completely positive if its Choi matrix J(Φ) is positive semidefinite."
    ),
)
def is_completely_positive(channel: QuantumChannel, *, tol: float | None = None) -> bool:
    """
    Return whether the channel is completely positive.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.
    tol : float | None, default=None
        Numerical tolerance used for positive semidefiniteness. If omitted,
        ``channel.tol`` is used.

    Notes
    -----
    By the Choi theorem, a linear map is completely positive if and only if
    its Choi matrix is positive semidefinite.
    """
    use_tol = channel.tol if tol is None else float(tol)
    choi = channel.choi
    herm = 0.5 * (choi + choi.conj().T)
    evals = np.linalg.eigvalsh(herm)
    return bool(np.all(evals >= -use_tol))


@invariant_metadata(
    display_name="Trace-preserving property",
    notation=r"\mathrm{TP}(\Phi)",
    category="quantum channel properties",
    aliases=("is trace preserving",),
    definition=(
        "A linear map Φ is trace preserving if it preserves the trace of every input operator."
    ),
)
def is_trace_preserving(channel: QuantumChannel, *, tol: float | None = None) -> bool:
    """
    Return whether the channel is trace preserving.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.
    tol : float | None, default=None
        Numerical tolerance for matrix comparison. If omitted, ``channel.tol``
        is used.

    Notes
    -----
    For the Choi convention used in ``QuantumChannel``, trace preservation is
    equivalent to the partial trace over the output subsystem equaling the
    identity on the input space.
    """
    use_tol = channel.tol if tol is None else float(tol)
    ptr_out = channel._partial_trace_output()
    ident = np.eye(channel.input_dim, dtype=complex)
    return bool(np.allclose(ptr_out, ident, atol=use_tol, rtol=0.0))


@invariant_metadata(
    display_name="Unital property",
    notation=r"\mathrm{unital}(\Phi)",
    category="quantum channel properties",
    aliases=("is unital",),
    definition=(
        "A linear map Φ is unital if it preserves the identity operator."
    ),
)
def is_unital(channel: QuantumChannel, *, tol: float | None = None) -> bool:
    """
    Return whether the channel is unital.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.
    tol : float | None, default=None
        Numerical tolerance for matrix comparison. If omitted, ``channel.tol``
        is used.

    Notes
    -----
    A channel is unital if it preserves the identity operator. For the Choi
    convention used in ``QuantumChannel``, this is equivalent to the partial
    trace over the input subsystem equaling the identity on the output space.
    """
    use_tol = channel.tol if tol is None else float(tol)
    ptr_in = channel._partial_trace_input()
    ident = np.eye(channel.output_dim, dtype=complex)
    return bool(np.allclose(ptr_in, ident, atol=use_tol, rtol=0.0))


@invariant_metadata(
    display_name="Quantum channel property",
    notation=r"\mathrm{CPTP}(\Phi)",
    category="quantum channel properties",
    aliases=("is quantum channel", "is CPTP"),
    definition=(
        "A quantum channel is a completely positive trace-preserving linear map."
    ),
)
def is_quantum_channel(channel: QuantumChannel, *, tol: float | None = None) -> bool:
    """
    Return whether the input defines a completely positive trace-preserving map.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.
    tol : float | None, default=None
        Numerical tolerance used in the underlying tests.

    Notes
    -----
    In this module, “quantum channel” means a completely positive,
    trace-preserving linear map.
    """
    return is_completely_positive(channel, tol=tol) and is_trace_preserving(
        channel, tol=tol
    )


@invariant_metadata(
    display_name="Unitary channel property",
    notation=r"\mathrm{unitary}(\Phi)",
    category="quantum channel properties",
    aliases=("is unitary channel",),
    definition=(
        "A quantum channel Φ is unitary if there exists a unitary operator U such that Φ(ρ) = UρU^† for all input states ρ."
    ),
)
def is_unitary_channel(channel: QuantumChannel, *, tol: float | None = None) -> bool:
    """
    Return whether the channel is a unitary channel.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.
    tol : float | None, default=None
        Numerical tolerance used in rank and matrix comparisons. If omitted,
        ``channel.tol`` is used.

    Notes
    -----
    A channel is a unitary channel if it has the form

    ``Phi(rho) = U rho U^dagger``

    for some unitary matrix ``U``.

    For finite-dimensional channels, this is equivalent to the existence of a
    Kraus representation with a single Kraus operator that is unitary.
    With the Choi representation, a necessary condition is that the Choi matrix
    has rank one. For a CPTP square channel, rank one is sufficient to recover
    a single Kraus operator, which is then tested for unitarity.
    """
    use_tol = channel.tol if tol is None else float(tol)

    if channel.input_dim != channel.output_dim:
        return False

    if not is_quantum_channel(channel, tol=use_tol):
        return False

    if channel.choi_rank != 1:
        return False

    kraus = channel.kraus_operators()
    if len(kraus) != 1:
        return False

    op = kraus[0]
    ident = np.eye(channel.input_dim, dtype=complex)

    return bool(
        np.allclose(op.conj().T @ op, ident, atol=use_tol, rtol=0.0)
        and np.allclose(op @ op.conj().T, ident, atol=use_tol, rtol=0.0)
    )
