# src/graphcalc/quantum/channel_invariants.py

from __future__ import annotations

import numpy as np

from graphcalc.metadata import invariant_metadata
from graphcalc.quantum.channels import QuantumChannel

__all__ = [
    "choi_rank",
    "kraus_rank",
    "input_dimension",
    "output_dimension",
    "choi_eigenvalues",
]


@invariant_metadata(
    display_name="Choi rank",
    notation=r"\operatorname{rank}(J(\Phi))",
    category="quantum channel invariants",
    aliases=("channel choi rank",),
    definition=(
        "The Choi rank of a quantum channel Φ is the rank of its Choi matrix J(Φ)."
    ),
)
def choi_rank(channel: QuantumChannel) -> int:
    """
    Return the rank of the Choi matrix.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.

    Notes
    -----
    This is the matrix rank of the Choi operator, computed with the numerical
    tolerance built into the channel object.
    """
    return channel.choi_rank


@invariant_metadata(
    display_name="Kraus rank",
    notation=r"r_K(\Phi)",
    category="quantum channel invariants",
    aliases=("channel kraus rank",),
    definition=(
        "The Kraus rank of a quantum channel Φ is the minimum number of Kraus operators needed to represent Φ; equivalently, it is the rank of the Choi matrix of Φ."
    ),
)
def kraus_rank(channel: QuantumChannel) -> int:
    """
    Return the Kraus rank of the channel.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.

    Notes
    -----
    For finite-dimensional completely positive maps, the Kraus rank equals the
    rank of the Choi matrix. This function is therefore an alias for
    ``choi_rank(channel)`` provided for mathematical convenience.
    """
    return channel.choi_rank


@invariant_metadata(
    display_name="Input dimension",
    notation=r"d_{\mathrm{in}}(\Phi)",
    category="quantum channel invariants",
    aliases=("channel input dimension",),
    definition=(
        "The input dimension of a quantum channel Φ is the dimension of its input Hilbert space."
    ),
)
def input_dimension(channel: QuantumChannel) -> int:
    """
    Return the input Hilbert-space dimension of the channel.
    """
    return channel.input_dim


@invariant_metadata(
    display_name="Output dimension",
    notation=r"d_{\mathrm{out}}(\Phi)",
    category="quantum channel invariants",
    aliases=("channel output dimension",),
    definition=(
        "The output dimension of a quantum channel Φ is the dimension of its output Hilbert space."
    ),
)
def output_dimension(channel: QuantumChannel) -> int:
    """
    Return the output Hilbert-space dimension of the channel.
    """
    return channel.output_dim


@invariant_metadata(
    display_name="Choi eigenvalues",
    notation=r"\operatorname{spec}(J(\Phi))",
    category="quantum channel invariants",
    aliases=("eigenvalues of choi matrix",),
    definition=(
        "The Choi eigenvalues of a quantum channel Φ are the eigenvalues of its Choi matrix J(Φ); numerically, this function computes the eigenvalues of the Hermitian part of J(Φ) and rounds values within tolerance to zero."
    ),
)
def choi_eigenvalues(channel: QuantumChannel) -> np.ndarray:
    """
    Return the eigenvalues of the Hermitian part of the Choi matrix.

    Parameters
    ----------
    channel : QuantumChannel
        Input channel.

    Notes
    -----
    For completely positive maps, these are the eigenvalues of the Choi matrix.
    Very small values within the channel tolerance are rounded to zero.
    """
    choi = channel.choi
    herm = 0.5 * (choi + choi.conj().T)
    evals = np.linalg.eigvalsh(herm)
    evals[np.abs(evals) < channel.tol] = 0.0
    return evals
