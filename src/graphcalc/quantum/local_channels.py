from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from graphcalc.quantum.channels import QuantumChannel
from graphcalc.quantum.states import QuantumState

__all__ = [
    "apply_channel_to_subsystem",
    "apply_channels_to_subsystems",
]


def _apply_operator_to_subsystem(
    operator: np.ndarray,
    dims: Sequence[int],
    subsystem: int,
) -> np.ndarray:
    """
    Return the full operator obtained by placing ``operator`` on one subsystem
    and identities on all others.
    """
    factors = []
    for i, dim in enumerate(dims):
        if i == subsystem:
            factors.append(operator)
        else:
            factors.append(np.eye(dim, dtype=complex))

    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


def apply_channel_to_subsystem(
    state: QuantumState,
    channel: QuantumChannel,
    subsystem: int,
) -> QuantumState:
    """
    Apply a square quantum channel to one subsystem of a multipartite state.

    Parameters
    ----------
    state : QuantumState
        Input multipartite quantum state.
    channel : QuantumChannel
        Local channel to apply.
    subsystem : int
        Index of the subsystem on which the channel acts.

    Notes
    -----
    This function currently supports only square local channels, meaning
    ``channel.input_dim = channel.output_dim``. The target subsystem dimension
    must equal this common value.

    The action is implemented by lifting each Kraus operator to the full tensor
    product space and applying the resulting channel to the global density
    operator.
    """
    if not 0 <= subsystem < state.num_subsystems:
        raise ValueError("subsystem index out of range.")

    local_dim = state.dims[subsystem]
    if channel.input_dim != channel.output_dim:
        raise ValueError(
            "apply_channel_to_subsystem currently requires a square channel."
        )
    if local_dim != channel.input_dim:
        raise ValueError(
            f"Subsystem dimension {local_dim} does not match "
            f"channel input_dim={channel.input_dim}."
        )

    rho = state.rho
    out = np.zeros_like(rho, dtype=complex)

    for kraus in channel.kraus_operators():
        full_kraus = _apply_operator_to_subsystem(kraus, state.dims, subsystem)
        out += full_kraus @ rho @ full_kraus.conj().T

    return QuantumState.from_density(out, dims=state.dims, tol=state.tol)


def apply_channels_to_subsystems(
    state: QuantumState,
    channels: Sequence[QuantumChannel],
    subsystems: Sequence[int],
) -> QuantumState:
    """
    Apply local channels to distinct subsystems of a multipartite state.

    Parameters
    ----------
    state : QuantumState
        Input multipartite quantum state.
    channels : sequence of QuantumChannel
        Local channels to apply.
    subsystems : sequence of int
        Target subsystem indices.

    Notes
    -----
    Each channel is applied to the corresponding subsystem. The subsystem
    indices must be distinct.

    For the current implementation, each channel must be square and dimension-
    preserving on its target subsystem.
    """
    if len(channels) != len(subsystems):
        raise ValueError("channels and subsystems must have the same length.")

    if len(set(subsystems)) != len(subsystems):
        raise ValueError("subsystems must be distinct.")

    out = state.copy()
    for subsystem, channel in sorted(zip(subsystems, channels), key=lambda t: t[0]):
        out = apply_channel_to_subsystem(out, channel, subsystem)
    return out
