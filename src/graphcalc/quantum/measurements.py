from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from graphcalc.quantum.states import QuantumState

__all__ = ["QuantumMeasurement"]


class QuantumMeasurement:
    """
    Finite quantum measurement represented by measurement operators.

    The measurement is stored as a family ``(M_a)`` of operators on a fixed
    finite-dimensional Hilbert space. The corresponding effects are
    ``E_a = M_a^dagger M_a``.

    Conventions
    -----------
    - all measurement operators act on the same Hilbert space
    - the measurement is complete when ``sum_a E_a = I``
    - projective measurements are included as a special case
    - post-measurement states are normalized conditional states

    Parameters
    ----------
    operators : iterable of array-like
        Measurement operators.
    dim : int | None, default=None
        Hilbert-space dimension. If omitted, inferred from the operators.
    validate : bool, default=True
        Whether to validate the measurement as complete.
    tol : float, default=1e-9
        Numerical tolerance used in validation.
    """

    def __init__(
        self,
        operators: Iterable[Sequence[Sequence[complex]] | np.ndarray],
        *,
        dim: int | None = None,
        validate: bool = True,
        tol: float = 1e-9,
    ) -> None:
        self.tol = float(tol)
        self._operators = [np.array(op, dtype=complex, copy=True) for op in operators]

        if not self._operators:
            raise ValueError("operators must be a nonempty iterable.")

        first_shape = self._operators[0].shape
        if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
            raise ValueError("Each measurement operator must be a square matrix.")

        inferred_dim = first_shape[0]
        self.dim = inferred_dim if dim is None else int(dim)

        self._validate_parameters()

        if validate:
            self.validate()

    def __repr__(self) -> str:
        return (
            f"QuantumMeasurement(num_outcomes={self.num_outcomes}, "
            f"dim={self.dim}, tol={self.tol})"
        )

    @property
    def operators(self) -> tuple[np.ndarray, ...]:
        """Return copies of the measurement operators."""
        return tuple(op.copy() for op in self._operators)

    @property
    def num_outcomes(self) -> int:
        """Return the number of outcomes."""
        return len(self._operators)

    @classmethod
    def from_projectors(
        cls,
        projectors: Iterable[Sequence[Sequence[complex]] | np.ndarray],
        *,
        dim: int | None = None,
        validate: bool = True,
        tol: float = 1e-9,
    ) -> "QuantumMeasurement":
        """
        Construct a projective measurement from projectors.

        Notes
        -----
        For projective measurements, the measurement operators and effects
        coincide.
        """
        return cls(projectors, dim=dim, validate=validate, tol=tol)

    @classmethod
    def computational_basis(
        cls,
        *,
        dim: int = 2,
        tol: float = 1e-9,
    ) -> "QuantumMeasurement":
        """
        Return the computational-basis projective measurement in dimension ``dim``.
        """
        if dim <= 0:
            raise ValueError("dim must be positive.")

        ops = []
        for i in range(dim):
            proj = np.zeros((dim, dim), dtype=complex)
            proj[i, i] = 1.0
            ops.append(proj)

        return cls.from_projectors(ops, dim=dim, tol=tol)

    def _validate_parameters(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if self.tol < 0:
            raise ValueError("tol must be nonnegative.")

        for op in self._operators:
            if len(op.shape) != 2 or op.shape != (self.dim, self.dim):
                raise ValueError(
                    f"Each measurement operator must have shape ({self.dim}, {self.dim})."
                )

    def effects(self) -> tuple[np.ndarray, ...]:
        """Return the POVM effects ``M_a^dagger M_a``."""
        return tuple(op.conj().T @ op for op in self._operators)

    def validate(self) -> None:
        """
        Validate that the measurement is complete.

        Notes
        -----
        Completeness means that the effects satisfy ``sum_a E_a = I``.
        """
        self._validate_parameters()

        total = np.zeros((self.dim, self.dim), dtype=complex)
        for eff in self.effects():
            herm = 0.5 * (eff + eff.conj().T)
            evals = np.linalg.eigvalsh(herm)
            if np.any(evals < -self.tol):
                raise ValueError("Each effect must be positive semidefinite.")
            total += eff

        ident = np.eye(self.dim, dtype=complex)
        if not np.allclose(total, ident, atol=self.tol, rtol=0.0):
            raise ValueError("Measurement effects must sum to the identity.")

    def outcome_probability(self, state: QuantumState, outcome: int) -> float:
        """
        Return the probability of a specified outcome.
        """
        if state.dimension != self.dim:
            raise ValueError(
                f"State dimension {state.dimension} does not match measurement dim={self.dim}."
            )
        if not 0 <= outcome < self.num_outcomes:
            raise ValueError("Outcome index out of range.")

        eff = self.effects()[outcome]
        prob = np.trace(eff @ state.rho)
        return float(prob.real)

    def outcome_probabilities(self, state: QuantumState) -> np.ndarray:
        """
        Return the vector of outcome probabilities.
        """
        probs = np.array(
            [self.outcome_probability(state, i) for i in range(self.num_outcomes)],
            dtype=float,
        )
        probs[np.abs(probs) < self.tol] = 0.0
        return probs

    def post_measurement_state(
        self,
        state: QuantumState,
        outcome: int,
    ) -> QuantumState:
        """
        Return the normalized post-measurement state conditioned on an outcome.

        Notes
        -----
        If the outcome probability is zero, this function raises ``ValueError``.
        """
        if state.dimension != self.dim:
            raise ValueError(
                f"State dimension {state.dimension} does not match measurement dim={self.dim}."
            )
        if not 0 <= outcome < self.num_outcomes:
            raise ValueError("Outcome index out of range.")

        op = self._operators[outcome]
        prob = self.outcome_probability(state, outcome)
        if prob <= self.tol:
            raise ValueError("Post-measurement state is undefined for zero-probability outcome.")

        rho = op @ state.rho @ op.conj().T / prob
        return QuantumState.from_density(rho, dims=(self.dim,), tol=state.tol)
