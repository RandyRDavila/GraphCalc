from __future__ import annotations

from math import isclose
from typing import Iterable, Sequence

import numpy as np

from graphcalc.quantum.states import QuantumState

__all__ = ["QuantumChannel"]


class QuantumChannel:
    """
    Finite-dimensional quantum channel represented by its Choi operator.

    The channel is stored as a triple ``(J, input_dim, output_dim)`` where
    ``J`` is the Choi matrix of shape
    ``(input_dim * output_dim, input_dim * output_dim)``.

    Conventions
    -----------
    - the Choi matrix is defined by
      ``J(Phi) = sum_{i,j} |i><j| \\otimes Phi(|i><j|)``
    - the input subsystem appears first and the output subsystem second
    - complete positivity is equivalent to positive semidefiniteness of ``J``
    - trace preservation is equivalent to the partial trace over the output
      subsystem equaling the identity on the input space
    - unitality is equivalent to the partial trace over the input subsystem
      equaling the identity on the output space

    Parameters
    ----------
    choi : array-like
        Choi matrix of the channel.
    input_dim : int
        Input Hilbert-space dimension.
    output_dim : int
        Output Hilbert-space dimension.
    validate : bool, default=True
        Whether to validate the input as a completely positive trace-preserving
        map.
    tol : float, default=1e-9
        Numerical tolerance used in validation.

    Notes
    -----
    The internal Choi matrix is stored privately as ``_choi``. Public access
    is provided through a read-only property that returns a copy.
    """

    def __init__(
        self,
        choi: Sequence[Sequence[complex]] | np.ndarray,
        *,
        input_dim: int,
        output_dim: int,
        validate: bool = True,
        tol: float = 1e-9,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tol = float(tol)
        self._choi = np.array(choi, dtype=complex, copy=True)

        self._validate_parameters()

        if validate:
            self.validate()

    def __repr__(self) -> str:
        return (
            f"QuantumChannel(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"choi_rank={self.choi_rank}, tol={self.tol})"
        )

    @property
    def choi(self) -> np.ndarray:
        """Return a copy of the Choi matrix."""
        return self._choi.copy()

    @property
    def dimension(self) -> int:
        """Return the dimension of the Choi matrix."""
        return self.input_dim * self.output_dim

    @property
    def choi_rank(self) -> int:
        """Return the rank of the Choi matrix."""
        evals = np.linalg.eigvalsh(self._hermitian_part(self._choi))
        return int(np.sum(evals > self.tol))

    @classmethod
    def from_choi(
        cls,
        choi: Sequence[Sequence[complex]] | np.ndarray,
        *,
        input_dim: int,
        output_dim: int,
        validate: bool = True,
        tol: float = 1e-9,
    ) -> "QuantumChannel":
        """Construct a quantum channel from a Choi matrix."""
        return cls(
            choi,
            input_dim=input_dim,
            output_dim=output_dim,
            validate=validate,
            tol=tol,
        )

    @classmethod
    def from_kraus(
        cls,
        operators: Iterable[Sequence[Sequence[complex]] | np.ndarray],
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        validate: bool = True,
        tol: float = 1e-9,
    ) -> "QuantumChannel":
        """
        Construct a quantum channel from Kraus operators.

        Parameters
        ----------
        operators : iterable of array-like
            Kraus operators ``K_a`` of shape ``(output_dim, input_dim)``.
        input_dim : int | None, default=None
            Optional input dimension. If omitted, inferred from the operators.
        output_dim : int | None, default=None
            Optional output dimension. If omitted, inferred from the operators.
        validate : bool, default=True
            Whether to validate the resulting channel as CPTP.
        tol : float, default=1e-9
            Numerical tolerance used in validation.
        """
        kraus = [np.array(op, dtype=complex, copy=True) for op in operators]
        if not kraus:
            raise ValueError("operators must be a nonempty iterable.")

        first_shape = kraus[0].shape
        if len(first_shape) != 2:
            raise ValueError("Each Kraus operator must be a matrix.")

        out_dim0, in_dim0 = first_shape

        if input_dim is None:
            input_dim = in_dim0
        if output_dim is None:
            output_dim = out_dim0

        input_dim = int(input_dim)
        output_dim = int(output_dim)

        for op in kraus:
            if op.shape != (output_dim, input_dim):
                raise ValueError(
                    "All Kraus operators must have shape "
                    f"({output_dim}, {input_dim})."
                )

        choi = np.zeros(
            (input_dim * output_dim, input_dim * output_dim),
            dtype=complex,
        )

        for op in kraus:
            vec = op.reshape(-1, order="F")
            choi += np.outer(vec, np.conjugate(vec))

        return cls(
            choi,
            input_dim=input_dim,
            output_dim=output_dim,
            validate=validate,
            tol=tol,
        )

    @classmethod
    def identity(cls, dim: int, *, tol: float = 1e-9) -> "QuantumChannel":
        """
        Return the identity channel on a ``dim``-dimensional system.
        """
        if dim <= 0:
            raise ValueError("dim must be positive.")
        return cls.from_kraus([np.eye(dim, dtype=complex)], tol=tol)

    def copy(self) -> "QuantumChannel":
        """Return a copy of the channel."""
        return QuantumChannel(
            self._choi,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            validate=False,
            tol=self.tol,
        )

    def _validate_parameters(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if self.tol < 0:
            raise ValueError("tol must be nonnegative.")

        shape = (self.input_dim * self.output_dim, self.input_dim * self.output_dim)
        if self._choi.shape != shape:
            raise ValueError(
                f"choi must have shape {shape}, but got {self._choi.shape}."
            )

    @staticmethod
    def _hermitian_part(mat: np.ndarray) -> np.ndarray:
        return 0.5 * (mat + mat.conj().T)

    def _partial_trace_output(self) -> np.ndarray:
        """
        Return the partial trace of the Choi matrix over the output subsystem.

        The result is an ``input_dim x input_dim`` matrix.
        """
        reshaped = self._choi.reshape(
            self.input_dim,
            self.output_dim,
            self.input_dim,
            self.output_dim,
        )
        return np.trace(reshaped, axis1=1, axis2=3)

    def _partial_trace_input(self) -> np.ndarray:
        """
        Return the partial trace of the Choi matrix over the input subsystem.

        The result is an ``output_dim x output_dim`` matrix.
        """
        reshaped = self._choi.reshape(
            self.input_dim,
            self.output_dim,
            self.input_dim,
            self.output_dim,
        )
        return np.trace(reshaped, axis1=0, axis2=2)

    def validate(self) -> None:
        """
        Validate that the stored Choi matrix defines a CPTP channel.
        """
        self._validate_parameters()

        if not np.allclose(
            self._choi, self._choi.conj().T, atol=self.tol, rtol=0.0
        ):
            raise ValueError("choi must be Hermitian.")

        evals = np.linalg.eigvalsh(self._hermitian_part(self._choi))
        if np.any(evals < -self.tol):
            raise ValueError("choi must be positive semidefinite.")

        ptr_out = self._partial_trace_output()
        ident = np.eye(self.input_dim, dtype=complex)
        if not np.allclose(ptr_out, ident, atol=self.tol, rtol=0.0):
            raise ValueError(
                "Partial trace over the output subsystem must equal the "
                "identity on the input space."
            )

    def is_completely_positive(self) -> bool:
        """
        Return whether the channel is completely positive.

        Notes
        -----
        With the Choi representation, complete positivity is equivalent to
        positive semidefiniteness of the Choi matrix.
        """
        evals = np.linalg.eigvalsh(self._hermitian_part(self._choi))
        return bool(np.all(evals >= -self.tol))

    def is_trace_preserving(self) -> bool:
        """
        Return whether the channel is trace preserving.

        Notes
        -----
        For the adopted Choi convention, trace preservation is equivalent to
        ``Tr_output(J) = I_input``.
        """
        ptr_out = self._partial_trace_output()
        ident = np.eye(self.input_dim, dtype=complex)
        return bool(np.allclose(ptr_out, ident, atol=self.tol, rtol=0.0))

    def is_unital(self) -> bool:
        """
        Return whether the channel is unital.

        Notes
        -----
        For the adopted Choi convention, unitality is equivalent to
        ``Tr_input(J) = I_output``.
        """
        ptr_in = self._partial_trace_input()
        ident = np.eye(self.output_dim, dtype=complex)
        return bool(np.allclose(ptr_in, ident, atol=self.tol, rtol=0.0))

    def kraus_operators(self) -> list[np.ndarray]:
        """
        Return a Kraus representation of the channel.

        Notes
        -----
        If ``J = sum_a |K_a><K_a|`` is a spectral decomposition of the Choi
        matrix, then the Kraus operators are obtained by reshaping the
        eigenvectors into matrices using column-major order.
        """
        herm = self._hermitian_part(self._choi)
        evals, evecs = np.linalg.eigh(herm)

        ops: list[np.ndarray] = []
        for val, vec in zip(evals, evecs.T):
            if val <= self.tol:
                continue
            op = sqrt_positive(val) * vec.reshape(
                (self.output_dim, self.input_dim),
                order="F",
            )
            ops.append(op)
        return ops

    def apply(self, state: QuantumState) -> QuantumState:
        """
        Apply the channel to a quantum state.

        Parameters
        ----------
        state : QuantumState
            Input state. Its total dimension must equal ``input_dim``.

        Returns
        -------
        QuantumState
            Output state on a single subsystem of dimension ``output_dim``.

        Notes
        -----
        The action is computed from a Kraus decomposition:
        ``Phi(rho) = sum_a K_a rho K_a^dagger``.
        """
        if state.dimension != self.input_dim:
            raise ValueError(
                f"State dimension {state.dimension} does not match "
                f"channel input_dim={self.input_dim}."
            )

        out = np.zeros((self.output_dim, self.output_dim), dtype=complex)
        for op in self.kraus_operators():
            out += op @ state.rho @ op.conj().T

        return QuantumState.from_density(out, dims=(self.output_dim,), tol=state.tol)

    def compose(self, other: "QuantumChannel") -> "QuantumChannel":
        """
        Return the composition ``self ∘ other``.

        Parameters
        ----------
        other : QuantumChannel
            Channel applied first.

        Notes
        -----
        If ``other : A -> B`` and ``self : B -> C``, then the result is a
        channel ``A -> C`` defined by ``rho |-> self(other(rho))``.
        """
        if other.output_dim != self.input_dim:
            raise ValueError(
                f"Dimension mismatch: other.output_dim={other.output_dim} "
                f"must equal self.input_dim={self.input_dim}."
            )

        new_kraus = []
        for a in self.kraus_operators():
            for b in other.kraus_operators():
                new_kraus.append(a @ b)

        return QuantumChannel.from_kraus(
            new_kraus,
            input_dim=other.input_dim,
            output_dim=self.output_dim,
            validate=True,
            tol=max(self.tol, other.tol),
        )

    def tensor(self, other: "QuantumChannel") -> "QuantumChannel":
        """
        Return the tensor product channel ``self ⊗ other``.
        """
        new_kraus = []
        for a in self.kraus_operators():
            for b in other.kraus_operators():
                new_kraus.append(np.kron(a, b))

        return QuantumChannel.from_kraus(
            new_kraus,
            input_dim=self.input_dim * other.input_dim,
            output_dim=self.output_dim * other.output_dim,
            validate=True,
            tol=max(self.tol, other.tol),
        )


def sqrt_positive(x: float) -> float:
    """
    Return the square root of a nonnegative number, clipped at zero.
    """
    return float(np.sqrt(max(x, 0.0)))
