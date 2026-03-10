from __future__ import annotations

from math import isclose, prod, sqrt
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["SubsystemDims", "QuantumState"]

SubsystemDims = Tuple[int, ...]


class QuantumState:
    """
    Finite-dimensional multipartite quantum state represented by a density operator.

    The state is stored as a pair ``(rho, dims)`` where ``rho`` is a square
    complex matrix of size ``d x d`` with ``d = prod(dims)``, and ``dims``
    records the subsystem dimensions.

    Conventions
    -----------
    - states are finite-dimensional
    - density operators are Hermitian, positive semidefinite, and trace one
    - pure states may be constructed from ket vectors but are stored internally
      as density operators
    - subsystem structure is explicit through ``dims``

    Parameters
    ----------
    rho : array-like
        Density matrix representation of the state.
    dims : tuple of int
        Dimensions of the subsystems.
    validate : bool, default=True
        Whether to validate the input as a density operator.
    tol : float, default=1e-9
        Numerical tolerance used in validation.

    Notes
    -----
    The internal density matrix is stored privately as ``_rho``. Public access
    is provided through a read-only property that returns a copy.
    """

    def __init__(
        self,
        rho: Sequence[Sequence[complex]] | np.ndarray,
        *,
        dims: Sequence[int],
        validate: bool = True,
        tol: float = 1e-9,
    ) -> None:
        self.tol = float(tol)
        self._dims: SubsystemDims = tuple(int(d) for d in dims)
        self._rho = np.array(rho, dtype=complex, copy=True)

        self._validate_parameters()

        if validate:
            self.validate()

    def __repr__(self) -> str:
        return (
            f"QuantumState(num_subsystems={self.num_subsystems}, "
            f"dimension={self.dimension}, is_pure={self.is_pure}, "
            f"rank={self.rank}, tol={self.tol})"
        )

    @property
    def rho(self) -> np.ndarray:
        """Return a copy of the density matrix."""
        return self._rho.copy()

    @property
    def dims(self) -> SubsystemDims:
        """Return the subsystem dimensions."""
        return self._dims

    @property
    def num_subsystems(self) -> int:
        """Return the number of subsystems."""
        return len(self._dims)

    @property
    def dimension(self) -> int:
        """Return the total Hilbert-space dimension."""
        return prod(self._dims)

    @property
    def trace(self) -> complex:
        """Return the trace of the density matrix."""
        return complex(np.trace(self._rho))

    @property
    def rank(self) -> int:
        """Return the matrix rank of the density operator."""
        evals = np.linalg.eigvalsh(self._hermitian_part(self._rho))
        return int(np.sum(evals > self.tol))

    @property
    def is_pure(self) -> bool:
        """Return True iff the state is pure."""
        return isclose(self.purity(), 1.0, rel_tol=0.0, abs_tol=self.tol)

    @property
    def is_mixed(self) -> bool:
        """Return True iff the state is mixed."""
        return not self.is_pure

    @classmethod
    def from_density(
        cls,
        rho: Sequence[Sequence[complex]] | np.ndarray,
        *,
        dims: Sequence[int],
        validate: bool = True,
        tol: float = 1e-9,
    ) -> "QuantumState":
        """Construct a quantum state from a density matrix."""
        return cls(rho, dims=dims, validate=validate, tol=tol)

    @classmethod
    def from_ket(
        cls,
        ket: Sequence[complex] | np.ndarray,
        *,
        dims: Sequence[int],
        normalize: bool = True,
        tol: float = 1e-9,
    ) -> "QuantumState":
        """
        Construct a quantum state from a ket vector.

        Parameters
        ----------
        ket : array-like
            State vector.
        dims : tuple of int
            Dimensions of the subsystems.
        normalize : bool, default=True
            Whether to normalize the ket before constructing the density matrix.
        tol : float, default=1e-9
            Numerical tolerance used in validation.
        """
        psi = np.array(ket, dtype=complex, copy=True).reshape(-1)
        dim = prod(int(d) for d in dims)

        if psi.shape != (dim,):
            raise ValueError(
                f"Ket has length {psi.shape[0]}, but prod(dims) = {dim}."
            )

        norm = np.linalg.norm(psi)
        if norm <= tol:
            raise ValueError("Ket vector must be nonzero.")

        if normalize:
            psi = psi / norm

        rho = np.outer(psi, np.conjugate(psi))
        return cls(rho, dims=dims, validate=True, tol=tol)

    @classmethod
    def basis_state(
        cls,
        index: int,
        *,
        dim: int = 2,
        tol: float = 1e-9,
    ) -> "QuantumState":
        """Construct a computational basis state ``|index><index|`` in dimension ``dim``."""
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if not 0 <= index < dim:
            raise ValueError(f"index must satisfy 0 <= index < {dim}.")

        ket = np.zeros(dim, dtype=complex)
        ket[index] = 1.0
        return cls.from_ket(ket, dims=(dim,), tol=tol)

    def copy(self) -> "QuantumState":
        """Return a copy of the quantum state."""
        return QuantumState(self._rho, dims=self._dims, validate=False, tol=self.tol)

    def _validate_parameters(self) -> None:
        if not self._dims:
            raise ValueError("dims must be a nonempty tuple of positive integers.")
        if any(d <= 0 for d in self._dims):
            raise ValueError("Every subsystem dimension must be positive.")
        if self.tol < 0:
            raise ValueError("tol must be nonnegative.")

        dim = self.dimension
        if self._rho.shape != (dim, dim):
            raise ValueError(
                f"rho must have shape ({dim}, {dim}) to match dims={self._dims}, "
                f"but got {self._rho.shape}."
            )

    @staticmethod
    def _hermitian_part(mat: np.ndarray) -> np.ndarray:
        return 0.5 * (mat + mat.conj().T)

    def validate(self) -> None:
        """Validate that the stored matrix is a density operator."""
        self._validate_parameters()

        if not np.allclose(self._rho, self._rho.conj().T, atol=self.tol, rtol=0.0):
            raise ValueError("rho must be Hermitian.")

        tr = np.trace(self._rho)
        if not isclose(float(tr.real), 1.0, rel_tol=0.0, abs_tol=self.tol) or abs(tr.imag) > self.tol:
            raise ValueError("rho must have trace 1.")

        evals = np.linalg.eigvalsh(self._hermitian_part(self._rho))
        if np.any(evals < -self.tol):
            raise ValueError("rho must be positive semidefinite.")

    def purity(self) -> float:
        """Return Tr(rho^2)."""
        value = np.trace(self._rho @ self._rho)
        return float(value.real)

    def eigenvalues(self) -> np.ndarray:
        """Return the eigenvalues of the density operator."""
        evals = np.linalg.eigvalsh(self._hermitian_part(self._rho))
        evals[np.abs(evals) < self.tol] = 0.0
        return evals

    def tensor(self, other: "QuantumState") -> "QuantumState":
        """Return the tensor product of this state with another state."""
        rho = np.kron(self._rho, other._rho)
        dims = self._dims + other._dims
        tol = max(self.tol, other.tol)
        return QuantumState(rho, dims=dims, validate=False, tol=tol)

    def partial_trace(self, subsystems: Iterable[int]) -> "QuantumState":
        """
        Trace out the given subsystems.

        Parameters
        ----------
        subsystems : iterable of int
            Indices of subsystems to trace out.

        Returns
        -------
        QuantumState
            The reduced state on the remaining subsystems.
        """
        trace_out = tuple(sorted(set(int(i) for i in subsystems)))
        n = self.num_subsystems

        if any(i < 0 or i >= n for i in trace_out):
            raise ValueError("Subsystem index out of range.")

        keep = tuple(i for i in range(n) if i not in trace_out)

        if not keep:
            reduced = np.array([[complex(np.trace(self._rho))]], dtype=complex)
            return QuantumState(reduced, dims=(1,), validate=True, tol=self.tol)

        dims = self._dims
        reshaped = self._rho.reshape(*dims, *dims)

        current_n = n
        for ax in sorted(trace_out, reverse=True):
            reshaped = np.trace(reshaped, axis1=ax, axis2=ax + current_n)
            current_n -= 1

        keep_dims = tuple(dims[i] for i in keep)
        reduced = reshaped.reshape(prod(keep_dims), prod(keep_dims))
        return QuantumState(reduced, dims=keep_dims, validate=True, tol=self.tol)

    def reduced_state(self, subsystems: Iterable[int]) -> "QuantumState":
        """
        Return the reduced state on the given subsystems.

        Parameters
        ----------
        subsystems : iterable of int
            Indices of subsystems to keep.
        """
        keep = tuple(sorted(set(int(i) for i in subsystems)))
        n = self.num_subsystems

        if any(i < 0 or i >= n for i in keep):
            raise ValueError("Subsystem index out of range.")

        trace_out = tuple(i for i in range(n) if i not in keep)
        return self.partial_trace(trace_out)

    def partial_transpose(self, subsystems: Iterable[int]) -> "QuantumState":
        """
        Return the partial transpose with respect to the given subsystems.

        Parameters
        ----------
        subsystems : iterable of int
            Indices of subsystems on which to apply matrix transposition.

        Notes
        -----
        The partial transpose of a valid quantum state need not be positive
        semidefinite, so the returned operator is constructed with
        ``validate=False``.
        """
        transpose_on = tuple(sorted(set(int(i) for i in subsystems)))
        n = self.num_subsystems

        if any(i < 0 or i >= n for i in transpose_on):
            raise ValueError("Subsystem index out of range.")

        dims = self._dims
        reshaped = self._rho.reshape(*dims, *dims)

        axes = list(range(2 * n))
        for i in transpose_on:
            axes[i], axes[i + n] = axes[i + n], axes[i]

        transposed = np.transpose(reshaped, axes=axes)
        mat = transposed.reshape(self.dimension, self.dimension)
        return QuantumState(mat, dims=self._dims, validate=False, tol=self.tol)

    def subsystem_dimensions(self, subsystems):
        """
        Return the dimensions of the specified subsystems.
        """
        subsystems = tuple(int(i) for i in subsystems)
        n = self.num_subsystems
        if any(i < 0 or i >= n for i in subsystems):
            raise ValueError("Subsystem index out of range.")
        return tuple(self._dims[i] for i in subsystems)
