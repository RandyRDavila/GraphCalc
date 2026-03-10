# src/graphcalc/quantum/measurement_properties.py

from __future__ import annotations

import numpy as np

from graphcalc.metadata import invariant_metadata
from graphcalc.quantum.measurements import QuantumMeasurement

__all__ = [
    "is_povm",
    "is_projective_measurement",
    "is_rank_one_measurement",
]


@invariant_metadata(
    display_name="POVM property",
    notation=r"\mathrm{POVM}(M)",
    category="quantum measurement properties",
    aliases=("is POVM", "valid POVM"),
    definition=(
        "A quantum measurement M is a POVM if its effects are positive semidefinite and sum to the identity operator."
    ),
)
def is_povm(measurement: QuantumMeasurement, *, tol: float | None = None) -> bool:
    """
    Return whether the measurement is a valid POVM.

    Parameters
    ----------
    measurement : QuantumMeasurement
        Input measurement.
    tol : float | None, default=None
        Numerical tolerance used in positivity and completeness checks. If
        omitted, ``measurement.tol`` is used.

    Notes
    -----
    A POVM is a family of positive semidefinite effects ``(E_a)`` satisfying

    ``sum_a E_a = I``.

    In this module, the effects are derived from the measurement operators by
    ``E_a = M_a^dagger M_a``.
    """
    use_tol = measurement.tol if tol is None else float(tol)

    total = np.zeros((measurement.dim, measurement.dim), dtype=complex)
    for eff in measurement.effects():
        herm = 0.5 * (eff + eff.conj().T)
        evals = np.linalg.eigvalsh(herm)
        if np.any(evals < -use_tol):
            return False
        total += eff

    ident = np.eye(measurement.dim, dtype=complex)
    return bool(np.allclose(total, ident, atol=use_tol, rtol=0.0))


@invariant_metadata(
    display_name="Projective measurement property",
    notation=r"\mathrm{projective}(M)",
    category="quantum measurement properties",
    aliases=("is projective measurement",),
    definition=(
        "A quantum measurement M is projective if its effects form a family of orthogonal projections summing to the identity."
    ),
)
def is_projective_measurement(
    measurement: QuantumMeasurement,
    *,
    tol: float | None = None,
) -> bool:
    """
    Return whether the measurement is projective.

    Parameters
    ----------
    measurement : QuantumMeasurement
        Input measurement.
    tol : float | None, default=None
        Numerical tolerance used in matrix comparisons. If omitted,
        ``measurement.tol`` is used.

    Notes
    -----
    This function uses the standard finite-dimensional criterion that the
    effects form a family of orthogonal projections:

    - ``E_a = E_a^dagger``
    - ``E_a^2 = E_a``
    - ``E_a E_b = 0`` for ``a != b``
    - ``sum_a E_a = I``

    Since effects in ``QuantumMeasurement`` are already of the form
    ``M_a^dagger M_a``, this test is performed on those effects.
    """
    use_tol = measurement.tol if tol is None else float(tol)

    if not is_povm(measurement, tol=use_tol):
        return False

    effects = measurement.effects()
    zero = np.zeros((measurement.dim, measurement.dim), dtype=complex)

    for eff in effects:
        if not np.allclose(eff, eff.conj().T, atol=use_tol, rtol=0.0):
            return False
        if not np.allclose(eff @ eff, eff, atol=use_tol, rtol=0.0):
            return False

    for i, eff_i in enumerate(effects):
        for j, eff_j in enumerate(effects):
            if i == j:
                continue
            if not np.allclose(eff_i @ eff_j, zero, atol=use_tol, rtol=0.0):
                return False

    return True


@invariant_metadata(
    display_name="Rank-one measurement property",
    notation=r"\mathrm{rank\text{-}one}(M)",
    category="quantum measurement properties",
    aliases=("is rank-one measurement",),
    definition=(
        "A quantum measurement M is rank-one if every nonzero effect in the measurement has matrix rank one."
    ),
)
def is_rank_one_measurement(
    measurement: QuantumMeasurement,
    *,
    tol: float | None = None,
) -> bool:
    """
    Return whether every nonzero effect has rank one.

    Parameters
    ----------
    measurement : QuantumMeasurement
        Input measurement.
    tol : float | None, default=None
        Numerical tolerance used in rank computation. If omitted,
        ``measurement.tol`` is used.

    Notes
    -----
    In this module, a rank-one measurement means that each nonzero POVM effect
    has matrix rank one. Zero effects, if present, are ignored for this test.
    """
    use_tol = measurement.tol if tol is None else float(tol)

    for eff in measurement.effects():
        herm = 0.5 * (eff + eff.conj().T)
        evals = np.linalg.eigvalsh(herm)
        rank = int(np.sum(evals > use_tol))
        if rank not in (0, 1):
            return False

    return True
