"""Lightweight physics-informed helpers.

This project already has a *physics-safe target* (E_star = Hs^2), which is a
nice start. This module adds two more places where you can inject physics:

1) **Post-processing constraints** (cheap, robust):
   - Energy must be non-negative.
   - Optional deep-water breaking/steepness limit using Tp.

2) **A simple energy-balance baseline** (a "physics layer" you can put in front
   of any ML model):

   dE/dt = alpha * U10^3  - beta * E

   where U10 is wind speed (m/s) and E is our proxy energy (Hs^2).

   You can fit (alpha, beta) from data and then train ML on the residuals:

   residual = E_true_next - E_phys_next

That residual-learning pattern is a super practical way to make tree models and
other non-differentiable models more physics-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional

import numpy as np

G_DEFAULT = 9.80665


def deep_water_wavelength(tp_s: np.ndarray | float, g: float = G_DEFAULT) -> np.ndarray:
    """Deep-water wavelength from period via dispersion.

    L = g * T^2 / (2*pi)

    Parameters
    ----------
    tp_s:
        Wave period in seconds.

    Returns
    -------
    np.ndarray
        Wavelength in meters.
    """

    tp = np.asarray(tp_s, dtype=float)
    return g * tp**2 / (2.0 * pi)


def steepness_limited_hmax(
    tp_s: np.ndarray | float,
    max_steepness: float = 0.14,
    g: float = G_DEFAULT,
) -> np.ndarray:
    """Very rough deep-water breaking limit: H/L <= max_steepness.

    This is not perfect ocean physics, but it prevents wildly unphysical
    predictions.
    """

    L = deep_water_wavelength(tp_s, g=g)
    return max_steepness * L


def clip_energy_physical(
    pred_E_star: np.ndarray | float,
    tp_s: Optional[np.ndarray | float] = None,
    max_steepness: float = 0.14,
    g: float = G_DEFAULT,
) -> np.ndarray:
    """Clip predicted energy proxy to simple physical bounds.

    - Enforces non-negativity.
    - Optionally enforces a steepness/breaking limit using Tp.

    Parameters
    ----------
    pred_E_star:
        Predicted energy proxy (Hs^2).
    tp_s:
        Wave period (seconds). If provided, apply steepness-based cap.

    Returns
    -------
    np.ndarray
        Clipped E_star.
    """

    E = np.asarray(pred_E_star, dtype=float)
    E = np.maximum(E, 0.0)

    if tp_s is None:
        return E

    Hmax = steepness_limited_hmax(tp_s, max_steepness=max_steepness, g=g)
    Emax = np.maximum(Hmax, 0.0) ** 2
    return np.minimum(E, Emax)


@dataclass(frozen=True)
class EnergyBalanceParams:
    """Parameters for the simple energy-balance model."""

    alpha: float
    beta: float


def fit_energy_balance_params(
    E_t: np.ndarray,
    E_next: np.ndarray,
    u10_ms: np.ndarray,
    dt_seconds: float,
) -> EnergyBalanceParams:
    """Fit alpha and beta in dE/dt = alpha*U^3 - beta*E using least squares.

    The fitted parameters are *phenomenological* (they depend on your definition
    of E_star and how you computed u10). Still, they often provide a strong
    physical baseline.

    Returns
    -------
    EnergyBalanceParams
    """

    E_t = np.asarray(E_t, dtype=float)
    E_next = np.asarray(E_next, dtype=float)
    u10_ms = np.asarray(u10_ms, dtype=float)

    # target: dE/dt
    y = (E_next - E_t) / float(dt_seconds)

    # design matrix: [U^3, -E]
    X = np.column_stack([u10_ms**3, -E_t])

    # Solve least squares: y ≈ alpha*U^3 + beta*(-E)
    # => y ≈ alpha*U^3 - beta*E
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])

    # beta should be >= 0 for dissipation; clamp tiny negatives due to noise.
    beta = max(beta, 0.0)

    return EnergyBalanceParams(alpha=alpha, beta=beta)


def energy_balance_step(
    E_t: np.ndarray | float,
    u10_ms: np.ndarray | float,
    dt_seconds: float,
    params: EnergyBalanceParams,
    use_exact: bool = True,
) -> np.ndarray:
    """One-step forecast from the energy-balance ODE.

    If beta > 0 and use_exact=True, we use the exact solution:

      E(t+dt) = E(t) * exp(-beta*dt) + (alpha*U^3/beta)*(1 - exp(-beta*dt))

    Else we fall back to an Euler step.
    """

    E0 = np.asarray(E_t, dtype=float)
    U = np.asarray(u10_ms, dtype=float)
    alpha, beta = params.alpha, params.beta
    dt = float(dt_seconds)

    if use_exact and beta > 0.0:
        decay = np.exp(-beta * dt)
        steady = (alpha * (U**3)) / beta
        E1 = E0 * decay + steady * (1.0 - decay)
    else:
        E1 = E0 + dt * (alpha * (U**3) - beta * E0)

    return np.maximum(E1, 0.0)
