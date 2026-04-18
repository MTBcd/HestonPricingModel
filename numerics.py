"""
Lightweight numerical utilities used across the reference projects.
Only NumPy is required; all core math is implemented directly here.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def ensure_even(n: int) -> int:
    """Return the next even integer greater than or equal to n."""
    return n if n % 2 == 0 else n + 1


def simpson_integrate(func: Callable[[float], complex | float], a: float, b: float, n: int = 2048) -> complex:
    """
    Composite Simpson integration on [a, b].

    Parameters
    ----------
    func:
        Real- or complex-valued integrand.
    a, b:
        Integration limits.
    n:
        Number of subintervals; automatically rounded up to an even value.
    """
    n = max(2, ensure_even(n))
    h = (b - a) / n
    x = a
    total = complex(func(a)) + complex(func(b))
    odd = 0.0j
    even = 0.0j
    for k in range(1, n):
        x = a + k * h
        if k % 2 == 0:
            even += complex(func(x))
        else:
            odd += complex(func(x))
    return h * (total + 4.0 * odd + 2.0 * even) / 3.0


def bisection_root(
    func: Callable[[float], float],
    low: float,
    high: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Bisection root finder for a continuous scalar function with a sign change.
    """
    f_low = func(low)
    f_high = func(high)
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise ValueError("Bisection requires a bracket with opposite signs.")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = func(mid)
        if abs(f_mid) < tol or 0.5 * (high - low) < tol:
            return mid
        if f_low * f_mid < 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def nelder_mead(
    objective: Callable[[np.ndarray], float],
    x0: Sequence[float],
    step: float | Sequence[float] = 0.1,
    max_iter: int = 500,
    tol: float = 1e-8,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Small pure-Python/Numpy Nelder-Mead implementation for calibration tasks.

    This is intentionally lightweight. It is suitable for reference projects,
    prototypes, and moderate-size calibration problems, but not a substitute
    for industrial optimization libraries on large surfaces.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    if np.isscalar(step):
        steps = np.full(n, float(step))
    else:
        steps = np.asarray(step, dtype=float)
        if steps.shape != x0.shape:
            raise ValueError("step must be scalar or match x0 shape")

    simplex = [x0]
    for i in range(n):
        point = x0.copy()
        point[i] = point[i] + steps[i]
        simplex.append(point)

    simplex = np.array(simplex, dtype=float)
    values = np.array([objective(p) for p in simplex], dtype=float)

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        if np.max(np.abs(simplex[1:] - simplex[0])) < tol and np.max(np.abs(values - values[0])) < tol:
            return simplex[0], float(values[0])

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        reflected = centroid + alpha * (centroid - worst)
        f_reflected = objective(reflected)

        if values[0] <= f_reflected < values[-2]:
            simplex[-1] = reflected
            values[-1] = f_reflected
            continue

        if f_reflected < values[0]:
            expanded = centroid + gamma * (reflected - centroid)
            f_expanded = objective(expanded)
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                values[-1] = f_expanded
            else:
                simplex[-1] = reflected
                values[-1] = f_reflected
            continue

        contracted = centroid + rho * (worst - centroid)
        f_contracted = objective(contracted)
        if f_contracted < values[-1]:
            simplex[-1] = contracted
            values[-1] = f_contracted
            continue

        best = simplex[0].copy()
        for i in range(1, len(simplex)):
            simplex[i] = best + sigma * (simplex[i] - best)
            values[i] = objective(simplex[i])

    order = np.argsort(values)
    simplex = simplex[order]
    values = values[order]
    return simplex[0], float(values[0])
