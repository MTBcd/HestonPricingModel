"""
Heston stochastic-volatility reference implementation.

Pricing uses the semi-closed-form characteristic-function approach.
Simulation uses full-truncation Euler for the variance process.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from numerics import simpson_integrate


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    rate: float
    dividend_yield: float = 0.0
    lambda_v: float = 0.0


def feller_condition(params: HestonParams) -> bool:
    return 2.0 * params.kappa * params.theta >= params.sigma**2


def _char_func(u: complex, spot: float, maturity: float, params: HestonParams) -> complex:
    """
    Characteristic function of log(S_T) under the risk-neutral Heston model.

    This implementation uses the stable "little trap" style parametrization
    with exp(-d T), which is algebraically equivalent to the original Heston
    formula but numerically more stable.
    """
    if maturity <= 0.0:
        return np.exp(1j * u * np.log(spot))
    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma
    rho = params.rho
    v0 = params.v0
    r = params.rate
    q = params.dividend_yield

    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma**2 * (u**2 + iu))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    exp_dt = np.exp(-d * maturity)

    C = (
        iu * (np.log(spot) + (r - q) * maturity)
        + (kappa * theta / sigma**2)
        * ((kappa - rho * sigma * iu - d) * maturity - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
    )
    D = ((kappa - rho * sigma * iu - d) / sigma**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.exp(C + D * v0)


def _probability_p2(spot: float, strike: float, maturity: float, params: HestonParams, upper: float, n: int) -> float:
    log_k = math.log(strike)

    def integrand(phi: float) -> float:
        u = complex(phi, 0.0)
        value = np.exp(-1j * u * log_k) * _char_func(u, spot, maturity, params) / (1j * u)
        return float(np.real(value))

    integral = simpson_integrate(integrand, 1e-8, upper, n=n).real
    return 0.5 + integral / math.pi


def _probability_p1(spot: float, strike: float, maturity: float, params: HestonParams, upper: float, n: int) -> float:
    log_k = math.log(strike)
    phi_minus_i = _char_func(-1j, spot, maturity, params)

    def integrand(phi: float) -> float:
        u = complex(phi, 0.0)
        numerator = np.exp(-1j * u * log_k) * _char_func(u - 1j, spot, maturity, params)
        value = numerator / (1j * u * phi_minus_i)
        return float(np.real(value))

    integral = simpson_integrate(integrand, 1e-8, upper, n=n).real
    return 0.5 + integral / math.pi


def call_price_cf(
    spot: float,
    strike: float,
    maturity: float,
    params: HestonParams,
    integration_upper: float = 150.0,
    n: int = 4096,
) -> float:
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive")

    p1 = _probability_p1(spot, strike, maturity, params, integration_upper, n)
    p2 = _probability_p2(spot, strike, maturity, params, integration_upper, n)
    discounted_spot = spot * math.exp(-params.dividend_yield * maturity)
    discounted_strike = strike * math.exp(-params.rate * maturity)
    return discounted_spot * p1 - discounted_strike * p2


def put_price_cf(
    spot: float,
    strike: float,
    maturity: float,
    params: HestonParams,
    integration_upper: float = 150.0,
    n: int = 4096,
) -> float:
    call = call_price_cf(spot, strike, maturity, params, integration_upper, n)
    return call - spot * math.exp(-params.dividend_yield * maturity) + strike * math.exp(-params.rate * maturity)


def simulate_paths(
    spot: float,
    maturity: float,
    steps: int,
    n_paths: int,
    params: HestonParams,
    seed: int = 123,
    risk_neutral: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if spot <= 0.0 or maturity <= 0.0 or steps <= 0 or n_paths <= 0:
        raise ValueError("spot, maturity, steps, and n_paths must be positive")

    dt = maturity / steps
    drift = params.rate - params.dividend_yield if risk_neutral else params.rate
    rng = np.random.default_rng(seed)

    s = np.empty((n_paths, steps + 1), dtype=float)
    v = np.empty((n_paths, steps + 1), dtype=float)
    s[:, 0] = spot
    v[:, 0] = params.v0

    sqrt_dt = math.sqrt(dt)
    for k in range(steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w_s = z1
        w_v = params.rho * z1 + math.sqrt(max(1.0 - params.rho**2, 0.0)) * z2

        v_prev = np.maximum(v[:, k], 0.0)
        v_next = v[:, k] + params.kappa * (params.theta - v_prev) * dt + params.sigma * np.sqrt(v_prev) * sqrt_dt * w_v
        v[:, k + 1] = np.maximum(v_next, 0.0)

        s[:, k + 1] = s[:, k] * np.exp((drift - 0.5 * v_prev) * dt + np.sqrt(v_prev) * sqrt_dt * w_s)

    return s, v


def call_price_mc(
    spot: float,
    strike: float,
    maturity: float,
    params: HestonParams,
    steps: int = 252,
    n_paths: int = 50000,
    seed: int = 123,
) -> float:
    s, _ = simulate_paths(spot, maturity, steps, n_paths, params, seed=seed, risk_neutral=True)
    payoff = np.maximum(s[:, -1] - strike, 0.0)
    return math.exp(-params.rate * maturity) * float(payoff.mean())
