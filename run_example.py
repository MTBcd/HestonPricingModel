from __future__ import annotations

from model import HestonParams, call_price_cf, call_price_mc, feller_condition, put_price_cf


def main() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        sigma=0.30,
        rho=-0.70,
        v0=0.04,
        rate=0.03,
        dividend_yield=0.0,
    )
    spot = 100.0
    strike = 100.0
    maturity = 1.0

    call_cf = call_price_cf(spot, strike, maturity, params, integration_upper=120.0, n=4096)
    put_cf = put_price_cf(spot, strike, maturity, params, integration_upper=120.0, n=4096)
    call_mc = call_price_mc(spot, strike, maturity, params, steps=252, n_paths=40000, seed=17)

    print("HESTON REFERENCE EXAMPLE")
    print("========================")
    print(f"Feller condition satisfied : {feller_condition(params)}")
    print(f"Call price (CF)            : {call_cf:.6f}")
    print(f"Put price (CF)             : {put_cf:.6f}")
    print(f"Call price (MC)            : {call_mc:.6f}")
    print(f"Absolute CF-MC gap         : {abs(call_cf - call_mc):.6f}")


if __name__ == "__main__":
    main()
