# Heston Stochastic Volatility Reference Project
## Model overviewThe Heston model extends Black-Scholes by making variance stochastic and correlated with spot returns.
## Mathematical details

Under the risk-neutral measure \(Q\):
```math
dS_t = (r-q) S_t dt + \sqrt{V_t} S_t dW_t^{(S)}
```
```math
dV_t = \kappa(\theta - V_t)dt + \sigma \sqrt{V_t} dW_t^{(V)}
```
```math
dW_t^{(S)} dW_t^{(V)} = \rho dt
```

Pricing uses the characteristic function of log(S_T) and Fourier inversion:
```math
C = S_0 e^{-qT} P_1 - K e^{-rT} P_2
```

The implementation includes:
1. characteristic-function pricing,
2. full-truncation Euler Monte Carlo.


## Assumptions
- Variance follows a CIR-type square-root diffusion.
- European vanilla pricing.
- Piecewise-constant parameters over the option life.
- No jumps.

## Folder structure
- `model.py`
- `numerics.py`
- `requirements.txt`
- `run_example.py`
- `README.md`

## Example run
```bash
python run_example.py
```

## Explanation of outputs
The example prints the Heston call/put from the semi-closed-form transform and compares the call to a full-truncation MonteCarlo estimate.

## Limitations
- Fourier integration still requires numerical quadrature.
- Monte Carlo uses Euler full truncation rather than exact variance sampling.
- No calibration surface is shipped in this standalone project.


## Notes on implementation style
- The project is intentionally lightweight and self-contained.
- Core formulas, integration, root-finding, and calibration logic are implemented directly in Python wherever feasible.
- NumPy is used only where it is clearly justified for numerical arrays, Monte Carlo sampling, and complex arithmetic.


