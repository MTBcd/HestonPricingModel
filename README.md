    # Heston Stochastic Volatility Reference Project

    ## Model overview
    The Heston model extends Black-Scholes by making variance stochastic and correlated with spot returns.

    ## Mathematical details
    ```text
    Under Q:
dS_t = (r-q) S_t dt + \sqrt{V_t} S_t dW_t^{(S)}
dV_t = \kappa(\theta - V_t)dt + \sigma \sqrt{V_t} dW_t^{(V)}
dW_t^{(S)} dW_t^{(V)} = \rho dt

Pricing uses the characteristic function of log(S_T) and Fourier inversion:
C = S_0 e^{-qT} P_1 - K e^{-rT} P_2

The implementation includes:
1. characteristic-function pricing,
2. full-truncation Euler Monte Carlo.
    ```

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
- `FULL_FILE_CONTENTS.md`
- `example_output.txt`


    ## Example run
    ```bash
    python run_example.py
    ```

    ### Example output
    ```text
    HESTON REFERENCE EXAMPLE
========================
Feller condition satisfied : True
Call price (CF)            : 9.242521
Put price (CF)             : 6.287074
Call price (MC)            : 9.313951
Absolute CF-MC gap         : 0.071430
    ```

    ## Explanation of outputs
    The example prints the Heston call/put from the semi-closed-form transform and compares the call to a full-truncation Monte Carlo estimate.

    ## Limitations
- Fourier integration still requires numerical quadrature.
- Monte Carlo uses Euler full truncation rather than exact variance sampling.
- No calibration surface is shipped in this small standalone project.


## Notes on implementation style
- The project is intentionally lightweight and self-contained.
- Core formulas, integration, root-finding, and calibration logic are implemented directly in Python wherever feasible.
- NumPy is used only where it is clearly justified for numerical arrays, Monte Carlo sampling, and complex arithmetic.

## Complete file contents
The full source for every file in this project is included in `FULL_FILE_CONTENTS.md`.
