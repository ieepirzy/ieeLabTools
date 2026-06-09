# ieeLabTools

[![PyPI](https://img.shields.io/pypi/v/ieeLabTools)](https://pypi.org/project/ieeLabTools/)
![Python Versions](https://img.shields.io/pypi/pyversions/ieeLabTools)
![License](https://img.shields.io/pypi/l/ieeLabTools)
[![Tests](https://github.com/ieepirzy/ieeLabTools/actions/workflows/tests.yml/badge.svg)](https://github.com/ieepirzy/ieeLabTools/actions/workflows/tests.yml)

**ieeLabTools** is a Python library for automating uncertainty propagation and weighted linear regression in physics and engineering lab work. It eliminates manual symbolic differentiation and repetitive numeric error calculation.

---

## Features

| Class                | What it does                                                           |
| -------------------- | ---------------------------------------------------------------------- |
| `Yvel`               | Symbolic and numeric uncertainty propagation via partial derivatives   |
| `WeightedLinregress` | Weighted least-squares linear regression with y-axis error support     |

- Handles an arbitrary number of variables and measurements
- Returns symbolic SymPy expressions for use in lab reports
- Vectorised numeric evaluation over full measurement series (NumPy)
- Lightweight — only `sympy` and `numpy` required

---

## Installation

```bash
pip install ieeLabTools
```

---

## Quick start

### Uncertainty propagation (`Yvel`)

```python
import sympy as sp
import numpy as np
from ieeLabTools import Yvel

# Define the function symbolically
U, I = sp.symbols("U I")
R = U / I

# Instantiate – pass vars explicitly to guarantee column order
calc = Yvel(R, vars=[U, I])

# Inspect the symbolic error formula
print(calc.symbolic())
# sqrt(σI**2*U**2/I**4 + σU**2/I**2)

# Real lab data: voltage divider measurements
U_values = np.array([0.131, 0.505, 1.370, 2.944, 6.74])
I_values = np.full(5, 10e-3)          # 10 mA constant current
U_errors = np.array([6.56e-4, 2.526e-3, 6.851e-3, 1.4721e-2, 3.3701e-2])
I_errors = np.zeros(5)                # current source assumed perfect

# Stack into m×k matrices (m measurements, k variables)
values = np.column_stack([U_values, I_values])
sigmas = np.column_stack([U_errors, I_errors])

sigma_R = calc.numeric(values, sigmas)
print(sigma_R)
# [0.0656  0.2526  0.6851  1.4721  3.3701]
```

> **Column order matters.** The order of columns in `values` and `sigmas` must match the order of variables in `vars`. Always pass `vars` explicitly.

### Weighted linear regression (`WeightedLinregress`)

```python
import numpy as np
from ieeLabTools import WeightedLinregress

x     = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y     = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
y_err = np.array([0.1, 0.2, 0.1, 0.3, 0.2])   # individual y uncertainties

reg = WeightedLinregress(y_err, x, y)
slope, intercept, slope_err, intercept_err = reg.fit()

print(f"slope     = {slope:.4f} ± {slope_err:.4f}")
print(f"intercept = {intercept:.4f} ± {intercept_err:.4f}")
```

The fit minimises $\chi^2 = \sum_i \frac{(y_i - a - b x_i)^2}{\sigma_i^2}$ using the closed-form weighted least-squares solution:

$$
b = \frac{W \sum w_i x_i y_i - \sum w_i x_i \sum w_i y_i}{D}, \quad
a = \frac{\sum w_i x_i^2 \sum w_i y_i - \sum w_i x_i \sum w_i x_i y_i}{D}
$$

where $w_i = 1/\sigma_i^2$ and $D = W \sum w_i x_i^2 - \left(\sum w_i x_i\right)^2$.

---

## API reference

### `Yvel(f, vars=None)`

| Parameter | Type           | Description                                                       |
| --------- | -------------- | ----------------------------------------------------------------- |
| `f`       | `sympy.Expr`   | Function to propagate errors through                              |
| `vars`    | `list[Symbol]` | Variables in column order (recommended; auto-detected if omitted) |

| Method                       | Returns                   | Description                                    |
| ---------------------------- | ------------------------- | ---------------------------------------------- |
| `.symbolic()`                | `sympy.Expr`              | Symbolic error propagation expression          |
| `.numeric(values, sigmas)`   | `np.ndarray` shape `(m,)` | Numeric uncertainties for `m` measurement rows |

`values` and `sigmas` are both `m × k` array-likes where `m` is the number of measurements and `k` is the number of variables.

### `WeightedLinregress(y_sigma, x, y)`

| Parameter | Type | Description |
|---|---|---|
| `y_sigma` | array-like | Per-point y uncertainties |
| `x` | array-like | x-axis measurements |
| `y` | array-like | y-axis measurements |

| Method | Returns | Description |
|---|---|---|
| `.fit()` | `(slope, intercept, slope_err, intercept_err)` | Weighted least-squares fit |

---

## Mathematical background

**Non-covariant error propagation:**

$$
\sigma_f = \sqrt{\sum_{i} \left(\frac{\partial f}{\partial x_i}\right)^2 \sigma_i^2}
$$

This assumes uncorrelated measurement variables. A covariance-aware version is planned for a future release.

---

## Development

```bash
git clone https://github.com/ieepirzy/ieeLabTools
cd ieeLabTools
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Documentation

Full documentation with derivations and worked examples: [Docs/Documentation.md](Docs/Documentation.md)

---

## License

MIT — see [LICENSE](LICENSE).
