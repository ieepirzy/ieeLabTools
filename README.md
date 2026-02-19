# ieeLabTools


[![PyPI](https://img.shields.io/pypi/v/ieeLabTools)](https://pypi.org/project/ieeLabTools/)
![Python Versions](https://img.shields.io/pypi/pyversions/ieeLabTools)
![License](https://img.shields.io/pypi/l/ieeLabTools)

Tools for **laboratory data analysis**, including:

- **General symbolic & numeric uncertainty propagation**
- **Weighted linear regression** (uncertainties in `y`)
- Designed for physics, engineering, and other quantitative lab work

This library is part of the **PhySiLight-Tools** ecosystem.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| `Yvel` | Propagate measurement uncertainties using partial derivatives |
| Symbolic mode | Generates algebraic uncertainty expressions via SymPy |
| Numeric mode | Evaluates uncertainty for data arrays of any length |
| LaTeX output(WIP) | Pretty-print formulas for lab reports |
| WeightedLinearRegression | Weighted least-squares fit (supports `y`-errors) |
| ODR support | *Not implemented yet* (planned) |

---

## 📦 Installation

```bash
pip install ieeLabTools
```

>🧪 This package was originally developed to automate general uncertainty propagation for physics lab courses, reducing >manual symbolic differentiation and repetitive numeric error calculation.

Part of the PhySiLight-Tools physics utilities collection.
