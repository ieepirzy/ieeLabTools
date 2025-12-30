"""
ieeLabTools
===========

Tools for symbolic and numeric uncertainty propagation
and weighted linear regression for laboratory data analysis.
"""


from .core import (
    Yvel,
    WeightedLinregress,
)

__all__ = [
    "Yvel",
    "WeightedLinearRegression",
]

__version__ = "0.1.3"