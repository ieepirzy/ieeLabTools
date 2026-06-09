"""
Comprehensive tests for ieeLabTools.

Expected values are derived analytically from the error propagation formula:
    σ = sqrt( Σ (∂f/∂xᵢ · σᵢ)² )

and from the weighted least-squares formulas documented in Docs/Documentation.md.
"""

import math
import numpy as np
import pytest
import sympy as sp

from ieeLabTools import Yvel, WeightedLinregress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(actual, expected, rtol=1e-9, atol=0.0):
    """Wrapper around np.testing.assert_allclose with tighter defaults."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Yvel – symbolic output
# ---------------------------------------------------------------------------

class TestYvelSymbolic:
    def test_addition_symbolic(self):
        """f = a + b  →  σ = sqrt(σa² + σb²)"""
        a, b = sp.symbols("a b")
        calc = Yvel(a + b, vars=[a, b])
        expr = calc.symbolic()
        σa, σb = sp.symbols("σa σb")
        expected = sp.sqrt(σa**2 + σb**2)
        assert sp.simplify(expr - expected) == 0

    def test_multiplication_symbolic(self):
        """f = x * y  →  σ = sqrt((y·σx)² + (x·σy)²)"""
        x, y = sp.symbols("x y")
        calc = Yvel(x * y, vars=[x, y])
        expr = calc.symbolic()
        σx, σy = sp.symbols("σx σy")
        expected = sp.sqrt((y * σx)**2 + (x * σy)**2)
        assert sp.simplify(expr - expected) == 0

    def test_ohms_law_symbolic(self):
        """f = U/I  →  symbolic expression has two terms."""
        U, I = sp.symbols("U I")
        calc = Yvel(U / I, vars=[U, I])
        expr = calc.symbolic()
        # Expression must contain both σU and σI symbols
        free = {s.name for s in expr.free_symbols}
        assert "σU" in free
        assert "σI" in free

    def test_constant_function_symbolic(self):
        """f = 5 (constant)  →  σ = 0."""
        x = sp.Symbol("x")
        calc = Yvel(sp.Integer(5), vars=[x])
        expr = calc.symbolic()
        assert expr == 0

    def test_single_variable_symbolic(self):
        """f = 3*x  →  numeric result equals 3·σx (linear scaling)."""
        x = sp.Symbol("x")
        calc = Yvel(3 * x, vars=[x])
        # Verify numerically: σ = |3| * σx = 3 * 0.5 = 1.5
        result = calc.numeric(np.array([[2.0]]), np.array([[0.5]]))
        assert_close(result, [1.5])

    def test_var_autodetect_sorted(self):
        """Auto-detected vars are sorted lexicographically."""
        a, b, c = sp.symbols("a b c")
        calc = Yvel(a + b + c)
        assert calc.vars == [a, b, c]

    def test_explicit_vars_override_autodetect(self):
        """Explicit vars list is preserved as-is."""
        U, I = sp.symbols("U I")
        calc = Yvel(U / I, vars=[U, I])
        assert calc.vars == [U, I]


# ---------------------------------------------------------------------------
# Yvel – numeric output (known analytical results)
# ---------------------------------------------------------------------------

class TestYvelNumeric:
    def test_addition_single_point(self):
        """f = a + b, σ = sqrt(σa² + σb²).
        a=1, b=2, σa=0.1, σb=0.2  →  σ = sqrt(0.05) ≈ 0.22360679"""
        a, b = sp.symbols("a b")
        calc = Yvel(a + b, vars=[a, b])
        values = np.array([[1.0, 2.0]])
        sigmas = np.array([[0.1, 0.2]])
        result = calc.numeric(values, sigmas)
        expected = math.sqrt(0.01 + 0.04)  # 0.22360679…
        assert_close(result, [expected])

    def test_multiplication_single_point(self):
        """f = x * y, σ = sqrt((y·σx)² + (x·σy)²).
        x=2, y=3, σx=0.1, σy=0.2  →  σ = sqrt(0.09+0.16) = 0.5"""
        x, y = sp.symbols("x y")
        calc = Yvel(x * y, vars=[x, y])
        values = np.array([[2.0, 3.0]])
        sigmas = np.array([[0.1, 0.2]])
        result = calc.numeric(values, sigmas)
        expected = math.sqrt((3 * 0.1) ** 2 + (2 * 0.2) ** 2)  # 0.5
        assert_close(result, [expected])

    def test_ohms_law_single_point(self):
        """f = U/I.  U=10, I=2, σU=0.1, σI=0.05.
        ∂f/∂U = 1/I = 0.5, ∂f/∂I = -U/I² = -2.5
        σ = sqrt((0.5·0.1)² + (2.5·0.05)²) = sqrt(0.0025+0.015625)"""
        U, I = sp.symbols("U I")
        calc = Yvel(U / I, vars=[U, I])
        values = np.array([[10.0, 2.0]])
        sigmas = np.array([[0.1, 0.05]])
        result = calc.numeric(values, sigmas)
        expected = math.sqrt(0.0025 + 0.015625)  # ≈ 0.13463
        assert_close(result, [expected], rtol=1e-9)

    def test_ohms_law_zero_current_error(self):
        """f = U/I with σI = 0: σR = σU/I (only voltage contributes)."""
        U, I = sp.symbols("U I")
        calc = Yvel(U / I, vars=[U, I])
        U_vals = np.array([0.131, 0.505, 6.74])
        I_vals = np.full(3, 0.01)
        U_err = np.array([0.000656, 0.002526, 0.033701])
        I_err = np.zeros(3)
        values = np.column_stack([U_vals, I_vals])
        sigmas = np.column_stack([U_err, I_err])
        result = calc.numeric(values, sigmas)
        expected = U_err / I_vals  # σU / I
        assert_close(result, expected)

    def test_full_ohms_law_lab_data(self):
        """Reproduces the lab-data example from Documentation.md.
        With σI = 0, every result equals σU / I."""
        U, I = sp.symbols("U I")
        calc = Yvel(U / I, vars=[U, I])
        U_values = np.array([
            0.131, 0.165, 0.204, 0.268, 0.361, 0.505,
            0.692, 0.958, 1.370, 1.997, 2.944, 4.33, 6.74
        ])
        I_values = np.full(13, 10e-3)
        U_errors = np.array([
            0.000656, 0.000826, 0.001021, 0.001341, 0.001806, 0.002526,
            0.003461, 0.004791, 0.006851, 0.009986, 0.014721, 0.021651, 0.033701
        ])
        I_errors = np.zeros(13)
        values = np.column_stack([U_values, I_values])
        sigmas = np.column_stack([U_errors, I_errors])
        result = calc.numeric(values, sigmas)
        expected = U_errors / I_values
        assert_close(result, expected)

    def test_three_variable_function(self):
        """f = x + y + z, all σ = 1  →  σ = sqrt(3) ≈ 1.73205."""
        x, y, z = sp.symbols("x y z")
        calc = Yvel(x + y + z, vars=[x, y, z])
        values = np.array([[1.0, 2.0, 3.0]])
        sigmas = np.array([[1.0, 1.0, 1.0]])
        result = calc.numeric(values, sigmas)
        expected = math.sqrt(3)
        assert_close(result, [expected])

    def test_series_of_measurements(self):
        """f = a + b over m=3 measurements returns array of length 3."""
        a, b = sp.symbols("a b")
        calc = Yvel(a + b, vars=[a, b])
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sigmas = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = calc.numeric(values, sigmas)
        expected = np.sqrt(sigmas[:, 0] ** 2 + sigmas[:, 1] ** 2)
        assert result.shape == (3,)
        assert_close(result, expected)

    def test_zero_error_gives_zero_result(self):
        """All σ = 0 must yield σ = 0."""
        x, y = sp.symbols("x y")
        calc = Yvel(x * y, vars=[x, y])
        values = np.array([[2.0, 3.0]])
        sigmas = np.zeros((1, 2))
        result = calc.numeric(values, sigmas)
        assert_close(result, [0.0], atol=1e-15)

    def test_list_input_accepted(self):
        """numeric() must accept plain Python lists, not just numpy arrays."""
        a, b = sp.symbols("a b")
        calc = Yvel(a + b, vars=[a, b])
        result = calc.numeric([[1.0, 2.0]], [[0.1, 0.2]])
        expected = math.sqrt(0.01 + 0.04)
        assert_close(result, [expected])

    def test_quadratic_function(self):
        """f = x², σ = 2|x|·σx.
        x=3, σx=0.1  →  σ = 0.6"""
        x = sp.Symbol("x")
        calc = Yvel(x ** 2, vars=[x])
        values = np.array([[3.0]])
        sigmas = np.array([[0.1]])
        result = calc.numeric(values, sigmas)
        expected = 2 * 3 * 0.1  # 0.6
        assert_close(result, [expected])

    def test_k_attribute(self):
        """calc.k equals number of variables."""
        x, y, z = sp.symbols("x y z")
        calc = Yvel(x + y + z, vars=[x, y, z])
        assert calc.k == 3


# ---------------------------------------------------------------------------
# Yvel – input validation
# ---------------------------------------------------------------------------

class TestYvelValidation:
    def test_shape_mismatch_raises(self):
        """values and sigmas with different shapes must raise ValueError."""
        a, b = sp.symbols("a b")
        calc = Yvel(a + b, vars=[a, b])
        with pytest.raises(ValueError, match="same shape"):
            calc.numeric(np.ones((3, 2)), np.ones((2, 2)))

    def test_wrong_column_count_raises(self):
        """Passing wrong number of variable columns must raise ValueError."""
        a, b = sp.symbols("a b")
        calc = Yvel(a + b, vars=[a, b])
        with pytest.raises(ValueError, match="Expected 2"):
            calc.numeric(np.ones((3, 3)), np.ones((3, 3)))

    def test_covariant_not_implemented(self):
        """covariant_numeric is not yet implemented (returns NotImplementedError class)."""
        x = sp.Symbol("x")
        calc = Yvel(x, vars=[x])
        result = calc.covariant_numeric(np.array([[1.0]]), np.array([[0.1]]))
        assert result is NotImplementedError


# ---------------------------------------------------------------------------
# WeightedLinregress
# ---------------------------------------------------------------------------

class TestWeightedLinregress:
    def test_perfect_line_through_origin(self):
        """y = 2x with uniform errors  →  slope=2, intercept=0.
        Derived by hand with w=[1,1,1], x=[1,2,3], y=[2,4,6]."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        y_err = np.array([1.0, 1.0, 1.0])
        slope, intercept, slope_err, intercept_err = WeightedLinregress(y_err, x, y).fit()
        # Analytical: D=6, slope=2, intercept=0
        assert_close(slope, 2.0)
        assert_close(intercept, 0.0, atol=1e-12)
        assert_close(slope_err, math.sqrt(3.0 / 6.0))      # sqrt(W/D)=sqrt(0.5)
        assert_close(intercept_err, math.sqrt(14.0 / 6.0))  # sqrt(Wxx/D)

    def test_line_with_nonzero_intercept(self):
        """y = 2x + 1, uniform errors  →  slope=2, intercept=1.
        x=[1,2,3], y=[3,5,7], y_err=[1,1,1]."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 5.0, 7.0])
        y_err = np.array([1.0, 1.0, 1.0])
        slope, intercept, slope_err, intercept_err = WeightedLinregress(y_err, x, y).fit()
        assert_close(slope, 2.0)
        assert_close(intercept, 1.0)

    def test_weighted_nonuniform_errors(self):
        """y = 2x, non-uniform y_err=[0.5, 1.0, 2.0].
        Lower-error points carry more weight; slope and intercept still exact.
        Computed analytically: W=5.25, Wx=6.75, D=8.25."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        y_err = np.array([0.5, 1.0, 2.0])
        slope, intercept, slope_err, intercept_err = WeightedLinregress(y_err, x, y).fit()
        assert_close(slope, 2.0, rtol=1e-9)
        assert_close(intercept, 0.0, atol=1e-10)
        W = np.sum(1 / y_err**2)
        D = W * np.sum(x**2 / y_err**2) - np.sum(x / y_err**2) ** 2
        assert_close(slope_err, math.sqrt(W / D))
        assert_close(intercept_err, math.sqrt(np.sum(x**2 / y_err**2) / D))

    def test_small_errors_dominate_fit(self):
        """A single very precise point should dominate the weighted fit."""
        x = np.array([0.0, 5.0, 10.0])
        # True line: y = x, but first point is far off and imprecise
        y = np.array([100.0, 5.0, 10.0])
        y_err = np.array([1000.0, 0.001, 0.001])
        slope, intercept, _, _ = WeightedLinregress(y_err, x, y).fit()
        # Precise points at (5,5) and (10,10) heavily constrain the fit to y=x
        assert_close(slope, 1.0, rtol=1e-3)
        assert_close(intercept, 0.0, atol=1e-1)

    def test_two_point_fit(self):
        """Two data points uniquely determine slope and intercept.
        x=[1,3], y=[2,8], y_err=[1,1]: slope=(8-2)/(3-1)=3, intercept=2-3=-1."""
        x = np.array([1.0, 3.0])
        y = np.array([2.0, 8.0])
        y_err = np.array([1.0, 1.0])
        slope, intercept, _, _ = WeightedLinregress(y_err, x, y).fit()
        assert_close(slope, 3.0)
        assert_close(intercept, -1.0)

    def test_fit_returns_four_values(self):
        """fit() must return a 4-tuple: (slope, intercept, slope_err, intercept_err)."""
        result = WeightedLinregress(
            np.ones(3), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
        ).fit()
        assert len(result) == 4

    def test_list_input_accepted(self):
        """Constructor must accept Python lists, not just numpy arrays."""
        slope, intercept, _, _ = WeightedLinregress(
            [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
        ).fit()
        assert_close(slope, 2.0)
        assert_close(intercept, 0.0, atol=1e-12)

    def test_uncertainties_are_positive(self):
        """Slope and intercept uncertainties must always be positive."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 0.5
        y_err = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
        slope, intercept, slope_err, intercept_err = WeightedLinregress(y_err, x, y).fit()
        assert slope_err > 0
        assert intercept_err > 0

    def test_y_errors_as_float_array(self):
        """Internal y_err attribute is a float numpy array."""
        reg = WeightedLinregress([1, 2, 3], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert reg.y_err.dtype == np.float64

    def test_large_dataset_slope(self):
        """y = 5x + 3 with 100 uniform-error points recovers slope and intercept."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y = 5.0 * x + 3.0 + rng.normal(0, 0.01, 100)
        y_err = np.full(100, 0.5)
        slope, intercept, _, _ = WeightedLinregress(y_err, x, y).fit()
        assert_close(slope, 5.0, rtol=1e-2)
        assert_close(intercept, 3.0, atol=0.1)


# ---------------------------------------------------------------------------
# Package-level import sanity
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_yvel_importable_from_package(self):
        from ieeLabTools import Yvel as _Yvel
        assert _Yvel is Yvel

    def test_weighted_linregress_importable_from_package(self):
        from ieeLabTools import WeightedLinregress as _WL
        assert _WL is WeightedLinregress

    def test_version_string_present(self):
        import ieeLabTools
        assert hasattr(ieeLabTools, "__version__")
        assert isinstance(ieeLabTools.__version__, str)
