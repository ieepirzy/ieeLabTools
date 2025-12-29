# Documentation

<a href="https://github.com/ieepirzy/PhySiLight-Tools/blob/main/Packages/ieeLabTools/suomidocs.md"> Dokumentaation pikaohjeet suomeksi </a>

## Quickstart / TL;DR

```python
from ieeLabTools.core import Yvel
import sympy as sp
import numpy as np

#Example variables and function
U, I = sp.symbols("U I")
R = U/I
calc = Yvel(R)

U_vals = np.array([1,2,3,4,5,6])
I_vals = np.array([6,5,4,3,2,1])
U_err = U_vals*0.001
I_err = I_vals*0.05
#Convert to desirable input format: m x k array-like, where k is your number of variables and m is your number of measurement events.
values = np.column_stack([U_vals, I_vals]) 
sigmas = np.column_stack([U_err, I_err])

sigma_R = calc.numeric(values, sigmas)
``` 

> ⚠ The current version uses the **non-covariant** general error propagation formula. Covariance-aware propagation is planned for a future release.

## 1. YVEL: Yleinen virheen etenemisen lauseke
> YVEL = General error propagation equation in finnish.

The class Yvel implements the general error propagation equation in non-covariant form, and exposes 2 methods for working with it.

The mathematical expression for the YVEL in non-covariant from is:

$$
\sigma = \sqrt{ \sum_{i}  \left( \frac{\partial f}{\partial x_{i}} \right) ^{2}\sigma_{i}^{2} }
$$

The **initializer expects a sympy object representing the function** referred to from now on as `f`.
An optional parameter for the initializer is a 1D array-like of the **variables as Sympy symbols** referred to as `var`.

For note, the covariant expression for the YVEL is:

$$
\sigma = \sqrt{ \sum_{i}  \left( \frac{\partial f}{\partial x_{i}} \right) ^{2}\sigma_{i}^{2} +2 \sum_{i>j} \frac{\partial f}{\partial x_{i}} \frac{\partial f}{\partial x_{j}} \text{cov}(x_{i},x_{j}) }
$$

> Note: Adding a method to take covariance into account is planned for a future release.

## Working principle:
After instantiation, the class immediately either assigns the given `var` or automatically detects variables from `f` using the  `free_symbols()` method.

Then the class calculates the partial derivatives for each variable by looping over all entries in `var` and calling `sp.diff`.

Then a symbolic representation of the general error propagation equation is built as a Sympy object.
#### Example.

For a simplistic example, consider Ohm's law:

$$
R=\frac{U}{I}
$$

To find the deviation for $R$ using the ieeLabTools package, we will first build an instance of the Yvel class:
>Note: for correct usage, some degree of familiarity with numpy and sympy is required.

```python
from ieeLabTools.core import Yvel
import sympy as sp

U , I = sp.symbols("U I") #Assign your symbols

R = U/I #Create the expression

instance = Yvel(R) #After this line the class has been instantiated.
```

The Yvel class currently provides 2 methods for working with the error propagation function. The first of these is the simpler one: `symbolic()`

The `symbolic()` method simply returns the symbolic expression of the general error propagation equation for the `f` used to initialize it. As referenced earlier, this computation is implemented using `sympy`, for further details, consult the  `sympy` documentation for `symbol(), diff(), lambdify() and free_symbols()` methods.

Example usage of the `symbolic()` method:
```python
#continuing from earlier...

symbolic_expression = instance.symbolic()
#For the example function, you should see:
print(symbolic_expression)
#return this:
#sqrt(sigma_U**2/I**2 + U**2*sigma_I**2/I**4)
```

The 2nd method exposed by the class is the `numeric()` method. This is the arguably more useful method, as it allows you to calculate the actual deviation numerically.

### Working principle for `numeric()` 

The `numeric()` method is intended to massively ease the computation of errors for functions with many variables each contributing values (and possibly errors), making the partial derivatives tedious to solve by hand.

To achieve this, I design the method with 2 principles in mind:

1. Handle an arbitrary amount of variables and their errors, with 2 parameters.
2. Do so in a clean, performant manner

The `numeric()` method expects the variables and arrays to be passed in a matrix (array-like) format, where for any number of variables $k$ and a measurement series of length $m$ for each variable, constituting an $m \times k$ 2D array-like.

A mathematical representation:

$$
\text{values}: m \times k \implies \begin{bmatrix}
x_{0} & x_{1} & x_{2} & x_{3} & \dots & x_{k} \\
val_{0} & val_{0} & val_{0} & val_{0} & \dots & val_{0} \\
val_{1} & val_{1} & val_{1} & val_{1} & \dots & val_{1} \\
val_{2} & val_{2} & val_{2} & val_{2} & \dots & val_{2} \\
val_{3} & val_{3} & val_{3} & val_{3} & \dots & val_{3} \\
\vdots & \vdots & \vdots & \vdots & \dots & \vdots \\
val_{m} & val_{m} & val_{m} & val_{m} & \dots & val_{m}
\end{bmatrix}
$$

$$
\text{deviations}: m \times k \implies \begin{bmatrix}
x_{0} & x_{1} & x_{2} & x_{3} & \dots & x_{k} \\
\sigma_{0} & \sigma_{0} & \sigma_{0} & \sigma_{0} & \dots & \sigma_{0} \\
\sigma_{1} & \sigma_{1} & \sigma_{1} & \sigma_{1} & \dots & \sigma_{1} \\
\sigma_{2} & \sigma_{2} & \sigma_{2} & \sigma_{2} & \dots & \sigma_{2} \\
\sigma_{3} & \sigma_{3} & \sigma_{3} & \sigma_{3} & \dots & \sigma_{3} \\
\vdots & \vdots & \vdots & \vdots & \dots & \vdots \\
\sigma_{m} & \sigma_{m} & \sigma_{m} & \sigma_{m} & \dots & \sigma_{m}
\end{bmatrix}
$$


> Note: The measurement series for all variables must be the same size
> Assumption: measurement events for all variables are equal

> Note: The series of deviations must be equal in length with its corresponding series of measurements, and each measurement series must have a corresponding deviation series.

> Note: For possible variables with 0 deviation, pass an array with all values being 0.

Internally, the method uses numpy for vectorized processing and does shape validation. 

Example usage:
```python
#lets assume the variables from earlier have some values and deviations:
import numpy as np

#Real measurement data from a lab-course:
U_values =  np.array([
    0.131, 0.165, 0.204, 0.268, 0.361, 0.505,
    0.692, 0.958, 1.370, 1.997, 2.944, 4.33, 6.74])
    
I_values = np.array([10/1000]*13) #mA -> A

U_errors = np.array([0.000656, 0.000826, 0.001021, 0.001341, 0.001806, 0.002526, 0.003461, 0.004791, 0.006851, 0.009986, 0.014721, 0.021651, 0.033701])
 
I_errors = np.zeros([13]) #good example of 0 deviations handling

# Now in order to call the method, we must remember its required parameters and construct them out of our existing arrays.
values = np.column_stack([U_values, I_values])
sigmas = np.column_stack([U_errors, I_errors])
# This constructs a 13x2 matrix for both values and sigmas, in alignment with what the method parameters expect.


# Now to find the deviations for each value we call:
deviation = instance.numeric(values,sigmas)

#with the example data you should expect something like:
print(deviation)
# "[3.82262106e-04, 3.03397612e-04, 2.45338331e-04, 1.86706393e-04,
# 5.22029629e-05, 3.65016783e-05, 2.50400639e-05, 1.69848494e-05, 1.15478775e-05,
# 7.41861776e-06,]

```


