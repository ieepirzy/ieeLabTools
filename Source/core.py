import sympy as sp
import numpy as np

class Yvel():
    
    def __init__(self,f,vars=None):
        """
        

        Parameters
        ----------
        f : Sympy expression representing the function
        vars : list/tuple of Sympy symbols (optional) The default is None.

        Returns
        -------
        None.

        """
        
        self.f = f
        
        if vars is not None:
            self.vars = vars
        else: 
            self.vars = list(f.free_symbols)
        
        self.partials = []
        for v in self.vars:
            self.partials.append(sp.diff(self.f,v))
        
        
        self.sigmas = []
        for v in self.vars:
            sigma = sp.Symbol(f"sigma_{v.name}")
            self.sigmas.append(sigma)
            
        self.symbolic_f = sp.sqrt(
            sum((partial*var)**2 for partial, var in zip(self.partials,self.sigmas))
            )
                    
        self.fn = sp.lambdify([*self.vars, *self.sigmas], self.symbolic_f, "numpy")
        self.k = len(self.vars)
        
    def symbolic(self):
        """Return symbolic uncertainty expression."""
        return self.symbolic_f
     
    def numeric(self,values,sigmas):
        """
        Returns the numerical uncertainties of a measurement series, using the non-covariant general error propagation equation. \n
        Expects
        
        values: 2D array-like (m x k)
        sigmas: 2D array-like (m x k)
        where k is your number of variables and m is the length of measurement data
        
        Returns: numpy array length m
        """
        values = np.array(values, dtype=float)
        sigmas = np.array(sigmas, dtype=float)
        
        if values.shape != sigmas.shape:
            raise ValueError("Measurement and uncertainty matrices must have same shape.")
        m, k = values.shape
        if k != self.k:
            raise ValueError(f"Expected {self.k} variables, got {k} columns.")
        
        vt = values.T
        st = sigmas.T

        # build argument list
        args = [vt[i] for i in range(k)] + [st[i] for i in range(k)]

        return self.fn(*args)

    def covariant_numeric(self,values,sigmas):
        return NotImplementedError

class weightedLinregress():
    def __init__(self, y_sigma,x,y):
        """
        Parameters
        ----------
        y_sigma : 1D array-like of y -axis errors/deviations
        x : 1D array-like of measurements for x 
        y: 1D array-like of measurements for y

        Returns
        -------
        None.

        ---
        Initializes the class.

        """
        self.x = np.array(x, float)
        self.y = np.array(y, float)
        self.y_err = np.array(y_sigma, float)

    def fit(self):

        """
        Method finds the characteristics of the fitted line, with weighting from errors.

        Expects: None

        Returns:

        slope: the fitted slope
        intercept: the fitted intercept
        slope_err: uncertainty in the slope
        intercept_err: uncertainty in the interception
        """

        w = 1 / self.y_err**2
        W  = np.sum(w)
        Wx = np.sum(w * self.x)
        Wy = np.sum(w * self.y)
        Wxx = np.sum(w * self.x * self.x)
        Wxy = np.sum(w * self.x * self.y)

        D = W * Wxx - Wx**2

        slope = (W * Wxy - Wx * Wy) / D
        intercept = (Wxx * Wy - Wx * Wxy) / D

        slope_err = np.sqrt(W / D)
        intercept_err = np.sqrt(Wxx / D)

        return slope, intercept, slope_err, intercept_err

class ortDistanceRegress():
    def __init__(self,x,y,x_sigma,y_sigma):
        
        """
        Parameters
        ----------
        y_sigma : 1D array-like of y -axis errors
        x_sigma : 1D array-like of x -axis errors
        x : 1D array-like of measurements for x 
        y: 1D array-like of measurements for y

        Returns
        -------
        None.

        ---
        Initializes the class.

        """ 
        return NotImplementedError

    def calculate(self):
        return NotImplementedError  
        