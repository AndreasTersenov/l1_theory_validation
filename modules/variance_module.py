# from cosmology_module import Cosmology_function
import numpy as np
import scipy.special
from scipy.integrate import simpson

class Variance:
    """
    A class to compute linear and nonlinear variance using power spectrum interpolators and a specific cosmological model.
    
    Attributes:
        cosmo (Cosmology): An instance of a cosmology class providing necessary cosmological functions and parameters.
        PK_interpolator_linear (Interpolator): An interpolator instance for linear power spectrum calculations.
        PK_interpolator_nonlinear (Interpolator): An interpolator instance for nonlinear power spectrum calculations.
        model (str): The name of the cosmological model to be used for variance calculations.
        
        Formulae: 
        \sigma^2\!(R_1,R_2;\!z) \!=\!\!\!\!\int \!\!\frac{{\rm d}^2\bm{k}_{{\perp}}}{(2\pi)^2}  \!P(k_{{\perp}};z)  W_{TH}(R_1 k_{\perp}\!)  W_{TH}(\!R_2 k_{\perp}\!)
        where $W_{TH}(l) = 2J_1(l)/l$ and $J_1$ is the first Bessel function of the first kind;
    """
    def __init__(self, cosmo, PK_interpolator_linear, PK_interpolator_nonlinear, model):
        """
        Initializes the Variance class with cosmology, power spectrum interpolators, and the model name.
        
        Parameters:
            cosmo (Cosmology): An instance of a cosmology class.
            PK_interpolator_linear (Interpolator): An interpolator for linear power spectrum.
            PK_interpolator_nonlinear (Interpolator): An interpolator for nonlinear power spectrum.
            model (str): The cosmological model name.
        """
        self.cosmo = cosmo
        self.PK_interpolator_linear = PK_interpolator_linear
        self.PK_interpolator_nonlinear = PK_interpolator_nonlinear
        self.model=model

    def top_hat_window(self, R):
        """
        Calculates the top-hat window function for a given radius.
        
        Parameters:
            R (float or numpy.ndarray): The scale (or array of scales) at which to calculate the window function.
            
        Returns:
            numpy.ndarray: The top-hat window function values at the given scale(s).
        """
        return 2. * scipy.special.j1(R) / R

    def linear_sigma2(self, redshift, R1, R2=None):
        """
        Calculates the linear variance σ² for given scales and redshift, considering the specified model adjustments.
        
        Parameters:
            redshift (float): The redshift at which to evaluate the variance.
            R1 (float): The first scale radius.
            R2 (float, optional): The second scale radius. Defaults to R1 if not specified.
            
        Returns:
            float: The linear variance σ² at the given scales and redshift.
        """
        if R2 is None:
            R2 = R1
        
        if self.model=='Takahashi' or 'takahashi':
            lres = 4096 * 1.6
            l = self.cosmo.k_values * self.cosmo.get_chi(redshift)
            pk_factor = 1. / (1. + (l / lres) ** 2.)
            c1, c2 = 9.5171e-4, 5.1543e-3
            a1, a2, a3 = 1.3063, 1.1475, 0.62793
            p = ((1. + c1 * (self.cosmo.k_values**-a1))**a1) / ((1. + c2 * (self.cosmo.k_values**-a2))**a3)
        else:
            pk_factor = 1.
            p = 1.

        pk = self.PK_interpolator_linear.P(redshift, self.cosmo.k_values) * pk_factor * p

        w1_2D = self.top_hat_window(self.cosmo.k_values * R1)
        w2_2D = self.top_hat_window(self.cosmo.k_values * R2)
        w2 = w1_2D * w2_2D
        constant = 1. / 2. / np.pi
        integrand = self.cosmo.k_values * pk * w2 * constant
        return simpson(integrand, x=self.cosmo.k_values)

    def nonlinear_sigma2(self, redshift, R1, R2=None):
        """
        Calculates the nonlinear variance σ² for given scales and redshift, considering the specified model adjustments.
        
        Parameters:
            redshift (float): The redshift at which to evaluate the variance.
            R1 (float): The first scale radius.
            R2 (float, optional): The second scale radius. Defaults to R1 if not specified.
            
        Returns:
            float: The nonlinear variance σ² at the given scales and redshift.
        """
        if R2 is None:
            R2 = R1
        else:
            R2 = R2
        
        if self.model == 'Takahashi' or 'takahashi':
            lres = 4096 * 1.6
            l = self.cosmo.k_values * self.cosmo.get_chi(redshift)
            c1, c2 = 9.5171e-4, 5.1543e-3
            a1, a2, a3 = 1.3063, 1.1475, 0.62793
            p = ((1 + c1 * (self.cosmo.k_values**-a1))**a1) / ((1. + c2 * (self.cosmo.k_values**-a2))**a3)
            pk_factor = 1. / (1. + (l / lres) ** 2.)
        else:
            pk_factor = 1.
            p = 1.
            
        pk = self.PK_interpolator_nonlinear.P(redshift, self.cosmo.k_values) * pk_factor * p
        w1_2D = self.top_hat_window(self.cosmo.k_values * R1)
        w2_2D = self.top_hat_window(self.cosmo.k_values * R2)
        w2 = w1_2D * w2_2D
        constant = 1. / 2. / np.pi
        integrand = self.cosmo.k_values * pk * w2 * constant
        return simpson(integrand, x=self.cosmo.k_values)

    def get_sig_slice(self, z, R1, R2):
        """
        Calculates the slice variance σ² for the given scales and redshift in the nonlinear regime.
        
        Parameters:
            z (float): The redshift at which to evaluate the slice variance.
            R1 (float): The first scale radius.
            R2 (float): The second scale radius.
            
        Returns:
            float: The slice variance σ² at the given scales and redshift.
        """
        chi = self.cosmo.get_chi(z)
        sigslice = (
            self.nonlinear_sigma2(z, R1)
            + self.nonlinear_sigma2(z, R2)
            - 2.0 * self.nonlinear_sigma2(z, R1, R2)
        )
        return sigslice