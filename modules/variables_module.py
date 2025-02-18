import numpy as np
import astropy.units as u
from cosmology_module import *
from variance_module import *

class VariablesGenerator:
    """
    A class for generating and storing simulation variables and derived quantities.

    Attributes:
        h (float): Hubble parameter.
        H0 (float): Hubble constant at z=0.
        Ob (float): Baryon density parameter.
        Oc (float): Cold dark matter density parameter.
        mnu (float): Sum of the neutrino masses.
        ns (float): Scalar spectral index.
        As (float): Amplitude of the primordial scalar perturbations.
        zs (float): Source redshift.
        theta1_radian (float): Angular scale in radians converted from arcminutes.
        theta2_radian (float): Twice the angular scale of theta1 in radians.
        speed_light (float): Speed of light in km/s.
        variables: Placeholder for derived variables.
        nz_file (str, optional): Path to a text file containing n(z) for integration.
    
    Methods:
        __init__(self, h, H0, Ob, Oc, mnu, ns, As, zs, theta1, nz_file=None, **kwargs):
            Initializes the VariablesGenerator with the given parameters and calculates initial derived quantities.
        get_derived_variables(self, **kwargs):
            Calculates additional derived quantities based on the cosmological parameters and stores them as attributes.
    """

    def __init__(self, h, H0, Ob, Oc, mnu, ns, As, zs, theta1, nz_file=None, **kwargs):
        """
        Initializes the VariablesGenerator with given parameters, converts angular scales, and computes initial derived variables.

        Parameters:
            h (float): Hubble parameter.
            H0 (float): Hubble constant at z=0.
            Ob (float): Baryon density parameter.
            Oc (float): Cold dark matter density parameter.
            mnu (float): Sum of the neutrino masses.
            ns (float): Scalar spectral index.
            As (float): Amplitude of the primordial scalar perturbations.
            zs (float): Source redshift.
            theta1 (float): Angular scale in arcminutes.
            nz_file (str, optional): Path to the n(z) text file.
            **kwargs: Additional keyword arguments passed to the initializer.
        """
        self.h = h
        self.H0 = H0
        self.Ob = Ob
        self.Oc = Oc
        self.mnu = mnu
        self.ns = ns
        self.As = As
        self.zs = zs 
        self.theta1_radian = (theta1 * u.arcmin).to(u.radian).value  # Convert theta1 from arcmin to radians
        self.theta2_radian = 2. * self.theta1_radian  # Double the angular scale for theta2
        self.nz_file = nz_file # Store nz_file if provided
        
        self.variables = self.get_derived_variables(nz_file=nz_file, **kwargs)  # Compute initial set of derived variables


    def get_derived_variables(self, nz_file=None, **kwargs):
        """
        Calculates and stores derived variables based on the initial parameters.

        This method computes the cosmological distance to the source, sets up lensing planes,
        calculates matter power spectrum interpolators, and computes variances for mass maps.
        
        Parameters:
            nz_file (str, optional): Path to the n(z) file.
            **kwargs: Additional keyword arguments for computing derived variables.
        
        Returns:
            None
        """
        # Extract kmin and kmax from kwargs with default values
        kmin = kwargs.get('kmin', 5e-4)
        kmax = kwargs.get('kmax', 1000.)
        nplanes = kwargs.get('nplanes', 51)  # Number of lensing planes
        
        # Initialize cosmology with given parameters
        # If nz_file is provided, pass it along to Cosmology_function.
        self.cosmo = Cosmology_function(self.h, self.H0, self.Ob, self.Oc, self.mnu, self.ns, self.As, self.zs, nz_file=nz_file)        
        
        # Compute comoving distance to source redshift (fallback for single-source case)
        self.chi_source = self.cosmo.get_chi(self.zs)
        
        # Setup lensing planes
        self.chis = np.linspace(0.3, self.chi_source-5, nplanes)
        # Compute differential comoving distances for trapezoidal integration
        self.dchis = np.ones(len(self.chis)) * (self.chis[1] - self.chis[0]) / 2
        self.dchis[1:-1] *= 2.
        
        # Calculate lensing weight array
        # If an n(z) file is provided, use the integrated version.    
        if self.cosmo.nz_file is not None:
            self.z_array, self.lensing_weight = self.cosmo.get_lensing_weight_array_nz(self.chis)
        else:
            self.z_array, self.lensing_weight = self.cosmo.get_lensing_weight_array(self.chis, self.chi_source)
        
        # # KDE version
        # if self.cosmo.nz_kde is not None:
        #     self.z_array, self.lensing_weight = self.cosmo.get_lensing_weight_array_kde(self.chis)
        # else:
        #     self.z_array, self.lensing_weight = self.cosmo.get_lensing_weight_array(self.chis, self.cosmo.get_chi(self.zs))

        
        print("the source redshift is: ", self.zs)
        print("the chistar is: ", self.chi_source)
        print("the number of planes being used is: ", nplanes)
        
        # Initialize power spectrum interpolators
        self.PK_interpolator_linear = self.cosmo.get_matter_power_interpolator(nonlinear=False, kmin=kmin, kmax=kmax, nk=500)
        self.PK_interpolator_nonlinear = self.cosmo.get_matter_power_interpolator(nonlinear=True, kmin=kmin, kmax=kmax, nk=500)
        
        # Compute variances for mass maps using the calculated interpolators
        self.variance = Variance(self.cosmo, self.PK_interpolator_linear, self.PK_interpolator_nonlinear, model='takahashi')
        lensing_weight_squared = self.lensing_weight ** 2.
        
        # Summations for sigma squared calculations across all lensing planes
        self.sigmasq_map = np.sum(self.dchis * lensing_weight_squared * np.array([self.variance.get_sig_slice(z, chi*self.theta1_radian, chi*self.theta2_radian) for z, chi in zip(self.z_array, self.chis)]))
        self.sigmasq_delta1 = np.sum(self.dchis * lensing_weight_squared * np.array([self.variance.nonlinear_sigma2(z, self.theta1_radian*chi) for z, chi in zip(self.z_array, self.chis)]))
        self.sigmasq_delta2 = np.sum(self.dchis * lensing_weight_squared * np.array([self.variance.nonlinear_sigma2(z, self.theta2_radian*chi) for z, chi in zip(self.z_array, self.chis)]))
        self.recal_value = 1.  # Placeholder for recalibration value if needed

        # Display calculated chi_source and variance as a sanity check
        print("The mass map variance from theory is: ", self.sigmasq_map)
        
        return None
