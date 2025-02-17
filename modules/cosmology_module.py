import numpy as np
import camb
import matplotlib.pyplot as plt

class Cosmology_function:
    """
        Initializes the cosmology based on the parameters
        H0 = hubble constant in km/s/Mpc
        Ob = Omega baryon
        Oc = Omega cdm
        Omnu = Omega massive neutrinos
        Om = Omega matter = Ob + Oc + Omnu
        Ol = Omega lambda 
        ns   = scalar spectral index
        As_ = amplitude (dimensionless, later scaled by 1e-9)
        zs = source redshift if using a single source plane
        nz_file (optional) = path to txt file containing n(z) for integration
        
        Note:
            The n(z) file should have two columns:
                - first column: redshift z
                - second column: n(z) (can be unnormalized)

    """
    def __init__(self, h, H0, Ob, Oc, mnu, ns, As_, zs, nz_file=None, **kwargs):
        self.h = h  # Assuming H0 is in km/s/Mpc
        self.Ob = Ob
        self.Oc = Oc
        self.mnu = mnu
        self.Omnu = mnu / 93.14 / self.h / self.h
        self.Om = Ob + Oc + (mnu / 93.14 / self.h / self.h)
        self.Oc = self.Om - Ob - self.Omnu
        self.ns = ns
        self.As_ = As_
        self.As = As_ * 1e-9
        self.Ol = 1.0 - self.Om
        self.H0 = H0
        self.speed_light = 299792.458
        
        # If nz_file is provided, store it for integrated lensing weight calculations
        self.nz_file = nz_file
        
        # Use provided source redshift if n(z) is not provided
        self.zsource = zs
        self.zini = 0.0
        self.zmax = 5      
        self.pars = self._set_params()
        self.results = camb.get_results(self.pars)

    def _set_params(self):
        return camb.set_params(
                               H0=self.H0 * self.h, 
                               omch2=self.Oc * (self.h ** 2.),
                               ombh2=self.Ob * (self.h ** 2.), 
                               NonLinear=camb.model.NonLinear_both,
                               mnu=self.mnu, 
                               omnuh2=self.Omnu * self.h * self.h, 
                               As=self.As, 
                               ns=self.ns,
                               halofit_version='takahashi'
                               )
        

    def get_chi(self, redshift):
        return self.results.comoving_radial_distance(redshift, tol=0.0000001) * self.h

    def get_z_from_chi(self, chi):
        return self.results.redshift_at_comoving_radial_distance(chi / self.h)

    def get_lensing_weight(self, chi, chi_source):
        """
        Lensing weight for a single source plane.
        """
        z = self.get_z_from_chi(chi)
        return 1.5 * self.Om * (self.speed_light ** -2.) * ((self.H0) ** 2.) * chi * (1 - (chi / chi_source)) * (1 + z)
    
    
    def get_lensing_weight_array(self, chis, chi_source):
        """
        Compute lensing weights for an array of chi values using a single source plane.
        """
        z_values = self.get_z_from_chi(chis)
        lensing_weight = np.zeros_like(chis)
        for i in range(len(chis)):
            lensing_weight[i] = self.get_lensing_weight(chis[i], chi_source)


        # plot the lensing weight:
        print("now plotting the lensing weight")
        plt.plot(chis, lensing_weight, label='lensing weight')
        plt.xlabel('chi')
        plt.ylabel('lensing weight')
        plt.legend()
        plt.show()
        return z_values, lensing_weight
    

    def get_lensing_weight_nz(self, chi):
        """
        Compute the lensing weight at a given lens distance chi by integrating over the source
        distribution n(z) from the provided nz_file.
        """
        if self.nz_file is None:
            raise ValueError("nz_file must be provided for integrated lensing weight calculations.")
     
        # Load and normalize n(z)
        nz = np.loadtxt(self.nz_file)
        z_nz = nz[:, 0]
        n_nz = nz[:, 1]
        n_norm = n_nz / np.trapezoid(n_nz, z_nz)
        
        # Compute the comoving distance for each source redshift
        chi_nz = self.results.comoving_radial_distance(z_nz, tol=0.0000001) * self.h
        
        # only consider sources that are behind the lens (chi_source > chi)
        mask = chi_nz > chi
        if np.sum(mask) == 0:
            return 0.0
        chi_nz_sel = chi_nz[mask]
        n_norm_sel = n_norm[mask]
        
        # The integrand: (1 - chi/chi_source) weighted by n(z)
        integrand = (1 - chi / chi_nz_sel) * n_norm_sel
        # Integrate over comoving distance
        integral = np.trapezoid(integrand, chi_nz_sel)
        
        # Use standard prefactor (z here is the *lens* redshift)
        z_lens = self.get_z_from_chi(chi)
        weight = 1.5 * self.Om * (self.speed_light ** -2.) * (self.H0 ** 2) * chi * (1 + z_lens) * integral
        
        return weight
    
    def get_lensing_weight_array_nz(self, chis):
        """
        Compute the lensing weight for an array of lens distances using integrated n(z).
        Returns the corresponding lens redshifts and lensing weights.
        """
        if self.nz_file is None:
            raise ValueError("nz_file must be provided for integrated lensing weight calculations.")
        
        # Load and normalize n(z)
        nz = np.loadtxt(self.nz_file)
        z_nz = nz[:, 0]
        n_nz = nz[:, 1]
        n_norm = n_nz / np.trapezoid(n_nz, z_nz)
        
        # Pre-compute the comoving distance for each source redshift
        chi_nz = self.results.comoving_radial_distance(z_nz, tol=0.0000001) * self.h
        
        lensing_weight = np.zeros_like(chis)
        z_values = self.get_z_from_chi(chis)
        for i, chi in enumerate(chis):
            # only consider sources that are behind the lens (chi_source > chi)
            mask = chi_nz > chi
            if np.sum(mask) == 0:
                lensing_weight[i] = 0.0
            else:
                chi_nz_sel = chi_nz[mask]
                n_norm_sel = n_norm[mask]
                
                # The integrand: (1 - chi/chi_source) weighted by n(z)
                integrand = (1 - chi / chi_nz_sel) * n_norm_sel
                # Integrate over comoving distance
                integral = np.trapezoid(integrand, chi_nz_sel)
                
                # Use standard prefactor (z here is the *lens* redshift)
                lensing_weight[i] = 1.5 * self.Om * (self.speed_light ** -2.) * (self.H0 ** 2) * chi * (1 + z_values[i]) * integral
                
        lensing_weight = lensing_weight *0.000583  # temporary fix for the lensing weight normalization with n(z)
            
        # plot the lensing weight:
        print("now plotting the lensing weight")
        plt.plot(chis, lensing_weight, label='lensing weight')
        plt.xlabel('chi')
        plt.ylabel('lensing weight')
        plt.legend()
        plt.show()        
                
        return z_values, lensing_weight
    
        
    def get_matter_power_interpolator(self, nonlinear=False, kmin=1e-3, kmax=150, nk=300):
        self.k_values = np.logspace(np.log10(kmin), np.log10(kmax), nk, base=10)

        if nonlinear:
            PK_interpolator = camb.get_matter_power_interpolator(self.pars, nonlinear=True,
                hubble_units=True, k_hunit=True, kmax=kmax, zmax=self.zmax)
        else:
            PK_interpolator = camb.get_matter_power_interpolator(self.pars, nonlinear=False,
                hubble_units=True, k_hunit=True, kmax=kmax, zmax=self.zmax)

        # You can use PK_interpolator.P(z, k) to get the power spectrum at redshift z and wavenumber k
        return PK_interpolator
