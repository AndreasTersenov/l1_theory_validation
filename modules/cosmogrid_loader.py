import numpy as np
import h5py
import healpy as hp
import multiprocessing as mp

class CosmoGridLoader:
    def __init__(self, data_path="/feynman/home/dap/lcs/at278006/data/cosmoGRID", nside_target=512):
        """
        data_path  : base directory containing CosmoGRID HDF5 files.
        nside_target: target Healpix resolution (CosmoGRID maps are at nside=512).
        """
        self.data_path = data_path
        self.nside_target = nside_target  # CosmoGRID maps are provided at nside=512

    def get_beam(self, theta, lmax):
        """
        Compute the beam transfer function for a top-hat filter.
        
        theta: smoothing scale in arcmin.
        lmax : maximum multipole.
        """
        def top_hat(b, radius):
            # Returns a top-hat filter; adjust normalization if needed.
            return np.where(np.abs(b) <= radius, 1/(np.cos(radius) - 1)/(-2*np.pi), 0)
        
        t = theta * np.pi / (60 * 180)  # convert arcmin to radians
        b = np.linspace(0.0, t * 1.2, 10000)
        bw = top_hat(b, t)
        beam = hp.sphtfunc.beam2bl(bw, b, lmax)
        return beam

    def smooth_map(self, args):
        """
        Smooth the input kappa map using the beam corresponding to the given theta.
        
        args: tuple (theta, lmax, kappa_map, nside)
        """
        theta, lmax, kappa_map, nside = args
        beam = self.get_beam(theta, lmax)
        almkappa = hp.sphtfunc.map2alm(kappa_map)
        kappa_smooth = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam), nside)
        return kappa_smooth

    def load_cosmogrid_map(self, id_zbin, perm_index=0):
        """
        Load the base κ map from the CosmoGRID HDF5 file.
        
        id_zbin   : redshift bin index (e.g., 3 for the fourth bin)
        perm_index: simulation permutation index.
        """
        # For simplicity we use the "nobaryons" file; update if needed.
        file_path = f"{self.data_path}/perm_{perm_index:04d}/projected_probes_maps_nobaryons512.h5"
        with h5py.File(file_path, 'r') as f:
            # The dataset is assumed to be stored as 'kg/stage3_lensing{id_zbin+1}'
            dataset_name = f"kg/stage3_lensing{id_zbin+1}"
            kappa = np.array(f[dataset_name])
        # Ensure the map is at the target nside (though it should already be 512)
        kappa_target = hp.ud_grade(kappa, self.nside_target)
        return kappa_target

    def run_loader(self, theta, id_zbin=3, perm_index=0, nprocess=1):
        """
        Mimics TakahashiLoader.run_loader:
          - Loads the base CosmoGRID κ map.
          - Smooths it twice: once with θ and once with 2θ.
          - Computes the variance of the difference between the two smoothed maps.
          - For simplicity, assumes the source redshift is the mean n(z) (here set as a default).
        
        Parameters:
          theta     : primary smoothing scale in arcmin.
          id_zbin   : redshift bin index.
          perm_index: permutation index (to choose which simulation realization to load).
          nprocess  : number of processes to use for smoothing.
        
        Returns:
          variance  : variance of (kappa_smooth2 - kappa_smooth1).
          kappa_smooth1: map smoothed with θ.
          kappa_smooth2: map smoothed with 2θ.
        """
        # Load the base map (at nside=512)
        kappa = self.load_cosmogrid_map(id_zbin, perm_index)
        nside_target = self.nside_target
        lmax = nside_target * 3 - 1  # same as in Takahashi
        
        # Setup arguments for the two smoothing scales
        args1 = (theta, lmax, kappa, nside_target)
        args2 = (theta * 2, lmax, kappa, nside_target)
        
        # Use parallel processing if requested
        if nprocess > 1:
            with mp.Pool(processes=nprocess) as pool:
                kappa_smooth1, kappa_smooth2 = pool.map(self.smooth_map, [args1, args2])
        else:
            kappa_smooth1 = self.smooth_map(args1)
            kappa_smooth2 = self.smooth_map(args2)
            
        variance = np.var(kappa_smooth2 - kappa_smooth1)
        
        # For now, we take the mean of n(z) as our source redshift.
        # (Update this if you later have the actual n(z) distribution.)
        source_z = 0.95  
        print("Assumed source redshift (mean n(z)):", source_z)
        
        return variance, kappa_smooth1, kappa_smooth2