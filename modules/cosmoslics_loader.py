import numpy as np
import h5py
import multiprocessing as mp
import scipy.special
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

def top_hat_window_FL(k, R):
    """
    Fourier-space top-hat window.
    
    Parameters:
        k : array_like
            Radial wavenumber.
        R : float
            Top-hat radius.
    
    Returns:
        array_like : Fourier-space top-hat, given by 2*J1(kR)/(kR).
    """
    return 2. * scipy.special.j1(k * R) / (k * R)


def get_W2D_FL(window_radius, map_shape, L=505):
    """
    Constructs a 2D Fourier-space window function for a top-hat filter.
    
    Parameters:
        window_radius : float
            The top-hat window radius in physical units (must be consistent with L).
        map_shape     : tuple
            Shape of the map (assumed square, e.g. (600,600)).
        L             : float, optional
            Physical size of the map (default is 505, as used for SLICS).
    
    Returns:
        2D numpy array representing the Fourier-space window.
    """
    N = map_shape[0]
    dx = L / N
    # Generate Fourier frequencies.
    kx = np.fft.fftshift(np.fft.fftfreq(N, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(N, dx))
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    # Convert to radial wavenumber (with 2pi factor).
    k = 2 * np.pi * np.sqrt(k2)
    # Avoid division by zero at the center.
    ind = int(N / 2)
    k[ind, ind] = 1e-7
    return top_hat_window_FL(k, window_radius)

def get_smoothed_app_pdf(mass_map, window_radius, binedges, L=505):
    """
    Applies top-hat smoothing in Fourier space at two scales and returns the PDF of the difference map.
    
    The map is filtered with a top-hat window of radius R and 2R, then the difference is computed.
    
    Parameters:
        mass_map     : 2D numpy array.
        window_radius: The smoothing scale (R) in physical units.
        binedges     : Bin edges for the histogram.
        L            : Physical size of the map (default 505 MPC/h).
    
    Returns:
        tuple : (bin_edges, pdf_counts, difference_map)
    """
    N = mass_map.shape[0]
    # Compute the Fourier-space top-hat windows.
    W2D_1 = get_W2D_FL(window_radius, mass_map.shape, L)
    W2D_2 = get_W2D_FL(window_radius * 2, mass_map.shape, L)
    
    # Fourier transform the input mass map.
    field_ft = fftshift(fftn(mass_map))
    
    # Apply the window functions in Fourier space.
    smoothed_ft1 = field_ft * W2D_1
    smoothed_ft2 = field_ft * W2D_2
    
    # Inverse Fourier transform to get back to real space.
    smoothed1 = fftshift(ifftn(ifftshift(smoothed_ft1))).real
    smoothed2 = fftshift(ifftn(ifftshift(smoothed_ft2))).real
    
    plt.imshow(smoothed1, cmap='inferno', origin='lower', vmin=-0.01, vmax=0.01)
    plt.colorbar()
    plt.title("Smoothed at R")
    plt.show()
    
    plt.imshow(smoothed2, cmap='inferno', origin='lower', vmin=-0.01, vmax=0.01)
    plt.colorbar()
    plt.title("Smoothed at 2R")
    plt.show()
    
    # Compute the difference map.
    difference_map = smoothed2 - smoothed1
    
    # Compute the histogram/PDF of the difference map.
    counts, bin_edges = np.histogram(difference_map, bins=binedges, density=True)
    return bin_edges, counts, difference_map



class CosmoSLICSLoader:
        
    def __init__(self, data_path="/feynman/home/dap/lcs/at278006/data/cosmoSLICS", L=505, map_shape=(600,600)):
        """
        Initializes the cosmoSLICS loader for top-hat smoothing.
        
        Parameters:
            data_path : str
                Base directory containing cosmoSLICS simulation maps (stored in .npy format).
            L         : float
                Physical size of the map (in MPC/h).
            map_shape : tuple
                Shape of the flat-sky map (e.g. (600,600)).
        """
        self.data_path = data_path
        self.L = L
        self.map_shape = map_shape
    
    def load_cosmoslics_map(self):
        """
        Loads the 2D flat-sky mass map from a .npy file.
        
        Returns:
            mass_map : 2D numpy array.
        """
        # Example filename convention. Adjust as necessary.
        file_path = f"{self.data_path}/true_kappa_map_DES_MocksCat_fid_a_4_Bin4_LOS1_R15.npy"
        mass_map = np.load(file_path)
        return mass_map
    
    def run_loader(self, window_radius, binedges=300):
        """
        Loads the cosmoSLICS mass map, applies top-hat smoothing at two scales (R and 2R),
        computes the difference map and its variance, and returns the PDF of the difference.
        
        Parameters:
            window_radius : float
                Smoothing scale (R) in physical units (same as used for L).
            binedges      : int or array_like
                Number of bins or bin edges for the histogram.
        
        Returns:
            tuple: (variance, difference_map, (bin_edges, pdf_counts))
                - variance: Variance of (smoothed at 2R minus smoothed at R).
                - difference_map: The 2D map of the difference.
                - (bin_edges, pdf_counts): The histogram (PDF) of the difference map.
        """
        # Load the mass map.
        mass_map = self.load_cosmoslics_map()
        
        # Get the smoothed PDF and the difference map.
        bin_edges, pdf_counts, diff_map = get_smoothed_app_pdf(mass_map, window_radius, binedges, L=self.L)
        
        # Compute the variance of the difference map.
        variance = np.var(diff_map)

        # Assume a fixed source redshift.
        source_z = 0.98
        print("Assumed source redshift (mean n(z)):", source_z)
        
        return variance, diff_map, (bin_edges, pdf_counts)