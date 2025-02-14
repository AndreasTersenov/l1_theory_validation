import numpy as np
from scipy.integrate import simpson

def calculate_moments(x, P):
    """
    Calculates the statistical moments (mean, variance, skewness) and normalization of a probability distribution.

    Parameters:
        x (numpy.ndarray): A 1D array of variable values.
        P (numpy.ndarray): A 1D array of probability densities corresponding to `x`.

    Returns:
        tuple: Contains mean, variance, skewness, and normalization of the distribution.
    """
    norm = simpson(P, x)
    mean_x = simpson(x * P, x) / norm
    variance_x = simpson(x**2 * P, x) / norm - mean_x**2
    skewness_x = (simpson(x**3 * P, x) / norm - 3 * mean_x * variance_x - mean_x**3) / variance_x**1.5
    return mean_x, variance_x, skewness_x, norm

def get_moments(kappa_values, pdf_values):
    """
    Calculates the moments (mean, variance, skewness, kurtosis) of a probability distribution function.

    Parameters:
        kappa_values (numpy.ndarray): A 1D array of kappa values.
        pdf_values (numpy.ndarray): A 1D array of PDF values corresponding to `kappa_values`.

    Returns:
        tuple: Contains mean, variance, skewness, kurtosis, and normalization of the PDF.
    """
    norm = np.trapz(pdf_values, kappa_values)
    normalized_pdf_values = pdf_values / norm
    mean_kappa = np.trapz(kappa_values * normalized_pdf_values, kappa_values)
    variance = np.trapz((kappa_values - mean_kappa)**2 * normalized_pdf_values, kappa_values)
    third_moment = np.trapz((kappa_values - mean_kappa)**3 * normalized_pdf_values, kappa_values)
    fourth_moment = np.trapz((kappa_values - mean_kappa)**4 * normalized_pdf_values, kappa_values)
    S_3 = third_moment / variance**2.0
    K = fourth_moment / variance**2 - 3
    return mean_kappa, variance, S_3, K, norm

def get_l1_from_pdf(counts, bins):
    """
    Calculates the L1 norm from a probability distribution function represented as a histogram.

    Parameters:
        counts (numpy.ndarray): The counts or heights of the histogram bins.
        bins (numpy.ndarray): The values of the bins.

    Returns:
        numpy.ndarray: L1 norm of the PDF represented by the histogram.
    """
    return counts * np.abs(bins)

def find_smallest_pair(critical_values):
    """
    Finds the pair of points with the smallest Euclidean distance between them from a set of critical values.

    Parameters:
        critical_values (numpy.ndarray): An array of critical points.

    Returns:
        tuple: The pair of points with the smallest distance and their Euclidean distance.
    """
    num_points = critical_values.shape[0]
    if num_points < 2:
        return None, float('inf')  # No pair exists

    smallest_distance = float('inf')
    smallest_pair = None

    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(critical_values[i] - critical_values[j])
            if distance < smallest_distance:
                smallest_distance = distance
                smallest_pair = (critical_values[i], critical_values[j])

    return smallest_pair
