"""
Calculations Module for CPDyAna
===============================

This module provides the core calculation functions for molecular dynamics analysis,
focusing on Mean Square Displacement (MSD) calculations and diffusion coefficient
determination for both tracer and collective diffusion.

The module implements efficient algorithms for:
- MSD calculation for individual and collective particle motion
- Linear regression analysis for diffusion coefficient extraction
- Multi-directional diffusivity analysis (X, Y, Z, XY, XZ, YZ, XYZ)
- Statistical block averaging for error estimation

Functions:
    calculate_msd: Main MSD calculation orchestrator
    msd_tracer: Individual particle MSD calculation
    msd_charged: Collective/charged species MSD calculation

Author: CPDyAna Development Team
Version: 01-02-2024
"""

import numpy as np
from scipy.stats import linregress

# FFT-based MSD calculation
def calc_fft(x):
    """
    Calculate FFT-based autocorrelation function for MSD computation.
    
    This function uses Fast Fourier Transform to efficiently compute the 
    autocorrelation function needed for mean square displacement calculations.
    
    Args:
        x (numpy.ndarray): Input array of position data for one dimension
        
    Returns:
        numpy.ndarray: Normalized autocorrelation function values
    """
    N = x.shape[0]
    F = np.fft.fft(x, n=2 * N)  # Zero-pad to 2N for proper autocorrelation
    PSD = F * F.conjugate()  # Power spectral density
    res = np.fft.ifft(PSD)  # Inverse FFT to get autocorrelation
    res = (res[:N]).real  # Take only the first N real values
    n = N * np.ones(N) - np.arange(N)  # Normalization factor
    return res / n

# Tracer MSD calculation helper
def calc_msd_tracer(r, d_idx):
    """
    Calculate tracer mean square displacement using FFT method.
    
    This function computes MSD for a single particle trajectory using
    the efficient FFT-based algorithm to avoid O(N²) complexity.
    
    Args:
        r (numpy.ndarray): Position array of shape (time_steps, 3) for one particle
        d_idx (numpy.ndarray): Array of time lag indices
        
    Returns:
        tuple: Contains:
            - MSD values for specified time lags
            - MSD components for each spatial dimension
    """
    step, dim = r.shape
    r_sq = np.square(r)  # Square of positions
    r_sq = np.append(r_sq, np.zeros((1, 3)), axis=0)  # Zero padding
    part1_c = np.zeros((3, step))
    r_sq_sum = 2 * np.sum(r_sq, axis=0)  # Initial sum for all dimensions
    
    # Calculate first part of MSD formula (direct term)
    for i in range(step):
        r_sq_sum -= r_sq[i - 1, :] + r_sq[step - i, :]
        part1_c[:, i] = r_sq_sum / (step - i)
    
    part1 = np.sum(part1_c, axis=0)  # Sum over all dimensions
    
    # Calculate second part using FFT (cross-correlation term)
    part2_c = np.array([calc_fft(r[:, i]) for i in range(r.shape[1])])
    part2 = np.sum(part2_c, axis=0)
    
    return (part1 - 2 * part2)[d_idx], (part1_c - 2 * part2_c)[:, d_idx]


def msd_tracer(structure, step):
    """
    Calculate tracer (individual particle) Mean Square Displacement.
    
    This function computes MSD for individual particle motion, which represents
    self-diffusion processes. Each particle's displacement is calculated independently
    and then averaged to obtain the tracer diffusion coefficient.
    
    Args:
        conduct_rectified_structure_array (np.ndarray): Unwrapped position array
            with shape (n_frames, n_atoms, 3).
        conduct_ions_array (list): Array of ion indices for the target element.
        dt (np.ndarray): Time step array in picoseconds.
        Last_term (int): Final time index for analysis window.
        initial_slope_time (float): Start time for diffusion coefficient fitting.
        final_slope_time (float): End time for diffusion coefficient fitting.
        block (int): Block size for statistical error estimation.
        
    Returns:
        tuple: (msd_array, time_array, diffusion_coeff, r_squared, slope, intercept)
            - msd_array (np.ndarray): MSD values vs time
            - time_array (np.ndarray): Time values in picoseconds
            - diffusion_coeff (float): Tracer diffusion coefficient in cm²/s
            - r_squared (float): Linear regression R² value
            - slope (float): MSD slope in Ų/ps
            - intercept (float): MSD intercept in Ų
            
    Note:
        The diffusion coefficient is calculated using the Einstein relation:
        D = MSD_slope / (2 * dimensionality) with appropriate unit conversion.
    """
    num_mobile_ions = structure.shape[0]
    msd_ions = np.empty([0, len(np.arange(1, step, 1))])  # Initialize empty array
    
    # Calculate MSD for each ion and collect results
    for i in range(num_mobile_ions):
        msd_i, _ = calc_msd_tracer(structure[i, :, :], np.arange(1, step, 1))
        msd_ions = np.append(msd_ions, msd_i.reshape(1, len(np.arange(1, step, 1))), axis=0)
    
    return np.average(msd_ions, axis=0)  # Return ensemble average

# Charged MSD calculation
def msd_charged(structure, final_msd):
    """
    Calculate collective (charged species) Mean Square Displacement.
    
    This function computes MSD for collective motion of charged species, which
    represents the center-of-mass diffusion and is relevant for ionic conductivity
    calculations. All particles of the same type move collectively.
    
    Args:
        conduct_rectified_structure_array (np.ndarray): Unwrapped position array
            with shape (n_frames, n_atoms, 3).
        conduct_ions_array (list): Array of ion indices for the target element.
        dt (np.ndarray): Time step array in picoseconds.
        Last_term (int): Final time index for analysis window.
        initial_slope_time (float): Start time for diffusion coefficient fitting.
        final_slope_time (float): End time for diffusion coefficient fitting.
        block (int): Block size for statistical error estimation.
        
    Returns:
        tuple: (msd_array, time_array, diffusion_coeff, r_squared, slope, intercept)
            - msd_array (np.ndarray): Collective MSD values vs time
            - time_array (np.ndarray): Time values in picoseconds
            - diffusion_coeff (float): Collective diffusion coefficient in cm²/s
            - r_squared (float): Linear regression R² value
            - slope (float): MSD slope in Ų/ps
            - intercept (float): MSD intercept in Ų
            
    Note:
        Collective diffusion is calculated as the MSD of the center of mass
        of all particles of the specified type.
    """
    num_mobile_ions = structure.shape[0]  # Number of mobile ions
    steps = structure.shape[1]           # Number of time steps
    MSD_charged = np.zeros(steps - 1)    # Initialize output array
    
    # Calculate charged MSD for each time lag
    for delt in range(1, steps):
        MSD_charged[delt-1] = (disp_sum(delt, structure) - 
                               final_msd[delt-1] * num_mobile_ions * (steps - delt)) / \
                              (0.5 * num_mobile_ions * (num_mobile_ions - 1) * (steps - delt))
    
    return MSD_charged

# Displacement sum for charged MSD
def disp_sum(delt, mobile_ion_pos):
    """
    Calculate the sum of squared displacements for a given time delta.
    
    This function computes the collective displacement term needed for
    charged MSD calculations.
    
    Args:
        delt (int): Time step difference (time lag)
        mobile_ion_pos (numpy.ndarray): 3D array of shape (num_mobile_ions, steps, 3)
    
    Returns:
        float: Sum of squared collective displacements
    """
    t = np.arange(mobile_ion_pos.shape[1] - delt)  # Valid time origins
    # Calculate displacements for all ions at time lag delt
    displacements = mobile_ion_pos[:, t + delt, :] - mobile_ion_pos[:, t, :]
    # Sum displacements over all ions, then square and sum over time and space
    return np.sum(np.square(np.sum(np.abs(displacements), axis=0)))

# Main MSD and diffusivity calculation function
def calculate_msd(diffusing_elements, diffusivity_direction_choices, diffusivity_type_choices, 
                  pos_full, conduct_rectified_structure_array, conduct_ions_array, dt, Last_term, 
                  initial_slope_time, final_slope_time, block):
    """
        Calculate Mean Square Displacement for specified elements and conditions.
        
        This function orchestrates MSD calculations for different elements, spatial directions,
        and diffusion types (tracer vs collective). It performs linear regression to extract
        diffusion coefficients and organizes results in a structured dictionary.
        
        Args:
            diffusing_elements (list): Element symbols to analyze (e.g., ['Li', 'Na']).
            diffusivity_direction_choices (list): Spatial directions for analysis
                (e.g., ['X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ']).
            diffusivity_choices (list): Types of diffusivity calculation
                (['Tracer'] for individual particles, ['Collective'] for charged species).
            pos_full (np.ndarray): Full position trajectory array.
            conduct_rectified_structure_array (np.ndarray): Unwrapped position arrays
                organized by direction and element.
            conduct_ions_array (list): Ion indices for each element and direction.
            dt (np.ndarray): Time step array in picoseconds.
            Last_term (int): Final time index for analysis.
            initial_slope_time (float): Start time for linear regression in ps.
            final_slope_time (float): End time for linear regression in ps.
            block (int): Block size for statistical averaging.
            
        Returns:
            dict: Nested dictionary with structure:
                {element: {direction: {diffusion_type: {
                    'msd_data': MSD values array,
                    'time_data': Time values array,
                    'diffusion_coeff': Diffusion coefficient in cm²/s,
                    'r_squared': Linear regression R² value,
                    'slope': MSD slope value,
                    'intercept': MSD intercept value
                }}}}
                
        Example:
            >>> msd_results = calculate_msd(
            ...     ['Li'], ['XYZ'], ['Tracer'],
            ...     pos_array, rectified_pos, ion_indices,
            ...     time_steps, last_frame, 5.0, 100.0, 500
            ... )
            >>> print(f"Li diffusion: {msd_results['Li']['XYZ']['Tracer']['diffusion_coeff']:.2e} cm²/s")
        """
    result_dict = {}
    
    # Loop over each diffusing element
    for ele in range(len(diffusing_elements)):
        element = diffusing_elements[ele]
        result_dict[element] = {}
        
        # Loop over each spatial direction
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            # Determine dimensionality for diffusivity calculation
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            
            posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            step_counts = len(posit[0, :, 0])  
            
            print(f"[DEBUG] posit shape: {posit.shape}")
            print(f"[DEBUG] step_counts: {step_counts}")
            
            # Loop over each diffusion type (Tracer, Charged)
            for diff_type_idx, diff_type in enumerate(diffusivity_type_choices):
                # Define result dictionary keys
                key_msd = f"{diff_type}_msd_array{suffix}"
                key_diff = f"{diff_type}_diffusivity{suffix}"
                key_sem = f"{diff_type}_slope_sem{suffix}"
                key_diff_err = f"{diff_type}_diffusivity_error{suffix}"
                
                if diff_type == "Tracer":
                    msd_array = msd_tracer(posit, step_counts)
                elif diff_type == "Charged":
                    tracer_msd = msd_tracer(posit, step_counts)
                    msd_array = msd_charged(posit, tracer_msd)
                else:
                    raise ValueError(f"Unknown diffusivity type: {diff_type}")
                
                print(f"[DEBUG] MSD array length: {len(msd_array)}")
                print(f"[DEBUG] dt length: {len(dt)}")
                
                # Find indices for linear regression time window
                first_idx = np.searchsorted(dt, initial_slope_time, side='left')
                last_idx = np.searchsorted(dt, final_slope_time, side='right') - 1
                
                # Perform linear regression to extract diffusivity
                if first_idx <= last_idx and last_idx < len(msd_array):
                    slope, _, _, _, std_err = linregress(dt[first_idx:last_idx + 1], 
                                                         msd_array[first_idx:last_idx + 1])
                    diffusivity = (slope * 1e-4) / (2 * d)  # Convert Å²/ps to cm²/s
                else:
                    diffusivity = np.nan
                
                B = step_counts // block  # Number of complete blocks
                slope_blocks = []
                
                # Calculate diffusivity for each block
                for b in range(B):
                    t_start = b * block
                    t_end = (b + 1) * block
                    if t_end > step_counts:
                        break
                    
                    pos_block = posit[:, t_start:t_end, :]  
                    step_block = t_end - t_start
                    
                    # Calculate MSD for current block
                    if diff_type == "Tracer":
                        msd_block = msd_tracer(pos_block, step_block)
                    elif diff_type == "Charged":
                        tracer_msd_block = msd_tracer(pos_block, step_block)
                        msd_block = msd_charged(pos_block, tracer_msd_block)
                    
                    # Create time array for block and perform regression
                    dt_block = np.arange(1, step_block) * (dt[1] - dt[0])
                    if len(dt_block) >= 2:
                        slope_block, _, _, _, _ = linregress(dt_block, msd_block)
                        slope_blocks.append(slope_block)
                
                # Calculate block statistics for error estimation
                if len(slope_blocks) > 1:
                    slope_mean = np.mean(slope_blocks)
                    slope_std = np.std(slope_blocks)
                    slope_sem = slope_std / np.sqrt(len(slope_blocks) - 1)
                    diffusivity_block_mean = (slope_mean * 1e-4) / (2 * d)
                    diffusivity_sem = (slope_sem * 1e-4) / (2 * d)
                else:
                    diffusivity_block_mean = np.nan
                    diffusivity_sem = np.nan
                    slope_sem = np.nan
                
                # Store results in nested dictionary structure
                result_dict[element].setdefault(key_msd, []).append(msd_array)
                result_dict[element].setdefault(key_diff, []).append(diffusivity)
                result_dict[element].setdefault(key_sem, []).append(slope_sem if 'slope_sem' in locals() else np.nan)
                result_dict[element].setdefault(key_diff_err, []).append(diffusivity_sem)
    
    return result_dict

def calc_ngp_tracer(r, d, d_idx):
    """
    Calculate Non-Gaussian Parameter (NGP) for tracer diffusion.
    
    The NGP, alpha_2(t), measures dynamic heterogeneity in particle diffusion
    as per the formula: alpha_2(t) = [d * <[r(t)-r(0)]^4>] / [(d+2) * <[r(t)-r(0)]^2>^2] - 1
    
    Parameters
    ----------
    r : np.ndarray
        Position array of shape (time_steps, 3) for one particle.
    d : int
        Dimensionality of the system (1, 2, or 3 based on direction).
    d_idx : np.ndarray
        Array of time lag indices.
    
    Returns
    -------
    np.ndarray
        Array of NGP values for specified time lags.
    """
    step = r.shape[0]
    ngp = np.zeros(len(d_idx))
    
    for i, delta in enumerate(d_idx):
        # Calculate displacements for all possible time origins
        disp = r[delta:, :] - r[:-delta, :]
        # Calculate r^2 and r^4 for the displacement
        r2 = np.sum(disp**2, axis=1)  # Shape: (step - delta,)
        r4 = r2**2  # Fourth power of displacement magnitude
        
        # Compute averages over time origins
        mean_r2 = np.mean(r2)
        mean_r4 = np.mean(r4)
        
        # Compute NGP using the formula
        if mean_r2 > 0:
            numerator = d * mean_r4
            denominator = (d + 2) * (mean_r2**2)
            ngp[i] = (numerator / denominator) - 1
        else:
            ngp[i] = 0.0  # Avoid division by zero
    
    return ngp

def calculate_ngp(diffusing_elements, diffusivity_direction_choices, pos_full, conduct_rectified_structure_array, conduct_ions_array, dt):
    """
    Calculate Non-Gaussian Parameter (NGP) for specified elements and directions.
    
    This function computes the NGP to measure dynamic heterogeneity in particle diffusion
    for different elements and spatial directions. Results are organized in a structured dictionary.
    
    Parameters
    ----------
    diffusing_elements : list
        Element symbols to analyze (e.g., ['Li', 'Na']).
    diffusivity_direction_choices : list
        Spatial directions for analysis (e.g., ['X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ']).
    pos_full : np.ndarray
        Full position trajectory array.
    conduct_rectified_structure_array : np.ndarray
        Unwrapped position arrays organized by direction and element.
    conduct_ions_array : list
        Ion indices for each element and direction.
    dt : np.ndarray
        Time step array in picoseconds.
    
    Returns
    -------
    dict
        Nested dictionary with structure:
        {element: {direction: {
            'ngp_data': NGP values array,
            'time_data': Time values array
        }}}
    
    Example
    -------
    >>> ngp_results = calculate_ngp(
    ...     ['Li'], ['XYZ'],
    ...     pos_array, rectified_pos, ion_indices, time_steps
    ... )
    >>> print(f"Li NGP: {ngp_results['Li']['XYZ']['ngp_data'][:5]}")
    """
    result_dict = {}
    
    # Loop over each diffusing element
    for ele in range(len(diffusing_elements)):
        element = diffusing_elements[ele]
        result_dict[element] = {}
        
        # Loop over each spatial direction
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            # Determine dimensionality for NGP calculation
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            step_counts = len(posit[0, :, 0])
            print(f"[DEBUG] posit shape for NGP: {posit.shape}")
            print(f"[DEBUG] step_counts for NGP: {step_counts}")
            
            # Calculate NGP for each particle and average
            num_mobile_ions = posit.shape[0]
            ngp_ions = np.empty([0, len(np.arange(1, step_counts, 1))])
            for i in range(num_mobile_ions):
                ngp_i = calc_ngp_tracer(posit[i, :, :], d, np.arange(1, step_counts, 1))
                ngp_ions = np.append(ngp_ions, ngp_i.reshape(1, len(np.arange(1, step_counts, 1))), axis=0)
            ngp_array = np.average(ngp_ions, axis=0)
            
            print(f"[DEBUG] NGP array length: {len(ngp_array)}")
            print(f"[DEBUG] dt length for NGP: {len(dt)}")
            
            # Store results in nested dictionary structure
            key_ngp = f"NGP_array{suffix}"
            result_dict[element].setdefault(key_ngp, []).append(ngp_array)
    
    return result_dict