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
from multiprocessing import Pool

# FFT-based MSD calculation
def calc_fft(x):
    N = x.shape[0]
    F = np.fft.fft(x, n=2 * N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(N)
    return res / n

# Tracer MSD calculation helper
def calc_msd_tracer(r, d_idx):
    step, dim = r.shape
    r_sq = np.square(r)
    r_sq = np.append(r_sq, np.zeros((1, 3)), axis=0)
    part1_c = np.zeros((3, step))
    r_sq_sum = 2 * np.sum(r_sq, axis=0)
    for i in range(step):
        r_sq_sum -= r_sq[i - 1, :] + r_sq[step - i, :]
        part1_c[:, i] = r_sq_sum / (step - i)
    part1 = np.sum(part1_c, axis=0)
    part2_c = np.array([calc_fft(r[:, i]) for i in range(r.shape[1])])
    part2 = np.sum(part2_c, axis=0)
    return (part1 - 2 * part2)[d_idx], (part1_c - 2 * part2_c)[:, d_idx]


def msd_tracer(structure, step):
    num_mobile_ions = structure.shape[0]
    msd_ions = np.empty([0, len(np.arange(1, step, 1))])
    for i in range(num_mobile_ions):
        msd_i, _ = calc_msd_tracer(structure[i, :, :], np.arange(1, step, 1))
        msd_ions = np.append(msd_ions, msd_i.reshape(1, len(np.arange(1, step, 1))), axis=0)
    return np.average(msd_ions, axis=0)

# Charged MSD calculation
def msd_charged(structure, final_msd):
    """
    Calculate the charged MSD for a given 3D position array and tracer MSD.
    
    Parameters:
    - structure: 3D numpy array of shape (num_mobile_ions, steps, 3) containing position data.
    - final_msd: 1D numpy array containing the tracer MSD values.
    
    Returns:
    - MSD_charged: 1D numpy array of charged MSD values.
    """
    num_mobile_ions = structure.shape[0]  # Number of mobile ions
    steps = structure.shape[1]           # Number of time steps
    MSD_charged = np.zeros(steps - 1)    # Initialize output array
    
    for delt in range(1, steps):
        MSD_charged[delt-1] = (disp_sum(delt, structure) - 
                               final_msd[delt-1] * num_mobile_ions * (steps - delt)) / \
                              (0.5 * num_mobile_ions * (num_mobile_ions - 1) * (steps - delt))
    
    return MSD_charged

# Displacement sum for charged MSD
def disp_sum(delt, mobile_ion_pos):
    """
    Calculate the sum of squared displacements for a given time delta.
    
    Parameters:
    - delt: Integer time step difference.
    - mobile_ion_pos: 3D numpy array of shape (num_mobile_ions, steps, 3).
    
    Returns:
    - Float representing the sum of squared displacements.
    """
    t = np.arange(mobile_ion_pos.shape[1] - delt)
    displacements = mobile_ion_pos[:, t + delt, :] - mobile_ion_pos[:, t, :]
    return np.sum(np.square(np.sum(np.abs(displacements), axis=0)))

# Main MSD and diffusivity calculation function
def calculate_msd(diffusing_elements, diffusivity_direction_choices, diffusivity_type_choices, 
                  pos_full, posit, mobile_ion, dt, last, initial_slope_time, final_slope_time, block, is_lammps=False, dt_value=None, lammps_units='metal'):
    result_dict = {}

    if is_lammps:
        # For different LAMMPS unit systems:
        # - metal: time in ps, distance in Å -> 1e-4 conversion to cm²/s
        # - real: time in fs, distance in Å -> 1e-7 conversion to cm²/s
        # - si: time in s, distance in m -> 1e+4 conversion to cm²/s
        # - lj: dimensionless -> use 1.0 
        unit_conversions = {
            "metal": 1e-4,  # Å²/ps to cm²/s
            "real": 1e-7,   # Å²/fs to cm²/s
            "si": 1e+4,     # m²/s to cm²/s
            "lj": 1.0       # dimensionless
        }
        conv_factor = unit_conversions.get(lammps_units, 1e-4)
    else:
        # Default for QE files
        conv_factor = 1e-4  # Å²/ps to cm²/s
    
    for ele in range(len(diffusing_elements)):
        element = diffusing_elements[ele]
        result_dict[element] = {}
        
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            step_counts = len(posit[direction_idx, 0, :, 0])  # Total time steps
            
            for diff_type_idx, diff_type in enumerate(diffusivity_type_choices):
                key_msd = f"{diff_type}_msd_array{suffix}"
                key_diff = f"{diff_type}_diffusivity{suffix}"
                key_sem = f"{diff_type}_slope_sem{suffix}"
                key_diff_err = f"{diff_type}_diffusivity_error{suffix}"
                
                if diff_type == "Tracer":
                    msd_array = msd_tracer(posit[direction_idx, :, :, :], step_counts)
                elif diff_type == "Charged":
                    tracer_msd = msd_tracer(posit[direction_idx, :, :, :], step_counts)
                    msd_array = msd_charged(posit[direction_idx, :, :, :], tracer_msd)
                else:
                    raise ValueError(f"Unknown diffusivity type: {diff_type}")
                
                # Fit diffusivity for the full trajectory
                first_idx = np.searchsorted(dt, initial_slope_time, side='left')
                last_idx = np.searchsorted(dt, final_slope_time, side='right') - 1
                if first_idx <= last_idx:
                    slope, _, _, _, std_err = linregress(dt[first_idx:last_idx + 1], 
                                                         msd_array[first_idx:last_idx + 1])
                    diffusivity = (slope * conv_factor) / (2 * d)  # Convert Å²/ps to cm²/s
                else:
                    diffusivity = np.nan
                
                # Block analysis for error estimation
                M = block  # Block size in time steps
                B = step_counts // M  # Number of complete blocks
                slope_blocks = []
                
                for b in range(B):
                    t_start = b * M
                    t_end = (b + 1) * M
                    if t_end > step_counts:
                        break
                    pos_block = posit[direction_idx, :, t_start:t_end, :]
                    step_block = t_end - t_start
                    if diff_type == "Tracer":
                        msd_block = msd_tracer(pos_block, step_block)
                    elif diff_type == "Charged":
                        tracer_msd_block = msd_tracer(pos_block, step_block)
                        msd_block = msd_charged(pos_block, tracer_msd_block)
                    dt_block = np.arange(1, step_block) * (dt[1] - dt[0])
                    if len(dt_block) >= 2:
                        slope_block, _, _, _, _ = linregress(dt_block, msd_block)
                        slope_blocks.append(slope_block)
                
                if len(slope_blocks) > 1:
                    slope_mean = np.mean(slope_blocks)
                    slope_std = np.std(slope_blocks)
                    slope_sem = slope_std / np.sqrt(len(slope_blocks) - 1)
                    diffusivity_block_mean = (slope_mean * 1e-4) / (2 * d)
                    diffusivity_sem = (slope_sem * 1e-4) / (2 * d)
                else:
                    diffusivity_block_mean = np.nan
                    diffusivity_sem = np.nan
                
                result_dict[element].setdefault(key_msd, []).append(msd_array)
                result_dict[element].setdefault(key_diff, []).append(diffusivity)
                result_dict[element].setdefault(key_sem, []).append(slope_sem if 'slope_sem' in locals() else np.nan)
                result_dict[element].setdefault(key_diff_err, []).append(diffusivity_sem)
    
    return result_dict

def calc_ngp_tracer(r, d, d_idx):
    """
    Calculate Non-Gaussian Parameter (NGP) for tracer diffusion.
    
    Args:
        r (np.ndarray): Position array (time_steps, 3) for one particle
        d (int): Dimensionality (1, 2, or 3)
        d_idx (np.ndarray): Array of time lag indices
    
    Returns:
        np.ndarray: NGP values for specified time lags
    """
    step = r.shape[0]
    ngp = np.zeros(len(d_idx))
    
    for i, delta in enumerate(d_idx):
        disp = r[delta:, :] - r[:-delta, :]
        r2 = np.sum(disp**2, axis=1)
        r4 = r2**2
        mean_r2 = np.mean(r2)
        mean_r4 = np.mean(r4)
        
        if mean_r2 > 0:
            numerator = d * mean_r4
            denominator = (d + 2) * (mean_r2**2)
            ngp[i] = (numerator / denominator) - 1
        else:
            ngp[i] = 0.0
    
    return ngp

def calculate_ngp(diffusing_elements, diffusivity_direction_choices, pos_full, 
                  conduct_rectified_structure_array, conduct_ions_array, dt, 
                  initial_time=2.0, final_time=200.0):
    """
    Calculate Non-Gaussian Parameter (NGP) for specified elements and directions.
    
    Args:
        diffusing_elements (list): Elements to analyze (e.g., ['Li'])
        diffusivity_direction_choices (list): Directions (e.g., ['XYZ'])
        pos_full (np.ndarray): Full position trajectory
        conduct_rectified_structure_array (np.ndarray): Unwrapped positions
        conduct_ions_array (list): Ion indices
        dt (np.ndarray): Time step array (ps)
        initial_time (float): Start time for analysis (ps)
        final_time (float): End time for analysis (ps)
    
    Returns:
        dict: Nested dictionary with NGP and time lag data
    """
    result_dict = {}
    time_step = dt[1] - dt[0]
    time_lags = np.arange(1, len(dt)) * time_step
    mask = (time_lags >= initial_time) & (time_lags <= final_time)
    d_idx = np.arange(1, len(dt))[mask]
    
    for ele in range(len(diffusing_elements)):
        element = diffusing_elements[ele]
        result_dict[element] = {}
        
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            step_counts = len(posit[0, :, 0])
            
            print(f"[DEBUG] posit shape for NGP: {posit.shape}")
            print(f"[DEBUG] step_counts for NGP: {step_counts}")
            
            num_mobile_ions = posit.shape[0]
            with Pool() as pool:
                ngp_ions = pool.starmap(calc_ngp_tracer, 
                                       [(posit[i, :, :], d, d_idx) for i in range(num_mobile_ions)])
            ngp_ions = np.array(ngp_ions)
            ngp_array = np.average(ngp_ions, axis=0)
            
            print(f"[DEBUG] NGP array length: {len(ngp_array)}")
            print(f"[DEBUG] dt length for NGP: {len(dt)}")
            
            key_ngp = f"NGP_array{suffix}"
            result_dict[element].setdefault(key_ngp, []).append(ngp_array)
            result_dict[element].setdefault('time_lags' + suffix, []).append(time_lags[mask])
    
    return result_dict