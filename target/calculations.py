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
Version: 06-25-2025 (Unified TP_slope method)
"""

import numpy as np
from scipy.stats import linregress
from multiprocessing import Pool

# FFT-based MSD calculation
def calc_fft(x):
    """FFT-based calculation for efficient MSD computation."""
    N = x.shape[0]
    F = np.fft.fft(x, n=2 * N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(N)
    return res / n

# Tracer MSD calculation helper
def calc_msd_tracer(r, d_idx):
    """Calculate MSD for a single particle trajectory using efficient algorithm."""
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
    """Calculate average tracer MSD for all mobile ions."""
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

# UNIFIED MSD and diffusivity calculation function (TP_slope method)
def calculate_msd(elements, diffusivity_direction_choices, diffusivity_choices,
                  pos_full, conduct_rectified_structure_array, conduct_ions_array,
                  t, Last_term, initial_slope_time, final_slope_time, block,
                  is_lammps=False, dt_value=1.0, lammps_units="metal", atom_types=None, cell_param_full=None):
    """
    UNIFIED MSD calculation using TP_slope method for both LAMMPS and QE files.
    This ensures consistent results and proper diffusion coefficient values.
    """
    result_dict = {}
    
    print(f"=== UNIFIED MSD CALCULATION (TP_slope method) ===")
    print(f"Processing {len(elements)} element(s): {elements}")
    print(f"Directions: {diffusivity_direction_choices}")
    print(f"Types: {diffusivity_choices}")
    print(f"Is LAMMPS: {is_lammps}, Units: {lammps_units if is_lammps else 'QE'}")
    print(f"Block size: {block}")
    print(f"Slope fitting: {initial_slope_time} - {final_slope_time} ps")
    
    # Create time arrays based on file type
    if is_lammps:
        # For LAMMPS, create time array using dt_value
        n_frames = conduct_rectified_structure_array.shape[2] if len(conduct_rectified_structure_array.shape) == 4 else len(t)
        dt_array = np.arange(1, n_frames) * dt_value
        print(f"LAMMPS time array: {dt_array[0]:.3f} to {dt_array[-1]:.3f} ps (dt={dt_value} ps)")
    else:
        # For QE, use existing time array
        dt_array = t[1:] if len(t) > 1 else np.array([1.0])
        print(f"QE time array: {dt_array[0]:.3f} to {dt_array[-1]:.3f} ps")
    
    # Process each element
    for ele_idx, element in enumerate(elements):
        result_dict[element] = {}
        print(f"\nProcessing element: {element}")
        
        # Process each direction
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            
            # Extract structure for this direction and element
            if len(conduct_rectified_structure_array.shape) == 4:
                # Structure is organized as [direction, ion, time, xyz]
                posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            else:
                # Structure is [ion, time, xyz] - use for all directions
                posit = conduct_rectified_structure_array
            
            step_counts = posit.shape[1]  # Number of time steps
            print(f"  Direction: {direction}, Steps: {step_counts}, Dimensionality: {d}D")
            
            # Process each diffusivity type
            for diff_type_idx, diff_type in enumerate(diffusivity_choices):
                key_msd = f"{diff_type}_msd_array{suffix}"
                key_time = f"{diff_type}_time_array{suffix}"
                key_diffusivity = f"{diff_type}_diffusivity{suffix}"
                key_diffusivity_error = f"{diff_type}_diffusivity_error{suffix}"
                
                print(f"    Calculating {diff_type} MSD for {direction}")
                
                try:
                    # Calculate MSD using TP_slope method
                    if diff_type == "Tracer":
                        msd_array = msd_tracer(posit, step_counts)
                    elif diff_type == "Charged" or diff_type == "Collective":
                        tracer_msd_array = msd_tracer(posit, step_counts)
                        msd_array = msd_charged(posit, tracer_msd_array)
                    else:
                        print(f"    Warning: Unknown diffusivity type {diff_type}")
                        continue
                    
                    # Use appropriate time array
                    time_array = dt_array[:len(msd_array)]
                    
                    print(f"    MSD array length: {len(msd_array)}")
                    print(f"    Time array length: {len(time_array)}")
                    print(f"    MSD range: {msd_array[0]:.3f} - {msd_array[-1]:.3f} Ų")
                    
                    # Store MSD and time arrays
                    result_dict[element].setdefault(key_msd, []).append(msd_array)
                    result_dict[element].setdefault(key_time, []).append(time_array)
                    
                    # Calculate diffusion coefficient using linear regression (TP_slope method)
                    try:
                        # Find fitting window indices
                        first_idx = np.searchsorted(time_array, initial_slope_time, side='left')
                        last_idx = np.searchsorted(time_array, final_slope_time, side='right') - 1
                        
                        if first_idx < last_idx and last_idx < len(msd_array):
                            fit_times = time_array[first_idx:last_idx + 1]
                            fit_msd = msd_array[first_idx:last_idx + 1]
                            
                            # Remove any NaN or infinite values
                            valid_mask = np.isfinite(fit_times) & np.isfinite(fit_msd)
                            fit_times = fit_times[valid_mask]
                            fit_msd = fit_msd[valid_mask]
                            
                            if len(fit_times) > 3:  # Need at least 4 points for fitting
                                # Linear regression: MSD = 2*d*D*t 
                                slope, intercept, r_value, p_value, std_err = linregress(fit_times, fit_msd)
                                diffusivity = (slope * 1e-4) / (2 * d)  # Convert Å²/ps to cm²/s
                                
                                print(f"    Slope: {slope:.3e} Ų/ps")
                                print(f"    Diffusion coefficient: {diffusivity:.3e} cm²/s")
                                print(f"    R² = {r_value**2:.4f}")
                                print(f"    Fitting window: {fit_times[0]:.1f} - {fit_times[-1]:.1f} ps ({len(fit_times)} points)")
                            else:
                                print(f"    Warning: Too few points for fitting ({len(fit_times)} points)")
                                diffusivity = np.nan
                                std_err = np.nan
                        else:
                            print(f"    Warning: Invalid fitting window [{initial_slope_time}, {final_slope_time}] ps")
                            diffusivity = np.nan
                            std_err = np.nan
                    
                    except Exception as e:
                        print(f"    Error calculating diffusion coefficient: {e}")
                        diffusivity = np.nan
                        std_err = np.nan
                    
                    # Block analysis for error estimation (TP_slope method)
                    diffusivity_sem = np.nan
                    try:
                        print(f"    Performing block analysis with block size: {block}")
                        M = max(1, min(block, step_counts // 4))  # Ensure reasonable block size
                        B = step_counts // M  # Number of complete blocks
                        slope_blocks = []
                        
                        for b in range(B):
                            t_start = b * M
                            t_end = (b + 1) * M
                            if t_end > step_counts:
                                break
                                
                            # Extract block data
                            pos_block = posit[:, t_start:t_end, :]
                            step_block = t_end - t_start
                            
                            if step_block < 10:  # Need minimum points for MSD
                                continue
                            
                            # Calculate MSD for this block
                            if diff_type == "Tracer":
                                msd_block = msd_tracer(pos_block, step_block)
                            elif diff_type == "Charged" or diff_type == "Collective":
                                tracer_msd_block = msd_tracer(pos_block, step_block)
                                msd_block = msd_charged(pos_block, tracer_msd_block)
                            
                            # Create time array for this block
                            if is_lammps:
                                dt_block = np.arange(1, len(msd_block) + 1) * dt_value
                            else:
                                time_step = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
                                dt_block = np.arange(1, len(msd_block) + 1) * time_step
                            
                            # Fit slope for this block (using middle portion for stability)
                            if len(dt_block) >= 6:
                                mid_start = len(dt_block) // 4
                                mid_end = 3 * len(dt_block) // 4
                                if mid_end > mid_start + 2:
                                    slope_block, _, _, _, _ = linregress(dt_block[mid_start:mid_end], 
                                                                       msd_block[mid_start:mid_end])
                                    slope_blocks.append(slope_block)
                        
                        # Calculate block statistics
                        if len(slope_blocks) > 1:
                            slope_mean = np.mean(slope_blocks)
                            slope_std = np.std(slope_blocks, ddof=1)
                            slope_sem = slope_std / np.sqrt(len(slope_blocks))
                            diffusivity_block_mean = (slope_mean * 1e-4) / (2 * d)
                            diffusivity_sem = (slope_sem * 1e-4) / (2 * d)
                            
                            print(f"    Block analysis: {len(slope_blocks)} blocks")
                            print(f"    Block diffusion: {diffusivity_block_mean:.3e} ± {diffusivity_sem:.3e} cm²/s")
                            
                            # Use block analysis results for error
                            if not np.isnan(diffusivity):
                                diffusivity_error = diffusivity_sem
                            else:
                                diffusivity = diffusivity_block_mean
                                diffusivity_error = diffusivity_sem
                        else:
                            print(f"    Block analysis failed: only {len(slope_blocks)} valid blocks")
                            diffusivity_error = std_err * 1e-4 / (2 * d) if not np.isnan(std_err) else np.nan
                            
                    except Exception as e:
                        print(f"    Block analysis error: {e}")
                        diffusivity_error = std_err * 1e-4 / (2 * d) if not np.isnan(std_err) else np.nan
                    
                    # Store diffusion coefficients
                    result_dict[element].setdefault(key_diffusivity, []).append(diffusivity)
                    result_dict[element].setdefault(key_diffusivity_error, []).append(diffusivity_error)
                    
                    print(f"    Final result: D = {diffusivity:.3e} ± {diffusivity_error:.3e} cm²/s")
                    
                except Exception as e:
                    print(f"    Error in MSD calculation for {diff_type}: {e}")
                    # Store empty/NaN data
                    result_dict[element].setdefault(key_msd, []).append(np.array([np.nan]))
                    result_dict[element].setdefault(key_time, []).append(np.array([1.0]))
                    result_dict[element].setdefault(key_diffusivity, []).append(np.nan)
                    result_dict[element].setdefault(key_diffusivity_error, []).append(np.nan)
        
        # Create summary data in the expected QE format for plotting compatibility
        if 'Tracer_msd_array' in result_dict[element] and len(result_dict[element]['Tracer_msd_array']) > 0:
            # Use the first (and typically only) entry for each array
            result_dict[element]['msd_mean'] = result_dict[element]['Tracer_msd_array'][0]
            result_dict[element]['time_ps'] = result_dict[element]['Tracer_time_array'][0]
            
            # Create diffusion_data structure in QE format
            result_dict[element]['diffusion_data'] = {}
            for diff_type in diffusivity_choices:
                for direction in diffusivity_direction_choices:
                    suffix = '' if direction == 'XYZ' else f'_{direction}'
                    diff_key = f"{diff_type}_{direction}"
                    
                    key_diffusivity = f"{diff_type}_diffusivity{suffix}"
                    key_diffusivity_error = f"{diff_type}_diffusivity_error{suffix}"
                    
                    if key_diffusivity in result_dict[element] and len(result_dict[element][key_diffusivity]) > 0:
                        diffusion_mean = result_dict[element][key_diffusivity][0]
                        diffusion_sem = result_dict[element][key_diffusivity_error][0]
                        
                        result_dict[element]['diffusion_data'][diff_key] = {
                            'diffusion_mean': diffusion_mean,
                            'diffusion_std': diffusion_sem,  # Use SEM as std for consistency
                            'diffusion_sem': diffusion_sem
                        }
            
            # Also store in the new format for compatibility
            if 'Tracer_diffusivity' in result_dict[element] and len(result_dict[element]['Tracer_diffusivity']) > 0:
                result_dict[element]['Tracer_diffusivity'] = result_dict[element]['Tracer_diffusivity']
                result_dict[element]['Tracer_diffusivity_error'] = result_dict[element]['Tracer_diffusivity_error']
            else:
                # Fallback to diffusion_data if available
                if 'diffusion_data' in result_dict[element] and 'Tracer_XYZ' in result_dict[element]['diffusion_data']:
                    tracer_data = result_dict[element]['diffusion_data']['Tracer_XYZ']
                    result_dict[element]['Tracer_diffusivity'] = [tracer_data['diffusion_mean']]
                    result_dict[element]['Tracer_diffusivity_error'] = [tracer_data['diffusion_sem']]
        
        print(f"MSD calculation completed for {element}")
    
    print(f"=== UNIFIED MSD CALCULATION COMPLETE ===")
    return result_dict

# NGP calculation function
def calc_ngp_tracer(r, d, d_idx):
    """Calculate non-Gaussian parameter for a single trajectory."""
    step = r.shape[0]
    
    # Calculate MSD components
    msd_ions, _ = calc_msd_tracer(r, d_idx)
    
    # Calculate fourth moment
    fourth_moment = np.zeros(len(d_idx))
    for i, dt in enumerate(d_idx):
        displacements = np.zeros((step - dt, 3))
        for t in range(step - dt):
            displacements[t] = r[t + dt] - r[t]
        
        # Fourth moment of displacement
        r4 = np.sum(displacements**4, axis=1)
        fourth_moment[i] = np.mean(r4)
    
    # NGP calculation
    ngp = (d / (d + 2)) * (fourth_moment / (msd_ions**2)) - 1.0
    
    return ngp

def calculate_ngp(diffusing_elements, diffusivity_direction_choices, pos_full, 
                  conduct_rectified_structure_array, conduct_ions_array, dt, 
                  initial_time=2.0, final_time=200.0):
    """Calculate Non-Gaussian Parameter for diffusing elements."""
    result_dict = {}
    
    # Create time mask
    time_lags = np.arange(1, len(dt))
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