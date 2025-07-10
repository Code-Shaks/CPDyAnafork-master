"""
Calculations Module for CPDyAna
===============================

This module provides the core calculation functions for molecular dynamics analysis,
focusing on Mean Square Displacement (MSD) calculations, diffusion coefficient
determination, and Non-Gaussian Parameter (NGP) analysis for both tracer and collective diffusion.

The module implements efficient algorithms for:
- MSD calculation for individual and collective particle motion
- Linear regression analysis for diffusion coefficient extraction
- Multi-directional diffusivity analysis (X, Y, Z, XY, XZ, YZ, XYZ)
- Statistical block averaging for error estimation
- Non-Gaussian Parameter (NGP) calculation for dynamical heterogeneity

Functions:
    calculate_msd: Main MSD calculation orchestrator (LAMMPS and QE support)
    msd_tracer: Individual particle MSD calculation
    msd_charged: Collective/charged species MSD calculation
    calculate_ngp: Non-Gaussian Parameter calculation for tracer diffusion

Author: CPDyAna Development Team
Version: 06-25-2025
"""

import numpy as np
from scipy.stats import linregress
from multiprocessing import Pool

# FFT-based MSD calculation
def calc_fft(x):
    """
    Efficiently compute the autocorrelation using FFT for MSD calculation.

    Args:
        x (np.ndarray): 1D array of positions for a single coordinate.

    Returns:
        np.ndarray: Autocorrelation result for MSD computation.
    """
    N = x.shape[0]
    F = np.fft.fft(x, n=2 * N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(N)
    return res / n

# Tracer MSD calculation helper
def calc_msd_tracer(r, d_idx):
    """
    Calculate MSD for a single particle trajectory using an efficient algorithm.

    Args:
        r (np.ndarray): Trajectory array (steps, 3) for one particle.
        d_idx (np.ndarray): Array of time lag indices.

    Returns:
        tuple: (MSD array for selected lags, MSD array per dimension for selected lags)
    """
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
    """
    Calculate average tracer MSD for all mobile ions.

    Args:
        structure (np.ndarray): Array of shape (num_mobile_ions, steps, 3).
        step (int): Number of time steps.

    Returns:
        np.ndarray: Average tracer MSD over all ions.
    """
    num_mobile_ions = structure.shape[0]
    msd_ions = np.empty([0, len(np.arange(1, step, 1))])
    for i in range(num_mobile_ions):
        msd_i, _ = calc_msd_tracer(structure[i, :, :], np.arange(1, step, 1))
        msd_ions = np.append(msd_ions, msd_i.reshape(1, len(np.arange(1, step, 1))), axis=0)
    return np.average(msd_ions, axis=0)

def msd_charged(structure, final_msd):
    """
    Calculate the charged MSD for a given 3D position array and tracer MSD.

    Args:
        structure (np.ndarray): Array of shape (num_mobile_ions, steps, 3).
        final_msd (np.ndarray): Tracer MSD array.

    Returns:
        np.ndarray: Charged MSD values.
    """
    num_mobile_ions = structure.shape[0]
    steps = structure.shape[1]
    MSD_charged = np.zeros(steps - 1)
    for delt in range(1, steps):
        MSD_charged[delt-1] = (disp_sum(delt, structure) -
                               final_msd[delt-1] * num_mobile_ions * (steps - delt)) / \
                              (0.5 * num_mobile_ions * (num_mobile_ions - 1) * (steps - delt))
    return MSD_charged

def disp_sum(delt, mobile_ion_pos):
    """
    Calculate the sum of squared displacements for a given time delta.

    Args:
        delt (int): Time step difference.
        mobile_ion_pos (np.ndarray): Array of shape (num_mobile_ions, steps, 3).

    Returns:
        float: Sum of squared displacements.
    """
    t = np.arange(mobile_ion_pos.shape[1] - delt)
    displacements = mobile_ion_pos[:, t + delt, :] - mobile_ion_pos[:, t, :]
    return np.sum(np.square(np.sum(np.abs(displacements), axis=0)))

def calculate_msd_lammps(elements, diffusivity_direction_choices, diffusivity_choices,
                         pos_full, conduct_rectified_structure_array, conduct_ions_array,
                         t, Last_term, initial_slope_time, final_slope_time, block,
                         dt_value=1.0, lammps_units="metal", atom_types=None, cell_param_full=None):
    """
    Calculate MSD and diffusivity for LAMMPS-style trajectory data.

    Args:
        elements (list): List of diffusing elements.
        diffusivity_direction_choices (list): Directions for analysis.
        diffusivity_choices (list): Types of diffusivity ('Tracer', 'Charged', etc.).
        pos_full, conduct_rectified_structure_array, conduct_ions_array: Trajectory and ion data.
        t (np.ndarray): Time array.
        Last_term (int): Last index for analysis.
        initial_slope_time (float): Start time for slope fitting.
        final_slope_time (float): End time for slope fitting.
        block (int): Block size for error estimation.
        dt_value (float): Time step value.
        lammps_units (str): LAMMPS units.
        atom_types, cell_param_full: Optional atom/cell info.

    Returns:
        dict: Nested dictionary with MSD, diffusivity, and error estimates.
    """
    result_dict = {}
    n_frames = conduct_rectified_structure_array.shape[2] if len(conduct_rectified_structure_array.shape) == 4 else len(t)
    dt_array = np.arange(1, n_frames) * dt_value

    for ele_idx, element in enumerate(elements):
        result_dict[element] = {}
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            if len(conduct_rectified_structure_array.shape) == 4:
                posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            else:
                posit = conduct_rectified_structure_array
            step_counts = posit.shape[1]
            for diff_type_idx, diff_type in enumerate(diffusivity_choices):
                key_msd = f"{diff_type}_msd_array{suffix}"
                key_time = f"{diff_type}_time_array{suffix}"
                key_diffusivity = f"{diff_type}_diffusivity{suffix}"
                key_diffusivity_error = f"{diff_type}_diffusivity_error{suffix}"
                key_slope_sem = f"{diff_type}_slope_sem{suffix}"

                # MSD calculation for each type
                if diff_type == "Tracer":
                    msd_array = msd_tracer(posit, step_counts)
                elif diff_type == "Charged" or diff_type == "Collective":
                    tracer_msd_array = msd_tracer(posit, step_counts)
                    msd_array = msd_charged(posit, tracer_msd_array)
                else:
                    continue

                time_array = dt_array[:len(msd_array)]
                result_dict[element].setdefault(key_msd, []).append(msd_array)
                result_dict[element].setdefault(key_time, []).append(time_array)

                # Linear regression for diffusivity
                first_idx = np.searchsorted(time_array, initial_slope_time, side='left')
                last_idx = np.searchsorted(time_array, final_slope_time, side='right') - 1
                if first_idx < last_idx and last_idx < len(msd_array):
                    fit_times = time_array[first_idx:last_idx + 1]
                    fit_msd = msd_array[first_idx:last_idx + 1]
                    slope, _, _, _, std_err = linregress(fit_times, fit_msd)
                    diffusivity = (slope * 1e-4) / (2 * d)
                    diffusivity_err = (std_err * 1e-4) / (2 * d)
                else:
                    diffusivity = np.nan
                    diffusivity_err = np.nan

                # Block analysis for error estimation
                M = block
                B = step_counts // M
                slope_blocks = []
                for b in range(B):
                    t_start = b * M
                    t_end = (b + 1) * M
                    if t_end > step_counts:
                        break
                    pos_block = posit[:, t_start:t_end, :]
                    step_block = t_end - t_start
                    if diff_type == "Tracer":
                        msd_block = msd_tracer(pos_block, step_block)
                    elif diff_type == "Charged" or diff_type == "Collective":
                        tracer_msd_block = msd_tracer(pos_block, step_block)
                        msd_block = msd_charged(pos_block, tracer_msd_block)
                    dt_block = np.arange(1, step_block) * dt_value
                    if len(dt_block) >= 2:
                        slope_block, _, _, _, _ = linregress(dt_block, msd_block)
                        slope_blocks.append(slope_block)
                if len(slope_blocks) > 1:
                    slope_mean = np.mean(slope_blocks)
                    slope_std = np.std(slope_blocks)
                    slope_sem = slope_std / np.sqrt(len(slope_blocks) - 1)
                else:
                    slope_sem = np.nan

                result_dict[element].setdefault(key_diffusivity, []).append(diffusivity)
                result_dict[element].setdefault(key_diffusivity_error, []).append(diffusivity_err)
                result_dict[element].setdefault(key_slope_sem, []).append(slope_sem)
    return result_dict

def calculate_msd_qe(elements, diffusivity_direction_choices, diffusivity_choices,
                     pos_full, conduct_rectified_structure_array, conduct_ions_array,
                     t, Last_term, initial_slope_time, final_slope_time, block,
                     atom_types=None, cell_param_full=None):
    """
    Calculate MSD and diffusivity for Quantum Espresso-style trajectory data.

    Args:
        elements (list): List of diffusing elements.
        diffusivity_direction_choices (list): Directions for analysis.
        diffusivity_choices (list): Types of diffusivity ('Tracer', 'Charged').
        pos_full, conduct_rectified_structure_array, conduct_ions_array: Trajectory and ion data.
        t (np.ndarray): Time array.
        Last_term (int): Last index for analysis.
        initial_slope_time (float): Start time for slope fitting.
        final_slope_time (float): End time for slope fitting.
        block (int): Block size for error estimation.
        atom_types, cell_param_full: Optional atom/cell info.

    Returns:
        dict: Nested dictionary with MSD, diffusivity, and error estimates.
    """
    result_dict = {}
    for ele in range(len(elements)):
        element = elements[ele]
        result_dict[element] = {}
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            step_counts = posit.shape[1]
            for diff_type_idx, diff_type in enumerate(diffusivity_choices):
                key_msd = f"{diff_type}_msd_array{suffix}"
                key_diff = f"{diff_type}_diffusivity{suffix}"
                key_sem = f"{diff_type}_slope_sem{suffix}"
                key_diff_err = f"{diff_type}_diffusivity_error{suffix}"

                # MSD calculation for each type
                if diff_type == "Tracer":
                    msd_array = msd_tracer(posit, step_counts)
                elif diff_type == "Charged":
                    tracer_msd = msd_tracer(posit, step_counts)
                    msd_array = msd_charged(posit, tracer_msd)
                else:
                    raise ValueError(f"Unknown diffusivity type: {diff_type}")

                # Linear regression for diffusivity
                first_idx = np.searchsorted(t[1:], initial_slope_time, side='left')
                last_idx = np.searchsorted(t[1:], final_slope_time, side='right') - 1
                if first_idx <= last_idx:
                    slope, _, _, _, std_err = linregress(t[1:][first_idx:last_idx + 1], msd_array[first_idx:last_idx + 1])
                    diffusivity = (slope * 1e-4) / (2 * d)
                else:
                    diffusivity = np.nan
                    std_err = np.nan

                # Block analysis for error estimation
                M = block
                B = step_counts // M
                slope_blocks = []
                for b in range(B):
                    t_start = b * M
                    t_end = (b + 1) * M
                    if t_end > step_counts:
                        break
                    pos_block = posit[:, t_start:t_end, :]
                    step_block = t_end - t_start
                    if diff_type == "Tracer":
                        msd_block = msd_tracer(pos_block, step_block)
                    elif diff_type == "Charged":
                        tracer_msd_block = msd_tracer(pos_block, step_block)
                        msd_block = msd_charged(pos_block, tracer_msd_block)
                    dt_block = np.arange(1, step_block) * (t[1] - t[0])
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
                    slope_sem = np.nan

                result_dict[element].setdefault(key_msd, []).append(msd_array)
                result_dict[element].setdefault(key_diff, []).append(diffusivity)
                result_dict[element].setdefault(key_sem, []).append(slope_sem if 'slope_sem' in locals() else np.nan)
                result_dict[element].setdefault(key_diff_err, []).append(diffusivity_sem)
    return result_dict

def calculate_msd(elements, diffusivity_direction_choices, diffusivity_choices,
                  pos_full, conduct_rectified_structure_array, conduct_ions_array,
                  t, Last_term, initial_slope_time, final_slope_time, block,
                  is_lammps=False, dt_value=1.0, lammps_units="metal", atom_types=None, cell_param_full=None):
    """
    Main orchestrator for MSD and diffusivity calculations.

    Args:
        elements (list): List of diffusing elements.
        diffusivity_direction_choices (list): Directions for analysis.
        diffusivity_choices (list): Types of diffusivity.
        pos_full, conduct_rectified_structure_array, conduct_ions_array: Trajectory and ion data.
        t (np.ndarray): Time array.
        Last_term (int): Last index for analysis.
        initial_slope_time (float): Start time for slope fitting.
        final_slope_time (float): End time for slope fitting.
        block (int): Block size for error estimation.
        is_lammps (bool): If True, use LAMMPS-style calculation.
        dt_value (float): Time step value for LAMMPS.
        lammps_units (str): LAMMPS units.
        atom_types, cell_param_full: Optional atom/cell info.

    Returns:
        dict: Nested dictionary with MSD, diffusivity, and error estimates.
    """
    if is_lammps:
        return calculate_msd_lammps(
            elements, diffusivity_direction_choices, diffusivity_choices,
            pos_full, conduct_rectified_structure_array, conduct_ions_array,
            t, Last_term, initial_slope_time, final_slope_time, block,
            dt_value=dt_value, lammps_units=lammps_units, atom_types=atom_types, cell_param_full=cell_param_full
        )
    else:
        return calculate_msd_qe(
            elements, diffusivity_direction_choices, diffusivity_choices,
            pos_full, conduct_rectified_structure_array, conduct_ions_array,
            t, Last_term, initial_slope_time, final_slope_time, block,
            atom_types=atom_types, cell_param_full=cell_param_full
        )

def calc_ngp_tracer(r, d, d_idx):
    """
    Calculate Non-Gaussian Parameter (NGP) for tracer diffusion.

    Args:
        r (np.ndarray): Position array (time_steps, 3) for one particle.
        d (int): Dimensionality (1, 2, or 3).
        d_idx (np.ndarray): Array of time lag indices.

    Returns:
        np.ndarray: NGP values for specified time lags.
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
        diffusing_elements (list): Elements to analyze (e.g., ['Li']).
        diffusivity_direction_choices (list): Directions (e.g., ['XYZ']).
        pos_full (np.ndarray): Full position trajectory.
        conduct_rectified_structure_array (np.ndarray): Unwrapped positions.
        conduct_ions_array (list): Ion indices.
        dt (np.ndarray): Time step array (ps).
        initial_time (float): Start time for analysis (ps).
        final_time (float): End time for analysis (ps).

    Returns:
        dict: Nested dictionary with NGP and time lag data.
    """
    result_dict = {}
    time_step = dt[1] - dt[0]
    time_lags = np.arange(1, len(dt)) * time_step
    mask = (time_lags >= initial_time) & (time_lags <= final_time)
    d_idx = np.arange(1, len(dt))[mask]
    print(f"[DEBUG] d_idx for NGP: {d_idx}")

    for ele in range(len(diffusing_elements)):
        element = diffusing_elements[ele]
        result_dict[element] = {}

        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            step_counts = len(posit[0, :, 0])

            print(f"posit shape for NGP: {posit.shape}")
            print(f"step_counts for NGP: {step_counts}")

            num_mobile_ions = posit.shape[0]
            # Parallel calculation of NGP for all ions
            with Pool() as pool:
                ngp_ions = pool.starmap(calc_ngp_tracer, 
                                       [(posit[i, :, :], d, d_idx) for i in range(num_mobile_ions)])
            ngp_ions = np.array(ngp_ions)
            ngp_array = np.average(ngp_ions, axis=0)

            print(f"NGP array length: {len(ngp_array)}")
            print(f"dt length for NGP: {len(dt)}")

            key_ngp = f"NGP_array{suffix}"
            result_dict[element].setdefault(key_ngp, []).append(ngp_array)
            result_dict[element].setdefault('time_lags' + suffix, []).append(time_lags[mask])

    return result_dict

# """
# Calculations Module for CPDyAna
# ===============================

# This module provides the core calculation functions for molecular dynamics analysis,
# focusing on Mean Square Displacement (MSD) calculations, diffusion coefficient
# determination, and Non-Gaussian Parameter (NGP) analysis for both tracer and collective diffusion.

# The module implements efficient algorithms for:
# - MSD calculation for individual and collective particle motion
# - Linear regression analysis for diffusion coefficient extraction
# - Multi-directional diffusivity analysis (X, Y, Z, XY, XZ, YZ, XYZ)
# - Statistical block averaging for error estimation
# - Non-Gaussian Parameter (NGP) calculation for dynamical heterogeneity

# Functions:
#     calculate_msd: Main MSD calculation orchestrator (LAMMPS and QE support)
#     msd_tracer: Individual particle MSD calculation
#     msd_charged: Collective/charged species MSD calculation
#     calculate_ngp: Non-Gaussian Parameter calculation for tracer diffusion
#     msd_tracer_lazy: Lazy loading MSD calculation
#     calc_ngp_tracer_lazy: Lazy loading NGP calculation

# Author: CPDyAna Development Team
# Version: 06-25-2025
# """

# import numpy as np
# from scipy.stats import linregress
# from multiprocessing import Pool

# # FFT-based MSD calculation
# def calc_fft(x):
#     """
#     Efficiently compute the autocorrelation using FFT for MSD calculation.

#     Args:
#         x (np.ndarray): 1D array of positions for a single coordinate.

#     Returns:
#         np.ndarray: Autocorrelation result for MSD computation.
#     """
#     N = x.shape[0]
#     F = np.fft.fft(x, n=2 * N)
#     PSD = F * F.conjugate()
#     res = np.fft.ifft(PSD)
#     res = (res[:N]).real
#     n = N * np.ones(N) - np.arange(N)
#     return res / n

# # Tracer MSD calculation helper
# def calc_msd_tracer(r, d_idx):
#     """
#     Calculate MSD for a single particle trajectory using an efficient algorithm.

#     Args:
#         r (np.ndarray): Trajectory array (steps, 3) for one particle.
#         d_idx (np.ndarray): Array of time lag indices.

#     Returns:
#         tuple: (MSD array for selected lags, MSD array per dimension for selected lags)
#     """
#     step, dim = r.shape
#     r_sq = np.square(r)
#     r_sq = np.append(r_sq, np.zeros((1, 3)), axis=0)
#     part1_c = np.zeros((3, step))
#     r_sq_sum = 2 * np.sum(r_sq, axis=0)
#     for i in range(step):
#         r_sq_sum -= r_sq[i - 1, :] + r_sq[step - i, :]
#         part1_c[:, i] = r_sq_sum / (step - i)
#     part1 = np.sum(part1_c, axis=0)
#     part2_c = np.array([calc_fft(r[:, i]) for i in range(r.shape[1])])
#     part2 = np.sum(part2_c, axis=0)
#     return (part1 - 2 * part2)[d_idx], (part1_c - 2 * part2_c)[:, d_idx]

# # def msd_tracer(structure, step):
# #     """
# #     Calculate average tracer MSD for all mobile ions.

# #     Args:
# #         structure (np.ndarray): Array of shape (num_mobile_ions, steps, 3).
# #         step (int): Number of time steps.

# #     Returns:
# #         np.ndarray: Average tracer MSD over all ions.
# #     """
# #     num_mobile_ions = structure.shape[0]
# #     msd_ions = np.empty([0, len(np.arange(1, step, 1))])
# #     for i in range(num_mobile_ions):
# #         msd_i, _ = calc_msd_tracer(structure[i, :, :], np.arange(1, step, 1))
# #         msd_ions = np.append(msd_ions, msd_i.reshape(1, len(np.arange(1, step, 1))), axis=0)
# #     return np.average(msd_ions, axis=0)

# # def msd_charged(structure, final_msd):
# #     """
# #     Calculate the charged MSD for a given 3D position array and tracer MSD.

# #     Args:
# #         structure (np.ndarray): Array of shape (num_mobile_ions, steps, 3).
# #         final_msd (np.ndarray): Tracer MSD array.

# #     Returns:
# #         np.ndarray: Charged MSD values.
# #     """
# #     num_mobile_ions = structure.shape[0]
# #     steps = structure.shape[1]
# #     MSD_charged = np.zeros(steps - 1)
# #     for delt in range(1, steps):
# #         MSD_charged[delt-1] = (disp_sum(delt, structure) -
# #                                final_msd[delt-1] * num_mobile_ions * (steps - delt)) / \
# #                               (0.5 * num_mobile_ions * (num_mobile_ions - 1) * (steps - delt))
# #     return MSD_charged

# def disp_sum(delt, mobile_ion_pos):
#     """
#     Calculate the sum of squared displacements for a given time delta.

#     Args:
#         delt (int): Time step difference.
#         mobile_ion_pos (np.ndarray): Array of shape (num_mobile_ions, steps, 3).

#     Returns:
#         float: Sum of squared displacements.
#     """
#     t = np.arange(mobile_ion_pos.shape[1] - delt)
#     displacements = mobile_ion_pos[:, t + delt, :] - mobile_ion_pos[:, t, :]
#     return np.sum(np.square(np.sum(np.abs(displacements), axis=0)))

# # --- LAZY LOADING VERSIONS ---

# def msd_tracer(frame_generator, n_atoms, max_lag=None):
#     """
#     Calculate average tracer MSD for all mobile ions using a frame generator.
#     Args:
#         frame_generator: yields frames with 'positions' (n_atoms, 3)
#         n_atoms: number of mobile ions
#         max_lag: maximum lag time to compute MSD for (int)
#     Returns:
#         msd: np.ndarray of MSD values for each lag
#     """
#     # Store positions for each atom as a list of arrays
#     positions = [[] for _ in range(n_atoms)]
#     for frame in frame_generator:
#         pos = frame["positions"]
#         for i in range(n_atoms):
#             positions[i].append(pos[i])
#     # Convert to arrays
#     positions = [np.array(p) for p in positions]
#     n_steps = len(positions[0])
#     if max_lag is None:
#         max_lag = n_steps - 1
#     msd = np.zeros(max_lag)
#     for lag in range(1, max_lag + 1):
#         msd_lag = []
#         for atom_pos in positions:
#             disp = atom_pos[lag:] - atom_pos[:-lag]
#             msd_lag.append(np.mean(np.sum(disp**2, axis=1)))
#         msd[lag - 1] = np.mean(msd_lag)
#     return msd

# def msd_charged(frame_generator, n_atoms, max_lag=None):
#     """
#     Calculate the charged MSD for a given frame generator (lazy loading).
#     Args:
#         frame_generator: yields frames with 'positions' (n_atoms, 3)
#         n_atoms: number of mobile ions
#         max_lag: maximum lag time to compute MSD for (int)
#     Returns:
#         msd_charged: np.ndarray of charged MSD values for each lag
#     """
#     # Collect all positions as in the tracer version
#     positions = [[] for _ in range(n_atoms)]
#     for frame in frame_generator:
#         pos = frame["positions"]
#         for i in range(n_atoms):
#             positions[i].append(pos[i])
#     positions = [np.array(p) for p in positions]
#     n_steps = len(positions[0])
#     if max_lag is None:
#         max_lag = n_steps - 1

#     msd_tracer = np.zeros(max_lag)
#     msd_charged = np.zeros(max_lag)
#     # Compute tracer MSD first
#     for lag in range(1, max_lag + 1):
#         msd_lag = []
#         for atom_pos in positions:
#             disp = atom_pos[lag:] - atom_pos[:-lag]
#             msd_lag.append(np.mean(np.sum(disp**2, axis=1)))
#         msd_tracer[lag - 1] = np.mean(msd_lag)

#     # Now compute charged MSD
#     for lag in range(1, max_lag + 1):
#         t = np.arange(n_steps - lag)
#         # Center of mass at each time
#         com_t = np.zeros((n_steps - lag, 3))
#         com_t_lag = np.zeros((n_steps - lag, 3))
#         for i in range(n_atoms):
#             com_t += positions[i][t]
#             com_t_lag += positions[i][t + lag]
#         com_t /= n_atoms
#         com_t_lag /= n_atoms
#         disp_com = com_t_lag - com_t
#         msd_com = np.mean(np.sum(disp_com**2, axis=1))
#         # Charged MSD formula
#         msd_charged[lag - 1] = (n_atoms * msd_com - msd_tracer[lag - 1]) / (n_atoms - 1) if n_atoms > 1 else 0.0

#     return msd_charged

# def calc_ngp_tracer(frame_generator, n_atoms, max_lag=None):
#     """
#     Calculate Non-Gaussian Parameter (NGP) for tracer diffusion using a frame generator.
#     Args:
#         frame_generator: yields frames with 'positions' (n_atoms, 3)
#         n_atoms: number of mobile ions
#         max_lag: maximum lag time to compute NGP for (int)
#     Returns:
#         ngp: np.ndarray of NGP values for each lag
#     """
#     positions = [[] for _ in range(n_atoms)]
#     for frame in frame_generator:
#         pos = frame["positions"]
#         for i in range(n_atoms):
#             positions[i].append(pos[i])
#     positions = [np.array(p) for p in positions]
#     n_steps = len(positions[0])
#     if max_lag is None:
#         max_lag = n_steps - 1
#     ngp = np.zeros(max_lag)
#     d = 3  # dimensionality
#     for lag in range(1, max_lag + 1):
#         r2_all = []
#         r4_all = []
#         for atom_pos in positions:
#             disp = atom_pos[lag:] - atom_pos[:-lag]
#             r2 = np.sum(disp**2, axis=1)
#             r4 = r2**2
#             r2_all.append(r2)
#             r4_all.append(r4)
#         mean_r2 = np.mean(np.concatenate(r2_all))
#         mean_r4 = np.mean(np.concatenate(r4_all))
#         if mean_r2 > 0:
#             numerator = d * mean_r4
#             denominator = (d + 2) * (mean_r2**2)
#             ngp[lag - 1] = (numerator / denominator) - 1
#         else:
#             ngp[lag - 1] = 0.0
#     return ngp

# # --- END LAZY LOADING VERSIONS ---

# def calculate_msd_lammps(elements, diffusivity_direction_choices, diffusivity_choices,
#                          pos_full, conduct_rectified_structure_array, conduct_ions_array,
#                          t, Last_term, initial_slope_time, final_slope_time, block,
#                          dt_value=1.0, lammps_units="metal", atom_types=None, cell_param_full=None):
#     """
#     Calculate MSD and diffusivity for LAMMPS-style trajectory data.

#     Args:
#         elements (list): List of diffusing elements.
#         diffusivity_direction_choices (list): Directions for analysis.
#         diffusivity_choices (list): Types of diffusivity ('Tracer', 'Charged', etc.).
#         pos_full, conduct_rectified_structure_array, conduct_ions_array: Trajectory and ion data.
#         t (np.ndarray): Time array.
#         Last_term (int): Last index for analysis.
#         initial_slope_time (float): Start time for slope fitting.
#         final_slope_time (float): End time for slope fitting.
#         block (int): Block size for error estimation.
#         dt_value (float): Time step value.
#         lammps_units (str): LAMMPS units.
#         atom_types, cell_param_full: Optional atom/cell info.

#     Returns:
#         dict: Nested dictionary with MSD, diffusivity, and error estimates.
#     """
#     result_dict = {}
#     n_frames = conduct_rectified_structure_array.shape[2] if len(conduct_rectified_structure_array.shape) == 4 else len(t)
#     dt_array = np.arange(1, n_frames) * dt_value

#     for ele_idx, element in enumerate(elements):
#         result_dict[element] = {}
#         for direction_idx, direction in enumerate(diffusivity_direction_choices):
#             d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
#             suffix = '' if direction == 'XYZ' else f'_{direction}'
#             if len(conduct_rectified_structure_array.shape) == 4:
#                 posit = conduct_rectified_structure_array[direction_idx, :, :, :]
#             else:
#                 posit = conduct_rectified_structure_array
#             step_counts = posit.shape[1]
#             for diff_type_idx, diff_type in enumerate(diffusivity_choices):
#                 key_msd = f"{diff_type}_msd_array{suffix}"
#                 key_time = f"{diff_type}_time_array{suffix}"
#                 key_diffusivity = f"{diff_type}_diffusivity{suffix}"
#                 key_diffusivity_error = f"{diff_type}_diffusivity_error{suffix}"
#                 key_slope_sem = f"{diff_type}_slope_sem{suffix}"

#                 # MSD calculation for each type
#                 if diff_type == "Tracer":
#                     msd_array = msd_tracer(posit, step_counts)
#                 elif diff_type == "Charged" or diff_type == "Collective":
#                     tracer_msd_array = msd_tracer(posit, step_counts)
#                     msd_array = msd_charged(posit, tracer_msd_array)
#                 else:
#                     continue

#                 time_array = dt_array[:len(msd_array)]
#                 result_dict[element].setdefault(key_msd, []).append(msd_array)
#                 result_dict[element].setdefault(key_time, []).append(time_array)

#                 # Linear regression for diffusivity
#                 first_idx = np.searchsorted(time_array, initial_slope_time, side='left')
#                 last_idx = np.searchsorted(time_array, final_slope_time, side='right') - 1
#                 if first_idx < last_idx and last_idx < len(msd_array):
#                     fit_times = time_array[first_idx:last_idx + 1]
#                     fit_msd = msd_array[first_idx:last_idx + 1]
#                     slope, _, _, _, std_err = linregress(fit_times, fit_msd)
#                     diffusivity = (slope * 1e-4) / (2 * d)
#                     diffusivity_err = (std_err * 1e-4) / (2 * d)
#                 else:
#                     diffusivity = np.nan
#                     diffusivity_err = np.nan

#                 # Block analysis for error estimation
#                 M = block
#                 B = step_counts // M
#                 slope_blocks = []
#                 for b in range(B):
#                     t_start = b * M
#                     t_end = (b + 1) * M
#                     if t_end > step_counts:
#                         break
#                     pos_block = posit[:, t_start:t_end, :]
#                     step_block = t_end - t_start
#                     if diff_type == "Tracer":
#                         msd_block = msd_tracer(pos_block, step_block)
#                     elif diff_type == "Charged" or diff_type == "Collective":
#                         tracer_msd_block = msd_tracer(pos_block, step_block)
#                         msd_block = msd_charged(pos_block, tracer_msd_block)
#                     dt_block = np.arange(1, step_block) * dt_value
#                     if len(dt_block) >= 2:
#                         slope_block, _, _, _, _ = linregress(dt_block, msd_block)
#                         slope_blocks.append(slope_block)
#                 if len(slope_blocks) > 1:
#                     slope_mean = np.mean(slope_blocks)
#                     slope_std = np.std(slope_blocks)
#                     slope_sem = slope_std / np.sqrt(len(slope_blocks) - 1)
#                 else:
#                     slope_sem = np.nan

#                 result_dict[element].setdefault(key_diffusivity, []).append(diffusivity)
#                 result_dict[element].setdefault(key_diffusivity_error, []).append(diffusivity_err)
#                 result_dict[element].setdefault(key_slope_sem, []).append(slope_sem)
#     return result_dict

# def calculate_msd_qe(elements, diffusivity_direction_choices, diffusivity_choices,
#                      pos_full, conduct_rectified_structure_array, conduct_ions_array,
#                      t, Last_term, initial_slope_time, final_slope_time, block,
#                      atom_types=None, cell_param_full=None):
#     """
#     Calculate MSD and diffusivity for Quantum Espresso-style trajectory data.

#     Args:
#         elements (list): List of diffusing elements.
#         diffusivity_direction_choices (list): Directions for analysis.
#         diffusivity_choices (list): Types of diffusivity ('Tracer', 'Charged').
#         pos_full, conduct_rectified_structure_array, conduct_ions_array: Trajectory and ion data.
#         t (np.ndarray): Time array.
#         Last_term (int): Last index for analysis.
#         initial_slope_time (float): Start time for slope fitting.
#         final_slope_time (float): End time for slope fitting.
#         block (int): Block size for error estimation.
#         atom_types, cell_param_full: Optional atom/cell info.

#     Returns:
#         dict: Nested dictionary with MSD, diffusivity, and error estimates.
#     """
#     result_dict = {}
#     for ele in range(len(elements)):
#         element = elements[ele]
#         result_dict[element] = {}
#         for direction_idx, direction in enumerate(diffusivity_direction_choices):
#             d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
#             suffix = '' if direction == 'XYZ' else f'_{direction}'
#             posit = conduct_rectified_structure_array[direction_idx, :, :, :]
#             step_counts = posit.shape[1]
#             for diff_type_idx, diff_type in enumerate(diffusivity_choices):
#                 key_msd = f"{diff_type}_msd_array{suffix}"
#                 key_diff = f"{diff_type}_diffusivity{suffix}"
#                 key_sem = f"{diff_type}_slope_sem{suffix}"
#                 key_diff_err = f"{diff_type}_diffusivity_error{suffix}"

#                 # MSD calculation for each type
#                 if diff_type == "Tracer":
#                     msd_array = msd_tracer(posit, step_counts)
#                 elif diff_type == "Charged":
#                     tracer_msd = msd_tracer(posit, step_counts)
#                     msd_array = msd_charged(posit, tracer_msd)
#                 else:
#                     raise ValueError(f"Unknown diffusivity type: {diff_type}")

#                 # Linear regression for diffusivity
#                 first_idx = np.searchsorted(t[1:], initial_slope_time, side='left')
#                 last_idx = np.searchsorted(t[1:], final_slope_time, side='right') - 1
#                 if first_idx <= last_idx:
#                     slope, _, _, _, std_err = linregress(t[1:][first_idx:last_idx + 1], msd_array[first_idx:last_idx + 1])
#                     diffusivity = (slope * 1e-4) / (2 * d)
#                 else:
#                     diffusivity = np.nan
#                     std_err = np.nan

#                 # Block analysis for error estimation
#                 M = block
#                 B = step_counts // M
#                 slope_blocks = []
#                 for b in range(B):
#                     t_start = b * M
#                     t_end = (b + 1) * M
#                     if t_end > step_counts:
#                         break
#                     pos_block = posit[:, t_start:t_end, :]
#                     step_block = t_end - t_start
#                     if diff_type == "Tracer":
#                         msd_block = msd_tracer(pos_block, step_block)
#                     elif diff_type == "Charged":
#                         tracer_msd_block = msd_tracer(pos_block, step_block)
#                         msd_block = msd_charged(pos_block, tracer_msd_block)
#                     dt_block = np.arange(1, step_block) * (t[1] - t[0])
#                     if len(dt_block) >= 2:
#                         slope_block, _, _, _, _ = linregress(dt_block, msd_block)
#                         slope_blocks.append(slope_block)
#                 if len(slope_blocks) > 1:
#                     slope_mean = np.mean(slope_blocks)
#                     slope_std = np.std(slope_blocks)
#                     slope_sem = slope_std / np.sqrt(len(slope_blocks) - 1)
#                     diffusivity_block_mean = (slope_mean * 1e-4) / (2 * d)
#                     diffusivity_sem = (slope_sem * 1e-4) / (2 * d)
#                 else:
#                     diffusivity_block_mean = np.nan
#                     diffusivity_sem = np.nan
#                     slope_sem = np.nan

#                 result_dict[element].setdefault(key_msd, []).append(msd_array)
#                 result_dict[element].setdefault(key_diff, []).append(diffusivity)
#                 result_dict[element].setdefault(key_sem, []).append(slope_sem if 'slope_sem' in locals() else np.nan)
#                 result_dict[element].setdefault(key_diff_err, []).append(diffusivity_sem)
#     return result_dict

# def calculate_msd(elements, diffusivity_direction_choices, diffusivity_choices,
#                   pos_full, conduct_rectified_structure_array, conduct_ions_array,
#                   t, Last_term, initial_slope_time, final_slope_time, block,
#                   is_lammps=False, dt_value=1.0, lammps_units="metal", atom_types=None, cell_param_full=None):
#     """
#     Main orchestrator for MSD and diffusivity calculations.

#     Args:
#         elements (list): List of diffusing elements.
#         diffusivity_direction_choices (list): Directions for analysis.
#         diffusivity_choices (list): Types of diffusivity.
#         pos_full, conduct_rectified_structure_array, conduct_ions_array: Trajectory and ion data.
#         t (np.ndarray): Time array.
#         Last_term (int): Last index for analysis.
#         initial_slope_time (float): Start time for slope fitting.
#         final_slope_time (float): End time for slope fitting.
#         block (int): Block size for error estimation.
#         is_lammps (bool): If True, use LAMMPS-style calculation.
#         dt_value (float): Time step value for LAMMPS.
#         lammps_units (str): LAMMPS units.
#         atom_types, cell_param_full: Optional atom/cell info.

#     Returns:
#         dict: Nested dictionary with MSD, diffusivity, and error estimates.
#     """
#     if is_lammps:
#         return calculate_msd_lammps(
#             elements, diffusivity_direction_choices, diffusivity_choices,
#             pos_full, conduct_rectified_structure_array, conduct_ions_array,
#             t, Last_term, initial_slope_time, final_slope_time, block,
#             dt_value=dt_value, lammps_units=lammps_units, atom_types=atom_types, cell_param_full=cell_param_full
#         )
#     else:
#         return calculate_msd_qe(
#             elements, diffusivity_direction_choices, diffusivity_choices,
#             pos_full, conduct_rectified_structure_array, conduct_ions_array,
#             t, Last_term, initial_slope_time, final_slope_time, block,
#             atom_types=atom_types, cell_param_full=cell_param_full
#         )

# # def calc_ngp_tracer(r, d, d_idx):
# #     """
# #     Calculate Non-Gaussian Parameter (NGP) for tracer diffusion.

# #     Args:
# #         r (np.ndarray): Position array (time_steps, 3) for one particle.
# #         d (int): Dimensionality (1, 2, or 3).
# #         d_idx (np.ndarray): Array of time lag indices.

# #     Returns:
# #         np.ndarray: NGP values for specified time lags.
# #     """
# #     step = r.shape[0]
# #     ngp = np.zeros(len(d_idx))
# #     for i, delta in enumerate(d_idx):
# #         disp = r[delta:, :] - r[:-delta, :]
# #         r2 = np.sum(disp**2, axis=1)
# #         r4 = r2**2
# #         mean_r2 = np.mean(r2)
# #         mean_r4 = np.mean(r4)
# #         if mean_r2 > 0:
# #             numerator = d * mean_r4
# #             denominator = (d + 2) * (mean_r2**2)
# #             ngp[i] = (numerator / denominator) - 1
# #         else:
# #             ngp[i] = 0.0
# #     return ngp

# def calculate_ngp(diffusing_elements, diffusivity_direction_choices, pos_full, 
#                   conduct_rectified_structure_array, conduct_ions_array, dt, 
#                   initial_time=2.0, final_time=200.0):
#     """
#     Calculate Non-Gaussian Parameter (NGP) for specified elements and directions.

#     Args:
#         diffusing_elements (list): Elements to analyze (e.g., ['Li']).
#         diffusivity_direction_choices (list): Directions (e.g., ['XYZ']).
#         pos_full (np.ndarray): Full position trajectory.
#         conduct_rectified_structure_array (np.ndarray): Unwrapped positions.
#         conduct_ions_array (list): Ion indices.
#         dt (np.ndarray): Time step array (ps).
#         initial_time (float): Start time for analysis (ps).
#         final_time (float): End time for analysis (ps).

#     Returns:
#         dict: Nested dictionary with NGP and time lag data.
#     """
#     result_dict = {}
#     time_step = dt[1] - dt[0]
#     time_lags = np.arange(1, len(dt)) * time_step
#     mask = (time_lags >= initial_time) & (time_lags <= final_time)
#     d_idx = np.arange(1, len(dt))[mask]
#     print(f"[DEBUG] d_idx for NGP: {d_idx}")

#     for ele in range(len(diffusing_elements)):
#         element = diffusing_elements[ele]
#         result_dict[element] = {}

#         for direction_idx, direction in enumerate(diffusivity_direction_choices):
#             d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
#             suffix = '' if direction == 'XYZ' else f'_{direction}'
#             posit = conduct_rectified_structure_array[direction_idx, :, :, :]
#             step_counts = len(posit[0, :, 0])

#             print(f"posit shape for NGP: {posit.shape}")
#             print(f"step_counts for NGP: {step_counts}")

#             num_mobile_ions = posit.shape[0]
#             # Parallel calculation of NGP for all ions
#             with Pool() as pool:
#                 ngp_ions = pool.starmap(calc_ngp_tracer, 
#                                        [(posit[i, :, :], d, d_idx) for i in range(num_mobile_ions)])
#             ngp_ions = np.array(ngp_ions)
#             ngp_array = np.average(ngp_ions, axis=0)

#             print(f"NGP array length: {len(ngp_array)}")
#             print(f"dt length for NGP: {len(dt)}")

#             key_ngp = f"NGP_array{suffix}"
#             result_dict[element].setdefault(key_ngp, []).append(ngp_array)
#             result_dict[element].setdefault('time_lags' + suffix, []).append(time_lags[mask])

#     return result_dict