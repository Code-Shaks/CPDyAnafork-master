"""
LAMMPS Data Processing Module for CPDyAna
=========================================

This module provides specialized functions for processing LAMMPS trajectory data,
optimized for the uniform timestep nature of LAMMPS simulations.

The module includes functions for time window selection, trajectory segmentation,
and data evaluation for diffusion analysis, with LAMMPS-specific optimizations.

Author: CPDyAna Development Team
Version: 06-25-2025
"""

from scipy.stats import linregress
from scipy.stats import norm
import numpy as np

def find_terms_lammps(dt_value, n_timesteps, first_value, last_value):
    """
    Find indices in a LAMMPS trajectory corresponding to a time range.

    Parameters:
    - dt_value: Timestep value in picoseconds
    - n_timesteps: Total number of timesteps
    - first_value: Start time for analysis window in picoseconds
    - last_value: End time for analysis window in picoseconds

    Returns:
    - tuple: (first_term, last_term) indices
    """
    # For LAMMPS, time points are uniformly spaced by dt_value
    # Calculate indices directly from time values
    first_term = min(max(0, int(first_value / dt_value)), n_timesteps - 2)
    last_term = min(max(first_term + 1, int(last_value / dt_value)), n_timesteps - 1)
    return first_term, last_term

def segmenter_func_lammps(first_term, last_term, pos_full, dt_value, n_timesteps, cell_full=None):
    """
    Segment LAMMPS trajectory data based on specified time range.

    Parameters:
    - first_term: Index of first timestep
    - last_term: Index of last timestep
    - pos_full: Full position array [atoms, frames, xyz]
    - dt_value: Timestep value in picoseconds
    - n_timesteps: Total number of timesteps
    - cell_full: Cell parameter array (optional)

    Returns:
    - tuple: (pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure)
      Segmented data arrays for analysis.
    """
    # Ensure indices are valid
    first_term = max(0, min(first_term, pos_full.shape[1] - 2))
    last_term = min(last_term, pos_full.shape[1] - 1)

    # Extract positions for the specified range
    pos = pos_full[:, first_term:last_term+1, :]
    step_counts = pos.shape[1]

    # Create uniform dt and time arrays
    dt = np.ones(step_counts - 1) * dt_value
    time = np.arange(step_counts) * dt_value + (first_term * dt_value)

    # Extract cell parameters if provided
    cell = None
    if cell_full is not None:
        cell = cell_full[first_term:last_term+1]

    # Initialize arrays for other properties (not typically available in LAMMPS)
    ke_elec = np.zeros(step_counts)
    cell_temp = np.zeros(step_counts)
    ion_temp = np.zeros(step_counts)
    tot_energy = np.zeros(step_counts)
    enthalpy = np.zeros(step_counts)
    tot_energy_ke_ion = np.zeros(step_counts)
    tot_energy_ke_ion_ke_elec = np.zeros(step_counts)
    vol = np.zeros(step_counts) if cell is None else np.array([np.linalg.det(cell[i].reshape(3, 3)) for i in range(len(cell))])
    pressure = np.zeros(step_counts)

    return pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure

def calculate_time_window_lammps(dt_value, n_timesteps, initial_slope_time, final_slope_time):
    """
    Calculate time window indices for fitting diffusion coefficients in LAMMPS data.

    Parameters:
    - dt_value: Timestep value in picoseconds
    - n_timesteps: Total number of timesteps
    - initial_slope_time: Start time for fitting in picoseconds
    - final_slope_time: End time for fitting in picoseconds

    Returns:
    - tuple: (first_idx, last_idx, time_array)
      Indices and time array for the fitting window.
    """
    # Calculate indices directly from time values
    first_idx = min(max(0, int(initial_slope_time / dt_value)), n_timesteps - 2)
    last_idx = min(max(first_idx + 1, int(final_slope_time / dt_value)), n_timesteps - 1)

    # Create time array for the window
    time_array = np.arange(first_idx, last_idx + 1) * dt_value

    return first_idx, last_idx, time_array

def data_evaluator_lammps(diffusivity_direction_choices, target_elements, pos, total_ion_array, steps):
    """
    Process LAMMPS position data for diffusion analysis.

    This function has the same API as the original data_evaluator, but is optimized
    for LAMMPS trajectory data processing.

    Parameters:
    - diffusivity_direction_choices: List of directions for analysis (e.g., ["XYZ", "X"])
    - target_elements: List of mobile ion elements
    - pos: Position array [atoms, frames, xyz]
    - total_ion_array: Array of element types
    - steps: Number of timesteps

    Returns:
    - tuple: Arrays needed for diffusion analysis
    """
    position_data_list = [] 
    drifted_rectified_structure_list = []
    conductor_indices_list = []
    framework_indices_list = [] 
    framework_pos_list = []
    mobile_pos_list = []
    pos_list = []
    mobile_drifted_rectified_structure_list = []
    framework_drifted_rectified_structure_list = []

    for direction_idx, direction in enumerate(diffusivity_direction_choices):
        # Calculate masks for selected directions
        dim_mask = np.zeros(3, dtype=bool)
        if 'X' in direction:
            dim_mask[0] = True
        if 'Y' in direction:
            dim_mask[1] = True
        if 'Z' in direction:
            dim_mask[2] = True

        # Create position data based on selected directions
        position_data = np.zeros_like(pos)

        # Copy only the required dimensions using the mask
        for dim in range(3):
            if dim_mask[dim]:
                position_data[:, :, dim] = pos[:, :, dim]

        # Calculate displacements from initial positions
        disp = position_data - position_data[:, 0:1, :]

        # Identify conductor and framework atoms
        conductor_indices = np.array([i for i, element in enumerate(total_ion_array) if element in target_elements])
        framework_indices = np.array([i for i in range(len(total_ion_array)) if i not in conductor_indices])

        # Extract framework and mobile positions
        framework_disp = disp[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
        framework_pos = position_data[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
        mobile_pos = position_data[conductor_indices] if len(conductor_indices) > 0 else np.zeros((0, steps, 3))

        # Calculate drift correction
        drift = np.mean(framework_disp, axis=0) if len(framework_indices) > 0 else np.zeros((steps, 3))
        corrected_displacements = disp - drift

        # Calculate drift-corrected positions
        drifted_rectified_structure = position_data[:, 0:1, :] + corrected_displacements

        # Extract mobile and framework drift-corrected positions
        mobile_drifted_rectified_structure = drifted_rectified_structure[conductor_indices] if len(conductor_indices) > 0 else np.zeros((0, steps, 3))
        framework_drifted_rectified_structure = drifted_rectified_structure[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))

        # Append results to lists
        position_data_list.append(position_data)
        drifted_rectified_structure_list.append(drifted_rectified_structure)
        conductor_indices_list.append(conductor_indices)
        framework_indices_list.append(framework_indices)
        framework_pos_list.append(framework_pos)
        mobile_pos_list.append(mobile_pos)
        pos_list.append(pos)
        mobile_drifted_rectified_structure_list.append(mobile_drifted_rectified_structure)
        framework_drifted_rectified_structure_list.append(framework_drifted_rectified_structure)

    return (np.array(position_data_list), np.array(drifted_rectified_structure_list), 
            np.array(conductor_indices_list), np.array(framework_indices_list), 
            np.array(framework_pos_list), np.array(mobile_pos_list), 
            np.array(mobile_drifted_rectified_structure_list), np.array(framework_drifted_rectified_structure_list))

# def data_evaluator_lammps(diffusivity_direction_choices, target_elements, frame_generator, total_ion_array):
#     """
#     Process LAMMPS position data for diffusion analysis using a frame generator.

#     Parameters:
#     - diffusivity_direction_choices: List of directions for analysis (e.g., ["XYZ", "X"])
#     - target_elements: List of mobile ion elements
#     - frame_generator: Generator yielding frames with 'positions' (n_atoms, 3)
#     - total_ion_array: Array of element types

#     Returns:
#     - tuple: Arrays needed for diffusion analysis
#     """
#     positions = []
#     for frame in frame_generator:
#         positions.append(frame["positions"])
#     pos = np.stack(positions, axis=1)  # shape: (n_atoms, n_frames, 3)
#     steps = pos.shape[1]

#     # Use the same logic as data_evaluator_lammps
#     position_data_list = [] 
#     drifted_rectified_structure_list = []
#     conductor_indices_list = []
#     framework_indices_list = [] 
#     framework_pos_list = []
#     mobile_pos_list = []
#     pos_list = []
#     mobile_drifted_rectified_structure_list = []
#     framework_drifted_rectified_structure_list = []

#     for direction_idx, direction in enumerate(diffusivity_direction_choices):
#         dim_mask = np.zeros(3, dtype=bool)
#         if 'X' in direction:
#             dim_mask[0] = True
#         if 'Y' in direction:
#             dim_mask[1] = True
#         if 'Z' in direction:
#             dim_mask[2] = True

#         position_data = np.zeros_like(pos)
#         for dim in range(3):
#             if dim_mask[dim]:
#                 position_data[:, :, dim] = pos[:, :, dim]

#         disp = position_data - position_data[:, 0:1, :]

#         conductor_indices = np.array([i for i, element in enumerate(total_ion_array) if element in target_elements])
#         framework_indices = np.array([i for i in range(len(total_ion_array)) if i not in conductor_indices])

#         framework_disp = disp[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
#         framework_pos = position_data[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
#         mobile_pos = position_data[conductor_indices] if len(conductor_indices) > 0 else np.zeros((0, steps, 3))

#         drift = np.mean(framework_disp, axis=0) if len(framework_indices) > 0 else np.zeros((steps, 3))
#         corrected_displacements = disp - drift

#         drifted_rectified_structure = position_data[:, 0:1, :] + corrected_displacements

#         mobile_drifted_rectified_structure = drifted_rectified_structure[conductor_indices] if len(conductor_indices) > 0 else np.zeros((0, steps, 3))
#         framework_drifted_rectified_structure = drifted_rectified_structure[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))

#         position_data_list.append(position_data)
#         drifted_rectified_structure_list.append(drifted_rectified_structure)
#         conductor_indices_list.append(conductor_indices)
#         framework_indices_list.append(framework_indices)
#         framework_pos_list.append(framework_pos)
#         mobile_pos_list.append(mobile_pos)
#         pos_list.append(pos)
#         mobile_drifted_rectified_structure_list.append(mobile_drifted_rectified_structure)
#         framework_drifted_rectified_structure_list.append(framework_drifted_rectified_structure)

#     return (np.array(position_data_list), np.array(drifted_rectified_structure_list), 
#             np.array(conductor_indices_list), np.array(framework_indices_list), 
#             np.array(framework_pos_list), np.array(mobile_pos_list), 
#             np.array(mobile_drifted_rectified_structure_list), np.array(framework_drifted_rectified_structure_list))

def detect_lammps_format(dt_array):
    """
    Detect if the given dt_array is from a LAMMPS trajectory.

    LAMMPS trajectories have uniform timesteps, while QE trajectories have varying timesteps.

    Parameters:
    - dt_array: Array of timestep values

    Returns:
    - bool: True if LAMMPS format detected, False otherwise
    """
    if len(dt_array) <= 1:
        return False
    # Check if all timesteps are equal (within numerical precision)
    dt_value = dt_array[0]
    return np.allclose(dt_array, dt_value, rtol=1e-5)

def is_lammps_simulation(data_dict):
    """
    Check if the provided data is from a LAMMPS simulation.

    Parameters:
    - data_dict: Dictionary containing simulation data

    Returns:
    - bool: True if LAMMPS data detected, False otherwise
    """
    # Check for common LAMMPS identifiers
    if not data_dict:
        return False
    # Find any key that contains dt_dict
    dt_found = False
    for key in data_dict.keys():
        if 'dt_dict' in data_dict[key]:
            dt_array = data_dict[key]['dt_dict']
            dt_found = True
            # Check if dt values are uniform (LAMMPS characteristic)
            if dt_array and len(dt_array) > 1:
                return np.allclose(dt_array, dt_array[0], rtol=1e-5)
    return False if dt_found else None  # None if can't determine

