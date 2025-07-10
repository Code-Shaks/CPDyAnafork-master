"""
General Data Processing Module for CPDyAna
==========================================

This module provides general-purpose data processing functions for molecular dynamics
analysis, including time window selection, trajectory segmentation, and data evaluation
for diffusion analysis. It is compatible with Quantum ESPRESSO and other DFT/MD codes.

Functions:
    find_terms: Find array indices corresponding to a time window.
    segmenter_func: Segment simulation data arrays for a specified time window.
    data_evaluator: Process position data for diffusion analysis, including drift correction.

Author: CPDyAna Development Team
Version: 06-25-2025
"""

from scipy.stats import linregress
from scipy.stats import norm
import numpy as np

def find_terms(array, first_value, last_value):
    """
    Find indices in an array corresponding to a specified value range.

    Parameters:
    - array (array-like): Array of time or frame values.
    - first_value (float): Start value for analysis window.
    - last_value (float): End value for analysis window.

    Returns:
    - tuple: (first_term, last_term) indices
    """
    arr = np.asarray(array)
    # Boolean masks → indices
    first_candidates = np.where(arr >= first_value)[0]
    last_candidates  = np.where(arr <= last_value)[0]

    if first_candidates.size == 0 or last_candidates.size == 0:
        raise ValueError("Requested range not present in array.")

    first_term = int(first_candidates.min())
    last_term  = int(last_candidates.max())
    return first_term, last_term

def segmenter_func(first_term, last, pos_full, dt_full, time_full, cell_full, ke_elec_full, cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full):
    """
    Segment the simulation data based on specified parameters and return relevant data arrays.

    Parameters:
    - first_term (int): Index of the starting time step.
    - last (int): Index of the ending time step.
    - pos_full, dt_full, time_full, cell_full, ke_elec_full, cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full (numpy.ndarray): Full arrays of simulation data.

    Returns:
    - tuple: Segmented data arrays based on the specified time range.
    """
    # Adjust last_term if last is within 70% of trajectory length (legacy logic)
    if last <= int(0.7 * len(pos_full[0, :, 0])):
        last_term = int(last / 0.7)
    else:
        last_term = int(last)

    # Extract data for the specified range
    pos = pos_full[:, first_term: last_term, :]
    step_counts = len(pos[0, :, 0])
    dt = dt_full[first_term: last_term-1]
    time = time_full[first_term: last_term]
    cell = cell_full[first_term: last_term, :]
    ke_elec = ke_elec_full[first_term: last_term]
    cell_temp = cell_temp_full[first_term: last_term]
    ion_temp = ion_temp_full[first_term: last_term]
    tot_energy = tot_energy_full[first_term: last_term]
    enthalpy = enthalpy_full[first_term: last_term]
    tot_energy_ke_ion = tot_energy_ke_ion_full[first_term: last_term]
    tot_energy_ke_ion_ke_elec = tot_energy_ke_ion_ke_elec_full[first_term: last_term]
    vol = vol_full[first_term: last_term]
    pressure = pressure_full[first_term: last_term]

    return pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure

def data_evaluator(diffusivity_direction_choices, target_elements, pos, total_ion_array, steps):
    """
    Evaluate and process simulation data for ion diffusion analysis.

    Parameters:
    - diffusivity_direction_choices (list): List of diffusion directions to analyze.
    - target_elements (list): List of elements considered as conductors.
    - pos, total_ion_array (numpy.ndarray): Position and element arrays from the simulation.
    - steps (int): Number of simulation steps.

    Returns:
    - tuple: Processed arrays related to ion diffusion analysis.
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

    for direction in diffusivity_direction_choices:
        position_data = np.zeros((len(total_ion_array), steps, 3))

        # Select position data based on direction
        if direction == "XYZ":
            position_data = pos
        elif direction == "XY":
            position_data[:, :, 0:2] = pos[:, :, 0:2]
        elif direction == "YZ":
            position_data[:, :, 1:3] = pos[:, :, 1:3]
        elif direction == "ZX":
            position_data[:, :, [0, 2]] = pos[:, :, [0, 2]]
        elif direction == "X":
            position_data[:, :, 0] = pos[:, :, 0]
        elif direction == "Y":
            position_data[:, :, 1] = pos[:, :, 1]
        elif direction == "Z":
            position_data[:, :, 2] = pos[:, :, 2]

        # Calculate displacement from initial position
        disp = position_data - position_data[:, 0:1, :]

        # Identify conductor and framework indices
        conductor_indices = [i for i, element in enumerate(total_ion_array) if element in target_elements]
        framework_indices = [i for i in range(len(total_ion_array)) if i not in conductor_indices]

        # Extract framework and mobile positions/displacements
        framework_disp = disp[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
        framework_pos = position_data[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
        mobile_pos = position_data[conductor_indices] if len(conductor_indices) > 0 else np.zeros((0, steps, 3))

        # Calculate drift correction using framework atoms
        drift = np.average(framework_disp, axis=0) if len(framework_indices) > 0 else np.zeros((steps, 3))
        corrected_displacements = disp - drift

        # Drift-corrected positions
        drifted_rectified_structure = position_data[:, 0:1, :] + corrected_displacements

        # Extract drift-corrected positions for mobile and framework ions
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

# def data_evaluator(diffusivity_direction_choices, target_elements, frame_generator, total_ion_array):
#     """
#     Evaluate and process simulation data for ion diffusion analysis using a frame generator.

#     Parameters:
#     - diffusivity_direction_choices (list): List of diffusion directions to analyze.
#     - target_elements (list): List of elements considered as conductors.
#     - frame_generator: Generator yielding frames with 'positions' (n_atoms, 3).
#     - total_ion_array (list): List of element symbols for all atoms.

#     Returns:
#     - tuple: Processed arrays related to ion diffusion analysis.
#     """
#     # Accumulate positions frame by frame
#     positions = []
#     for frame in frame_generator:
#         positions.append(frame["positions"])
#     pos = np.stack(positions, axis=1)  # shape: (n_atoms, n_frames, 3)
#     steps = pos.shape[1]

#     # Now use the same logic as the original data_evaluator
#     position_data_list = []
#     drifted_rectified_structure_list = []
#     conductor_indices_list = []
#     framework_indices_list = []
#     framework_pos_list = []
#     mobile_pos_list = []
#     pos_list = []
#     mobile_drifted_rectified_structure_list = []
#     framework_drifted_rectified_structure_list = []

#     for direction in diffusivity_direction_choices:
#         position_data = np.zeros((len(total_ion_array), steps, 3))

#         # Select position data based on direction
#         if direction == "XYZ":
#             position_data = pos
#         elif direction == "XY":
#             position_data[:, :, 0:2] = pos[:, :, 0:2]
#         elif direction == "YZ":
#             position_data[:, :, 1:3] = pos[:, :, 1:3]
#         elif direction == "ZX":
#             position_data[:, :, [0, 2]] = pos[:, :, [0, 2]]
#         elif direction == "X":
#             position_data[:, :, 0] = pos[:, :, 0]
#         elif direction == "Y":
#             position_data[:, :, 1] = pos[:, :, 1]
#         elif direction == "Z":
#             position_data[:, :, 2] = pos[:, :, 2]

#         disp = position_data - position_data[:, 0:1, :]

#         conductor_indices = [i for i, element in enumerate(total_ion_array) if element in target_elements]
#         framework_indices = [i for i in range(len(total_ion_array)) if i not in conductor_indices]

#         framework_disp = disp[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
#         framework_pos = position_data[framework_indices] if len(framework_indices) > 0 else np.zeros((0, steps, 3))
#         mobile_pos = position_data[conductor_indices] if len(conductor_indices) > 0 else np.zeros((0, steps, 3))

#         drift = np.average(framework_disp, axis=0) if len(framework_indices) > 0 else np.zeros((steps, 3))
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