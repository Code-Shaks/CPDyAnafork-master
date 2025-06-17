"""
Data processing utilities for CPDyAna.

This module provides functions for:
- Finding array indices within a time range
- Segmenting trajectory and property data
- Evaluating and correcting position data for diffusivity analysis
"""

from scipy.stats import linregress 
from scipy.stats import norm 
import numpy as np

# Function to find indices in an array corresponding to a given time range
def find_terms(array, first_value, last_value):
    """
    Find the first and last indices in 'array' where the values are within [first_value, last_value].

    Parameters:
        array (array-like): Array of time or step values.
        first_value (float): Lower bound of the range.
        last_value (float): Upper bound of the range.

    Returns:
        first_term (int or None): Index of first value >= first_value.
        last_term (int or None): Index of last value <= last_value.
    """
    first_term = next((i for i, x in enumerate(array) if x >= first_value), None)
    last_term = max((i for i, x in enumerate(array) if x <= last_value), default=None)
    return first_term, last_term

# Function to segment trajectory and property data arrays
def segmenter_func(first_term, last, pos_full, dt_full, time_full, cell_full, ke_elec_full, 
                   cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, 
                   tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full):
    """
    Extracts a segment of the trajectory and associated properties between first_term and last_term.

    Parameters:
        first_term (int): Starting index for the segment.
        last (int): Ending index for the segment.
        pos_full, dt_full, ... (arrays): Full arrays of trajectory and property data.

    Returns:
        Tuple of arrays: (pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, 
                          tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure)
    """
    # Adjust last_term based on a 0.7 fraction of the number of steps, if needed
    last_term = int(last / 0.7) if last <= int(0.7 * len(pos_full[0, :, 0])) else int(last)
    pos = pos_full[:, first_term:last_term, :]
    step_counts = len(pos[0, :, 0])
    dt = dt_full[first_term:last_term - 1]
    time = time_full[first_term:last_term]
    cell = cell_full[first_term:last_term, :]
    ke_elec = ke_elec_full[first_term:last_term]
    cell_temp = cell_temp_full[first_term:last_term]
    ion_temp = ion_temp_full[first_term:last_term]
    tot_energy = tot_energy_full[first_term:last_term]
    enthalpy = enthalpy_full[first_term:last_term]
    tot_energy_ke_ion = tot_energy_ke_ion_full[first_term:last_term]
    tot_energy_ke_ion_ke_elec = tot_energy_ke_ion_ke_elec_full[first_term:last_term]
    vol = vol_full[first_term:last_term]
    pressure = pressure_full[first_term:last_term]
    return (pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, 
            tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure)

# Function to evaluate and correct position data for diffusivity analysis
def data_evaluator(diffusivity_direction_choices, target_elements, pos, total_ion_array, steps):
    """
    Processes position data for diffusivity analysis, including drift correction and selection of mobile/framework ions.

    Parameters:
        diffusivity_direction_choices (list of str): Directions to analyze (e.g., ["XYZ", "XY", "Z"]).
        target_elements (list): Elements considered as mobile ions (e.g., ["Li"]).
        pos (ndarray): Position array (n_atoms, n_steps, 3).
        total_ion_array (list): List of element symbols for all atoms.
        steps (int): Number of time steps.

    Returns:
        Tuple of arrays containing:
            - position_data_list: Position data for each direction.
            - drifted_rectified_structure_list: Drift-corrected positions.
            - conductor_indices_list: Indices of mobile ions.
            - framework_indices_list: Indices of framework ions.
            - framework_pos_list: Positions of framework ions.
            - mobile_pos_list: Positions of mobile ions.
            - mobile_drifted_rectified_structure_list: Drift-corrected positions of mobile ions.
            - framework_drifted_rectified_structure_list: Drift-corrected positions of framework ions.
    """
    position_data_list, drifted_rectified_structure_list = [], []
    conductor_indices_list, framework_indices_list = [], []
    framework_pos_list, mobile_pos_list = [], []
    mobile_drifted_rectified_structure_list, framework_drifted_rectified_structure_list = [], []
    
    for direction in diffusivity_direction_choices:
        # Initialize position data for this direction
        position_data = np.zeros((len(total_ion_array), steps, 3))
        # Select the relevant components based on direction
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
        
        # Calculate displacement from initial positions
        disp = position_data - position_data[:, [0], :]
        # Identify mobile (conductor) and framework ions
        conductor_indices = [i for i, element in enumerate(total_ion_array) if element in target_elements]
        framework_indices = [i for i in range(len(total_ion_array)) if i not in conductor_indices]
        
        framework_disp = disp[framework_indices, :, :]
        framework_pos = position_data[framework_indices, :, :]
        mobile_pos = position_data[conductor_indices, :, :]
        
        # Compute average drift of the framework and correct all displacements
        drift = np.average(framework_disp, axis=0)
        corrected_displacements = disp - drift
        drifted_rectified_structure = position_data[:, [0], :] + corrected_displacements
        
        # Extract drift-corrected positions for mobile and framework ions
        mobile_drifted_rectified_structure = drifted_rectified_structure[conductor_indices, :, :]
        framework_drifted_rectified_structure = drifted_rectified_structure[framework_indices, :, :]
        
        # Collect results for this direction
        position_data_list.append(position_data)
        drifted_rectified_structure_list.append(drifted_rectified_structure)
        conductor_indices_list.append(conductor_indices)
        framework_indices_list.append(framework_indices)
        framework_pos_list.append(framework_pos)
        mobile_pos_list.append(mobile_pos)
        mobile_drifted_rectified_structure_list.append(mobile_drifted_rectified_structure)
        framework_drifted_rectified_structure_list.append(framework_drifted_rectified_structure)
    
    return (np.array(position_data_list), np.array(drifted_rectified_structure_list), 
            np.array(conductor_indices_list, dtype=object), np.array(framework_indices_list, dtype=object), 
            np.array(framework_pos_list), np.array(mobile_pos_list), 
            np.array(mobile_drifted_rectified_structure_list), 
            np.array(framework_drifted_rectified_structure_list))