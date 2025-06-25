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
    """
    # Ensure indices are valid and within bounds
    n_frames = pos_full.shape[1]
    first_term = max(0, first_term)
    last_term = min(last, n_frames - 1)
    if last_term <= first_term:
        last_term = n_frames - 1

    pos = pos_full[:, first_term:last_term+1, :]
    step_counts = pos.shape[1]
    dt = dt_full[first_term:last_term] if len(dt_full) >= last_term else dt_full[first_term:]
    time = time_full[first_term:last_term+1]
    cell = cell_full[first_term:last_term+1, :]
    ke_elec = ke_elec_full[first_term:last_term+1]
    cell_temp = cell_temp_full[first_term:last_term+1]
    ion_temp = ion_temp_full[first_term:last_term+1]
    tot_energy = tot_energy_full[first_term:last_term+1]
    enthalpy = enthalpy_full[first_term:last_term+1]
    tot_energy_ke_ion = tot_energy_ke_ion_full[first_term:last_term+1]
    tot_energy_ke_ion_ke_elec = tot_energy_ke_ion_ke_elec_full[first_term:last_term+1]
    vol = vol_full[first_term:last_term+1]
    pressure = pressure_full[first_term:last_term+1]
    return (pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, 
            tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure)

# Function to evaluate and correct position data for diffusivity analysis
def data_evaluator(diffusivity_direction_choices, target_elements, pos, total_ion_array, steps):
    """
    Evaluates and prepares position data for diffusivity analysis.
    
    Parameters:
        diffusivity_direction_choices (list): List of directions for diffusion analysis ('X', 'Y', 'Z', etc.)
        target_elements (list): List of elements to extract
        pos (numpy.ndarray): Position array with shape (n_atoms, n_frames, 3)
        total_ion_array (list): List of atom types/elements
        steps (int): Number of time steps
        
    Returns:
        tuple: Various arrays needed for diffusion analysis
    """
    n_atoms = pos.shape[0]
    n_frames = pos.shape[1]
    n_directions = len(diffusivity_direction_choices)
    
    # Create arrays to hold results
    pos_array = np.zeros((n_directions, n_atoms, n_frames, 3))
    rectified_structure_array = np.zeros((n_directions, n_atoms, n_frames, 3))
    conduct_ions_array = [[] for _ in range(n_directions)]
    frame_ions_array = [[] for _ in range(n_directions)]
    frame_pos_array = np.zeros((n_directions, n_atoms, n_frames, 3))
    conduct_pos_array = np.zeros((n_directions, n_atoms, n_frames, 3))
    conduct_rectified_structure_array = np.zeros((n_directions, n_atoms, n_frames, 3))
    frame_rectified_structure_array = np.zeros((n_directions, n_atoms, n_frames, 3))
    
    # Process each direction
    for i, direction in enumerate(diffusivity_direction_choices):
        # Calculate masks for selected directions
        dim_mask = np.zeros(3, dtype=bool)
        if 'X' in direction:
            dim_mask[0] = True
        if 'Y' in direction:
            dim_mask[1] = True
        if 'Z' in direction:
            dim_mask[2] = True
        
        # Initialize arrays for this direction
        pos_array[i, :, :, :] = pos
        rectified_structure_array[i, :, :, :] = pos
        
        # Create element-specific arrays
        conduct_indices = []
        frame_indices = []
        
        # Select atoms of target elements
        for j, element in enumerate(total_ion_array):
            if element in target_elements:
                conduct_indices.append(j)
            else:
                frame_indices.append(j)
        
        conduct_ions_array[i] = conduct_indices
        frame_ions_array[i] = frame_indices
        
        if conduct_indices:  # Only proceed if we found matching atoms
            # Extract position data for ions of interest
            if frame_indices:  # If there are frame atoms
                frame_pos_array[i, :len(frame_indices), :, :] = pos[frame_indices, :, :]
            
            # Extract position data for conducting ions
            conduct_pos_array[i, :len(conduct_indices), :, :] = pos[conduct_indices, :, :]
            
            # Copy the rectified structures
            conduct_rectified_structure_array[i, :len(conduct_indices), :, :] = pos[conduct_indices, :, :]
            if frame_indices:
                frame_rectified_structure_array[i, :len(frame_indices), :, :] = pos[frame_indices, :, :]
            
            # Apply directional mask if needed
            if not all(dim_mask):
                # Zero out dimensions not in the selected direction
                for dim in range(3):
                    if not dim_mask[dim]:
                        conduct_rectified_structure_array[i, :len(conduct_indices), :, dim] = 0.0
                        if frame_indices:
                            frame_rectified_structure_array[i, :len(frame_indices), :, dim] = 0.0
    
    return (pos_array, rectified_structure_array, conduct_ions_array, frame_ions_array,
            frame_pos_array, conduct_pos_array, conduct_rectified_structure_array,
            frame_rectified_structure_array)