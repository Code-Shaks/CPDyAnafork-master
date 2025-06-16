from scipy.stats import linregress # Import the linregress function from scipy.stats for linear regression.
from scipy.stats import norm # Import the norm function from scipy.stats for normal distribution.
import numpy as np # Import numpy for numerical operations.

def find_terms(array, first_value, last_value):
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
    if last <= int(0.7*len(pos_full[0,:,0])): # Check if last index is within 70% of the length of pos_full.
        last_term = int(last/0.7) # Adjust last_term if within 70%.
    else:
        last_term = int(last) # Use the provided last index if beyond 70%.

    pos = pos_full[:, first_term: last_term, :] # Extract positions for the specified range.
    step_counts = len(pos[0,:,0]) # Calculate the number of steps in the range.
    dt = dt_full[first_term: last_term-1] # Extract time step sizes for the range.
    time = time_full[first_term: last_term] # Extract times for the range.
    cell = cell_full[first_term: last_term, :] # Extract cell data for the range.
    ke_elec = ke_elec_full[first_term: last_term] # Extract kinetic energy of electrons for the range.
    cell_temp = cell_temp_full[first_term: last_term] # Extract cell temperature for the range.
    ion_temp = ion_temp_full[first_term: last_term] # Extract ion temperature for the range.
    tot_energy = tot_energy_full[first_term: last_term] # Extract total energy for the range.
    enthalpy = enthalpy_full[first_term: last_term] # Extract enthalpy for the range.
    tot_energy_ke_ion = tot_energy_ke_ion_full[first_term: last_term] # Extract total energy including kinetic energy of ions for the range.
    tot_energy_ke_ion_ke_elec = tot_energy_ke_ion_ke_elec_full[first_term: last_term] # Extract total energy including kinetic energy of ions and electrons for the range.
    vol = vol_full[first_term: last_term] # Extract volume for the range.
    pressure = pressure_full[first_term: last_term] # Extract pressure for the range.

    return pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure
    # Return all the extracted arrays.

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
    position_data_list = [] # Initialize list to store position data.
    drifted_rectified_structure_list = [] # Initialize list to store drift-corrected structures.
    conductor_indices_list = [] # Initialize list to store indices of conductors.
    framework_indices_list = [] # Initialize list to store indices of framework elements.
    framework_pos_list = [] # Initialize list to store positions of framework elements.
    mobile_pos_list = [] # Initialize list to store positions of mobile elements.
    pos_list = [] # Initialize list to store original positions.
    mobile_drifted_rectified_structure_list = [] # Initialize list to store drift-corrected structures of mobile elements.
    framework_drifted_rectified_structure_list = [] # Initialize list to store drift-corrected structures of framework elements.

    for direction in diffusivity_direction_choices: # Iterate over each diffusion direction choice.
        position_data = np.zeros((len(total_ion_array), steps, 3)) # Initialize position data array.

        if direction == "XYZ": # If direction is "XYZ", use the original positions.
            position_data = pos
        elif direction == "XY": # If direction is "XY", extract XY positions.
            for Y in range(len(total_ion_array)):
                for X in range(steps):
                    position_data[Y, X, 0] = pos[Y, X, 0]
                    position_data[Y, X, 1] = pos[Y, X, 1]
        elif direction == "YZ": # If direction is "YZ", extract YZ positions.
            for Y in range(len(total_ion_array)):
                for X in range(steps):
                    position_data[Y, X, 1] = pos[Y, X, 1]
                    position_data[Y, X, 2] = pos[Y, X, 2]
        elif direction == "ZX": # If direction is "ZX", extract ZX positions.
            for Y in range(len(total_ion_array)):
                for X in range(steps):
                    position_data[Y, X, 0] = pos[Y, X, 0]
                    position_data[Y, X, 2] = pos[Y, X, 2]
        elif direction == "X": # If direction is "X", extract X positions.
            for Y in range(len(total_ion_array)):
                for X in range(steps):
                    position_data[Y, X, 0] = pos[Y, X, 0]
        elif direction == "Y": # If direction is "Y", extract Y positions.
            for Y in range(len(total_ion_array)):
                for X in range(steps):
                    position_data[Y, X, 1] = pos[Y, X, 1]
        elif direction == "Z": # If direction is "Z", extract Z positions.
            for Y in range(len(total_ion_array)):
                for X in range(steps):
                    position_data[Y, X, 2] = pos[Y, X, 2]

        disp = np.zeros((len(total_ion_array), steps, 3)) # Initialize displacement array.
        for Y in range(len(total_ion_array)): # Iterate over all ions.
            for X in range(steps):
                for Z in range(3):
                    disp[Y, X, Z] = position_data[Y, X, Z] - position_data[Y, 0, Z] # Calculate displacement from initial position.

        conductor_indices = [i for i, element in enumerate(total_ion_array) if element in target_elements] # Get indices of conductor elements.
        framework_indices = [i for i in range(len(total_ion_array)) if i not in conductor_indices] # Get indices of framework elements.

        framework_disp = np.zeros((len(framework_indices), steps, 3)) # Initialize framework displacement array.
        framework_pos = np.zeros((len(framework_indices), steps, 3)) # Initialize framework position array.
        for Y in range(len(framework_indices)): # Iterate over framework indices.
            for X in range(steps):
                for Z in range(3):
                    index = framework_indices[Y]
                    framework_disp[Y, X, Z] = disp[int(index), X, Z] # Calculate framework displacement.
                    framework_pos[Y, X, Z] = position_data[int(index), X, Z] # Store framework positions.

        mobile_pos = np.zeros((len(conductor_indices), steps, 3)) # Initialize mobile positions array.
        for Y in range(len(conductor_indices)): # Iterate over conductor indices.
            for X in range(steps):
                for Z in range(3):
                    index = conductor_indices[Y]
                    mobile_pos[Y, X, Z] = position_data[int(index), X, Z] # Store mobile positions.

        drift = np.average(framework_disp, axis=0) # Calculate average drift of framework.
        corrected_displacements = disp - drift # Correct displacements by subtracting drift.

        drifted_rectified_structure = np.zeros((len(total_ion_array), steps, 3)) # Initialize drift-corrected structure array.
        for i in range(steps): # Iterate over steps.
            for Z in range(3):
                drifted_rectified_structure[:, i, Z] = position_data[:, 0, Z] + corrected_displacements[:, i, Z] # Calculate drift-corrected positions.

        mobile_drifted_rectified_structure = np.zeros((len(conductor_indices), steps, 3)) # Initialize mobile drift-corrected structure array.
        framework_drifted_rectified_structure = np.zeros((len(framework_indices), steps, 3)) # Initialize framework drift-corrected structure array.
        for Y in range(len(conductor_indices)): # Iterate over conductor indices.
            for X in range(steps):
                for Z in range(3):
                    index = conductor_indices[Y]
                    mobile_drifted_rectified_structure[Y, X, Z] = drifted_rectified_structure[int(index), X, Z] # Store mobile drift-corrected positions.
        for Y in range(len(framework_indices)): # Iterate over framework indices.
            for X in range(steps):
                for Z in range(3):
                    index = framework_indices[Y]
                    framework_drifted_rectified_structure[Y, X, Z] = drifted_rectified_structure[int(index), X, Z] # Store framework drift-corrected positions.

        position_data_list.append(position_data) # Append position data to list.
        drifted_rectified_structure_list.append(drifted_rectified_structure) # Append drift-corrected structure to list.
        conductor_indices_list.append(conductor_indices) # Append conductor indices to list.
        framework_indices_list.append(framework_indices) # Append framework indices to list.
        framework_pos_list.append(framework_pos) # Append framework positions to list.
        mobile_pos_list.append(mobile_pos) # Append mobile positions to list.
        pos_list.append(pos) # Append original positions to list.
        mobile_drifted_rectified_structure_list.append(mobile_drifted_rectified_structure) # Append mobile drift-corrected structure to list.
        framework_drifted_rectified_structure_list.append(framework_drifted_rectified_structure) # Append framework drift-corrected structure to list.

    return np.array(position_data_list), np.array(drifted_rectified_structure_list), np.array(conductor_indices_list), np.array(framework_indices_list), np.array(framework_pos_list), np.array(mobile_pos_list), np.array(mobile_drifted_rectified_structure_list), np.array(framework_drifted_rectified_structure_list)
    # Return all processed arrays.


