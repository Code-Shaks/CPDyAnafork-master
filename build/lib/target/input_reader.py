import pandas as pd
import numpy as np

def read_ion_file(ion_file):
    """
    Read atomic positions of ions from a file.

    Parameters:
    - ion_file (str): Path to the file containing ion positions.

    Returns:
    - list: Atomic symbols representing the ions.
    """
    total_ions = [] # Initialize an empty list to store ion symbols.
    with open(ion_file, 'r') as file: # Open the ion file for reading.
        lines = file.readlines() # Read all lines from the file.
    line_number = None # Initialize line number to None.
    for i, line in enumerate(lines): # Iterate over lines with index.
        elements = line.split() # Split the line into elements.
        if len(elements) == 0: # If the line is empty, continue to the next line.
            continue
        if 'ATOMIC_POSITIONS' in elements: # Check for 'ATOMIC_POSITIONS' keyword.
            line_number = i # Set the line number where 'ATOMIC_POSITIONS' is found.
            break
    for line in lines[line_number + 1:]: # Iterate over lines after 'ATOMIC_POSITIONS'.
        if len(line.strip()) == 0: # If the line is empty, continue.
            if len(total_ions) == 0: # If no ions have been added yet, continue.
                continue
            else:
                break # Break the loop if the line is empty and ions have been added.
        elements = line.split() # Split the line into elements.
        element = elements[0] # Get the first element (ion symbol).
        total_ions.append(element) # Append the ion symbol to the list.
    return total_ions # Return the list of ion symbols.

def read_cel_file(cel_file, Length_conversion_factor):
    """
    Read cell parameters from a file and convert them to desired units.

    Parameters:
    - cel_file (str): Path to the file containing cell parameters.
    - Length_conversion_factor (float): Factor for converting lengths to desired units.

    Returns:
    - numpy.ndarray: Cell parameters converted to the specified units.
    """
    cell_data = [] # Initialize an empty list to store cell data.
    with open(cel_file, 'r') as file: # Open the cell file for reading.
        lines = file.readlines() # Read all lines from the file.
        for i in range(0, len(lines), 4): # Iterate over lines in steps of 4.
            values = [float(val)*Length_conversion_factor for val in lines[i + 1].split()] # Convert and append first line values.
            values += [float(val)*Length_conversion_factor for val in lines[i + 2].split()] # Convert and append second line values.
            values += [float(val)*Length_conversion_factor for val in lines[i + 3].split()] # Convert and append third line values.
            cell_data.append(values) # Append the converted values to cell_data list.
    return np.array(cell_data) # Convert cell_data to numpy array and return.

def read_evp_file(evp_file, Length_conversion_factor):
    """
    Read thermodynamic properties from an evp file and convert units.

    Parameters:
    - evp_file (str): Path to the evp file containing thermodynamic properties.
    - Length_conversion_factor (float): Factor for converting lengths to desired units.

    Returns:
    - tuple: Thermodynamic properties with converted units, along with time-related information.
    """
    ke_electronic = [] # Initialize list for kinetic energy of electrons.
    cell_temp = [] # Initialize list for cell temperature.
    ion_temp = [] # Initialize list for ion temperature.
    tot_energy = [] # Initialize list for total energy.
    enthalpy = [] # Initialize list for enthalpy.
    tot_energy_ke_ion = [] # Initialize list for total energy including kinetic energy of ions.
    tot_energy_ke_ion_ke_elec = [] # Initialize list for total energy including kinetic energy of ions and electrons.
    vol = [] # Initialize list for volume.
    pressure = [] # Initialize list for pressure.
    first_term = None # Initialize first term to None.
    first_time = None # Initialize first time to None.
    second_term = None # Initialize second term to None.
    second_time = None # Initialize second time to None.
    with open(evp_file, 'r') as file: # Open the evp file for reading.
        lines = [line.strip() for line in file.readlines() if not line.startswith('#')] # Read lines, ignoring comments.
    for index, line in enumerate(lines): # Iterate over lines with index.
        values = line.split() # Split line into values.
        if index == 0: # For the first line.
            first_term = float(values[0]) # Get the first term.
            first_time = float(values[1]) # Get the first time.
        elif index == 1: # For the second line.
            second_term = float(values[0]) # Get the second term.
            second_time = float(values[1]) # Get the second time.
        ke_electronic.append(float(values[2])) # Append kinetic energy of electrons.
        cell_temp.append(float(values[3])) # Append cell temperature.
        ion_temp.append(float(values[4])) # Append ion temperature.
        tot_energy.append(float(values[5])) # Append total energy.
        enthalpy.append(float(values[6])) # Append enthalpy.
        tot_energy_ke_ion.append(float(values[7])) # Append total energy with ion kinetic energy.
        tot_energy_ke_ion_ke_elec.append(float(values[8])) # Append total energy with ion and electron kinetic energy.
        vol.append(float(values[9])*Length_conversion_factor**3) # Append volume with conversion.
        pressure.append(float(values[10])) # Append pressure.
    if first_term is not None and second_term is not None: # Check if terms are defined.
        number_of_time_frames = second_term - first_term # Calculate number of time frames.
        time_difference = second_time - first_time # Calculate time difference.
    else:
        number_of_time_frames = None # Set number of time frames to None.
        time_difference = None # Set time difference to None.
    return ke_electronic, cell_temp, ion_temp, tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure, number_of_time_frames, time_difference
    # Return all thermodynamic properties and time-related information.

def read_pos_file(pos_file, total_ions, Length_conversion_factor, number_of_time_frames, time_difference):
    """
    Read ion position data from a file, convert units, and calculate time-related parameters.

    Parameters:
    - pos_file (str): Path to the file containing ion position data.
    - total_ions (list): List of atomic symbols representing the ions.
    - Length_conversion_factor (float): Factor for converting lengths to desired units.
    - number_of_time_frames (int): Number of time frames in the simulation.
    - time_difference (float): Time difference between consecutive frames.

    Returns:
    - tuple: Ion displacement data, number of time steps, time increments, and absolute time values.
    """
    pos_data = pd.read_fwf(pos_file, header=None) # Read fixed-width formatted file into DataFrame.
    data_steps = pos_data[0: len(pos_data): len(total_ions) + 1] # Extract data steps.
    d_idx = np.arange(1, len(data_steps), 1) # Create array of indices for time increments.
    t_ind = np.arange(0, len(data_steps), 1) # Create array of time indices.
    time = t_ind * time_difference + np.ones(len(t_ind))*time_difference # Calculate absolute time values.
    dt = d_idx * time_difference # Calculate time increments.
    print(len(dt), len(time)) # Print lengths of dt and time arrays.
    ion_disp = np.zeros((len(total_ions), len(data_steps), 3)) # Initialize array for ion displacements.
    for i in range(len(total_ions)): # Iterate over each ion.
        ion_data = pos_data[i + 1:len(pos_data):len(total_ions) + 1] # Extract ion data for each time step.
        ion_pos = ion_data.to_numpy().astype('float')*Length_conversion_factor # Convert ion positions to numpy array and apply length conversion.
        ion_disp[i] = ion_pos # Store converted ion positions in ion_disp array.
    return ion_disp, len(data_steps), dt, time # Return ion displacement data, number of steps, time increments, and absolute time values.
