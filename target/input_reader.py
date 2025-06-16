import numpy as np
import pandas as pd

def read_ion_file(ion_file):
    """
    Read and extract ion information from an ion file.
    
    This function parses an ion file to extract atomic species information
    from the ATOMIC_POSITIONS section.
    
    Args:
        ion_file (str): Path to the ion file containing atomic position data
        
    Returns:
        list: List of element symbols (strings) representing the atomic species
    """
    total_ions = []
    
    # Read all lines from the ion file
    with open(ion_file, 'r') as file:
        lines = file.readlines()
    
    # Find the line containing 'ATOMIC_POSITIONS' keyword
    line_number = None
    for i, line in enumerate(lines):
        elements = line.split()
        if len(elements) == 0:
            continue
        if 'ATOMIC_POSITIONS' in elements:
            line_number = i
            break
    
    # Extract ion types from lines following ATOMIC_POSITIONS
    for line in lines[line_number + 1:]:
        # Skip empty lines at the beginning
        if len(line.strip()) == 0:
            if len(total_ions) == 0:
                continue
            else:
                break  # Stop when we reach an empty line after collecting ions
        
        elements = line.split()
        element = elements[0]  # First element is the atomic species
        total_ions.append(element)
    
    return total_ions

def read_cel_file(cel_file, Length_conversion_factor):
    """
    Read cell parameters from a cel file and apply length conversion.
    
    This function reads cell vectors from a cel file, where every 4 lines
    represent one time step (header + 3 cell vectors).
    
    Args:
        cel_file (str): Path to the cel file containing cell parameter data
        Length_conversion_factor (float): Factor to convert length units
        
    Returns:
        numpy.ndarray: Array of shape (n_timesteps, 9) containing flattened
                      cell vectors for each time step
    """
    cell_data = []
    
    with open(cel_file, 'r') as file:
        lines = file.readlines()
        
        # Process every 4 lines (skip header, read 3 cell vectors)
        for i in range(0, len(lines), 4):
            # Read and convert the three cell vectors (lines i+1, i+2, i+3)
            values = [float(val)*Length_conversion_factor for val in lines[i + 1].split()]
            values += [float(val)*Length_conversion_factor for val in lines[i + 2].split()]
            values += [float(val)*Length_conversion_factor for val in lines[i + 3].split()]
            cell_data.append(values)
    
    return np.array(cell_data)

def read_evp_file(evp_file, Length_conversion_factor):
    """
    Read energy, volume, and pressure data from an EVP file.
    
    This function parses an EVP file to extract thermodynamic properties
    and calculates time step information from the first two entries.
    
    Args:
        evp_file (str): Path to the EVP file containing energy/volume/pressure data
        Length_conversion_factor (float): Factor to convert length units (affects volume)
        
    Returns:
        tuple: Contains the following 11 elements:
            - ke_electronic (list): Electronic kinetic energy values
            - cell_temp (list): Cell temperature values
            - ion_temp (list): Ion temperature values
            - tot_energy (list): Total energy values
            - enthalpy (list): Enthalpy values
            - tot_energy_ke_ion (list): Total energy + ion kinetic energy
            - tot_energy_ke_ion_ke_elec (list): Total energy + ion KE + electronic KE
            - vol (list): Volume values (converted to target units)
            - pressure (list): Pressure values
            - number_of_time_frames (float): Time step difference between first two frames
            - time_difference (float): Time difference between first two frames
    """
    # Initialize lists for all thermodynamic properties
    ke_electronic, cell_temp, ion_temp, tot_energy, enthalpy = [], [], [], [], []
    tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure = [], [], [], []
    
    # Variables to calculate time step information
    first_term, first_time, second_term, second_time = None, None, None, None
    
    # Read file and filter out comment lines
    with open(evp_file, 'r') as file:
        lines = [line.strip() for line in file.readlines() if not line.startswith('#')]
    
    # Parse each line of data
    for index, line in enumerate(lines):
        values = line.split()
        
        # Store first and second frame information for time calculation
        if index == 0:
            first_term = float(values[0])
            first_time = float(values[1])
        elif index == 1:
            second_term = float(values[0])
            second_time = float(values[1])
        
        # Extract all thermodynamic properties
        ke_electronic.append(float(values[2]))
        cell_temp.append(float(values[3]))
        ion_temp.append(float(values[4]))
        tot_energy.append(float(values[5]))
        enthalpy.append(float(values[6]))
        tot_energy_ke_ion.append(float(values[7]))
        tot_energy_ke_ion_ke_elec.append(float(values[8]))
        vol.append(float(values[9]) * Length_conversion_factor**3)  # Convert volume units
        pressure.append(float(values[10]))
    
    # Calculate time step information
    number_of_time_frames = second_term - first_term if first_term and second_term else None
    time_difference = second_time - first_time if first_time and second_time else None
    
    return (ke_electronic, cell_temp, ion_temp, tot_energy, enthalpy, 
            tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure, 
            number_of_time_frames, time_difference)

def read_pos_file(pos_file, total_ions, Length_conversion_factor, number_of_time_frames, time_difference):
    """
    Read atomic positions from a position file and organize by ion and time step.
    
    This function reads position data where each time step contains positions
    for all ions, separated by header lines.
    
    Args:
        pos_file (str): Path to the position file
        total_ions (list): List of ion types/elements
        Length_conversion_factor (float): Factor to convert length units
        number_of_time_frames (float): Number of time frames between steps
        time_difference (float): Time difference between consecutive frames
        
    Returns:
        tuple: Contains the following 4 elements:
            - ion_disp (numpy.ndarray): Array of shape (n_ions, n_timesteps, 3) 
                                      containing ion positions
            - len(data_steps) (int): Total number of time steps
            - dt (numpy.ndarray): Array of time differences for each step
            - time (numpy.ndarray): Array of absolute time values
    """
    # Read position data using pandas fixed-width format
    pos_data = pd.read_fwf(pos_file, header=None)
    
    # Extract time step headers (every (n_ions + 1) lines)
    data_steps = pos_data[0: len(pos_data): len(total_ions) + 1]
    
    # Create time arrays
    d_idx = np.arange(1, len(data_steps), 1)  # Index array starting from 1
    t_ind = np.arange(0, len(data_steps), 1)  # Index array starting from 0
    
    # Calculate time arrays
    time = t_ind * time_difference + np.ones(len(t_ind)) * time_difference
    dt = d_idx * time_difference
    
    # Initialize displacement array: (n_ions, n_timesteps, 3_coordinates)
    ion_disp = np.zeros((len(total_ions), len(data_steps), 3))
    
    # Extract position data for each ion
    for i in range(len(total_ions)):
        # Get positions for ion i at all time steps (skip headers)
        ion_data = pos_data[i + 1:len(pos_data):len(total_ions) + 1]
        
        # Convert to numpy array and apply length conversion
        ion_pos = ion_data.to_numpy().astype('float') * Length_conversion_factor
        ion_disp[i] = ion_pos
    
    return ion_disp, len(data_steps), dt, time