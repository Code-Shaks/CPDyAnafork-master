"""
Input File Reader Module for CPDyAna
====================================

This module provides functions for reading and parsing various input file formats
used in molecular dynamics simulations, particularly those from Quantum ESPRESSO
and other DFT/MD codes.

Supported file formats:
- .pos files: Position trajectories with atomic coordinates
- .cel files: Unit cell parameters and lattice vectors
- .evp files: Energy, volume, and pressure data
- .in files: Ion definitions and atomic species information

The module handles unit conversions (typically Bohr to Angstrom) and provides
structured data arrays suitable for analysis functions.

Functions:
    read_ion_file: Parse ion definition files
    read_cel_file: Read unit cell parameter files
    read_evp_file: Parse energy/volume/pressure files
    read_pos_file: Read position trajectory files

Author: CPDyAna Development Team
Version: 01-02-2024
"""

import numpy as np
import pandas as pd

def read_ion_file(ion_file):
    """
    Read and parse ion definition file containing atomic species information.
    
    The ion file typically contains atomic symbols, masses, and other properties
    for each species in the simulation. This function extracts this information
    and returns it in a structured format.
    
    Args:
        ion_file_path (str): Path to the ion definition file (.in format).
        
    Returns:
        list: List of dictionaries containing ion information with keys:
            - 'symbol': Atomic symbol (e.g., 'Li', 'Al', 'P', 'S')
            - 'mass': Atomic mass in atomic mass units
            - 'index': Index in the simulation
            - Additional properties as available
            
    Raises:
        FileNotFoundError: If the ion file cannot be found.
        ValueError: If the file format is invalid or corrupted.
        
    Example:
        >>> ions = read_ion_file('simulation.in')
        >>> print(f"Found {len(ions)} ion types")
        >>> for ion in ions:
        ...     print(f"{ion['symbol']}: {ion['mass']} amu")
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
    Read unit cell parameter file containing lattice vectors and cell dimensions.
    
    The cell file contains time-dependent lattice vectors that define the
    simulation box boundaries. This is crucial for applying periodic boundary
    conditions and calculating volumes.
    
    Args:
        cel_file_path (str): Path to the cell parameter file (.cel format).
        conv_factor (float): Unit conversion factor (default: 1.0).
            Typically 0.529177249 for Bohr to Angstrom conversion.
            
    Returns:
        np.ndarray: Array with shape (n_frames, 9) containing lattice vectors:
            [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z, a3_x, a3_y, a3_z]
            where a1, a2, a3 are the three lattice vectors for each frame.
            
    Raises:
        FileNotFoundError: If the cell file cannot be found.
        ValueError: If the file contains invalid cell data.
        
    Example:
        >>> cell_params = read_cel_file('md.cel', conv_factor=0.529177249)
        >>> print(f"Cell shape: {cell_params.shape}")
        >>> # Calculate volume for first frame
        >>> a1, a2, a3 = cell_params[0].reshape(3, 3)
        >>> volume = np.abs(np.dot(a1, np.cross(a2, a3)))
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
    Read energy, volume, and pressure file containing thermodynamic properties.
    
    The EVP file contains time-series data of various thermodynamic quantities
    including kinetic energies, total energy, enthalpy, volume, pressure, and
    temperatures for different components of the system.
    
    Args:
        evp_file_path (str): Path to the EVP file (.evp format).
        conv_factor (float): Unit conversion factor for energy and volume.
            
    Returns:
        tuple: Contains the following arrays:
            - ke_elec (np.ndarray): Electronic kinetic energy
            - cell_temp (np.ndarray): Cell temperature
            - ion_temp (np.ndarray): Ionic temperature  
            - tot_energy (np.ndarray): Total energy
            - enthalpy (np.ndarray): Enthalpy
            - tot_energy_ke_ion (np.ndarray): Total energy + ionic KE
            - tot_energy_ke_ion_ke_elec (np.ndarray): Total energy + all KE
            - volume (np.ndarray): Cell volume
            - pressure (np.ndarray): Pressure
            - n_frames (int): Number of time frames
            - time_diff (float): Time step between frames
            
    Raises:
        FileNotFoundError: If the EVP file cannot be found.
        ValueError: If the file format is invalid.
        
    Example:
        >>> evp_data = read_evp_file('md.evp', conv_factor=0.529177249)
        >>> ke_elec, temp, pressure = evp_data[0], evp_data[2], evp_data[8]
        >>> print(f"Average temperature: {np.mean(temp):.1f} K")
        >>> print(f"Average pressure: {np.mean(pressure):.2f} GPa")
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
    Read position trajectory file containing atomic coordinates over time.
    
    The position file contains the time evolution of atomic coordinates for all
    atoms in the simulation. This function parses the file and organizes the
    data into arrays suitable for analysis.
    
    Args:
        pos_file_path (str): Path to the position file (.pos format).
        ion_array (list): Ion information from read_ion_file().
        conv_factor (float): Unit conversion factor (Bohr to Angstrom).
        n_frames (int, optional): Number of frames to read. If None, read all.
        time_diff (float, optional): Time step between frames in ps.
        
    Returns:
        tuple: Contains:
            - positions (np.ndarray): Position array with shape (n_frames, n_atoms, 3)
            - steps (np.ndarray): Time step numbers
            - dt (np.ndarray): Time differences between steps
            - time (np.ndarray): Absolute time values in picoseconds
            
    Raises:
        FileNotFoundError: If the position file cannot be found.
        ValueError: If file format is invalid or inconsistent with ion data.
        
    Example:
        >>> ions = read_ion_file('md.in')
        >>> pos, steps, dt, time = read_pos_file('md.pos', ions, 0.529177249)
        >>> print(f"Trajectory shape: {pos.shape}")
        >>> print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} ps")
        >>> # Access Li atom positions (assuming Li is first species)
        >>> li_positions = pos[:, :n_li_atoms, :]
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