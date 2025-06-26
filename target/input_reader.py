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
import os
import glob

from samos_modules.samos_io import read_lammps_dump

from ase import Atoms
from ase.io import read, write

def detect_trajectory_format(data_dir):
    """
    Automatically detect trajectory file formats in the data directory.
    Returns:
        dict: Format information with file paths
    """
    format_info = {
        'format': None,
        'pos_files': [],
        'cel_files': [],
        'evp_files': [],
        'ion_files': [],
        'lammps_files': []
    }

    # Check for LAMMPS trajectory files
    lammps_patterns = ['*.lammpstrj', '*.dump', '*.lmp']
    for pattern in lammps_patterns:
        lammps_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if lammps_files:
            format_info['format'] = 'lammps'
            format_info['lammps_files'] = lammps_files
            return format_info

    # Check for Quantum ESPRESSO files
    qe_files = {
        'pos': sorted(glob.glob(os.path.join(data_dir, "*.pos"))),
        'cel': sorted(glob.glob(os.path.join(data_dir, "*.cel"))),
        'evp': sorted(glob.glob(os.path.join(data_dir, "*.evp"))),
        'ion': sorted(glob.glob(os.path.join(data_dir, "*.in")))
    }

    if all(qe_files.values()):
        format_info['format'] = 'quantum_espresso'
        format_info.update({f'{k}_files': v for k, v in qe_files.items()})
        return format_info

    # Check for generic formats supported by ASE
    ase_patterns = ['*.xyz', '*.extxyz', '*.traj']
    for pattern in ase_patterns:
        ase_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if ase_files:
            format_info['format'] = 'ase_compatible'
            format_info['trajectory_files'] = ase_files
            return format_info

    return format_info

def generate_analysis_recommendations(times, time_difference, n_frames, inp_array):
    """
    Generate smart analysis parameter recommendations based on trajectory characteristics.
    """
    total_time = times[-1] - times[0]
    equilibration_time = max(10.0, total_time * 0.1)  # 10% or min 10 ps
    analysis_start = equilibration_time
    analysis_end = times[-1]
    slope_start = equilibration_time + 5.0
    slope_end = times[-1] - 5.0
    
    # Frame indices for analysis
    start_frame = int(analysis_start / time_difference)
    end_frame = int(analysis_end / time_difference)
    analysis_frames = end_frame - start_frame
    
    print("\n=== ANALYSIS PARAMETER RECOMMENDATIONS ===")
    print("FOR MSD/VH/NGP ANALYSIS:")
    print(f" --initial-time {analysis_start:.1f}")
    print(f" --final-time {analysis_end:.1f}")
    print(f" --initial-slope-time {slope_start:.1f}")
    print(f" --final-slope-time {slope_end:.1f}")
    print(f" --lammps-timestep {time_difference:.6f}")
    
    # Get unique elements
    unique_elements = list(set(inp_array))
    print(f" --diffusing-elements {' '.join(unique_elements[:3])}")  # First 3 elements
    print(f" --lammps-elements {' '.join(unique_elements)}")
    
    print("\nFOR RDF ANALYSIS:")
    print(f" --time-after-start {equilibration_time:.1f}")
    print(f" --time-interval {time_difference:.6f}")
    print(f" --num-frames {min(500, analysis_frames)}")
    print(f" --central-atom {' '.join(unique_elements)}")
    
    print("\nFOR VAF/VDOS ANALYSIS:")
    print(f" --start {equilibration_time:.1f}")
    print(f" --nframes {analysis_frames}")
    print(f" --stride 1")
    print(f" --time-interval {time_difference:.6f}")

def export_verification_trajectory(positions, cells, inp_array, times, output_path="cpdyana_verification.xyz"):
    """
    Export verification trajectory for quality assurance.
    """
    print(f"\n=== EXPORTING VERIFICATION TRAJECTORY ===")
    atoms_list = []
    export_frames = min(len(positions), 100)  # First 100 frames
    
    for i in range(export_frames):
        atoms = Atoms(symbols=inp_array,
                     positions=positions[i],
                     cell=cells[i],
                     pbc=True)
        atoms.info['time'] = times[i]
        atoms.info['step'] = i
        atoms_list.append(atoms)
    
    write(output_path, atoms_list, format='extxyz')
    print(f"Verification trajectory (first {export_frames} frames) saved as '{output_path}'")

def has_type_column(lammps_file):
    # Read the first 100 lines to find the header
    with open(lammps_file, 'r') as f:
        for line in f:
            if line.startswith('ITEM: ATOMS'):
                # Example: ITEM: ATOMS id type xu yu zu
                header = line.strip().split()[2:]
                return 'type' in header
    return False

def read_lammps_trajectory(lammps_file, elements=None, timestep=None,
                           thermo_file=None, Conv_factor=1.0, element_mapping=None,
                           export_verification=False, show_recommendations=False):
    """
    Enhanced LAMMPS trajectory reader using SAMOS's methodology for MSD calculation.
    """
    print(f"\n=== ENHANCED LAMMPS TRAJECTORY PROCESSING WITH SAMOS ===")
    print(f"Reading LAMMPS file: {lammps_file}")
    print(f"Element types provided: {elements}")
    print(f"Timestep: {timestep} ps")
    print(f"Element mapping provided: {element_mapping}")

    # Use SAMOS I/O to read LAMMPS trajectory
    from samos.io.lammps import read_lammps_dump
    has_type = has_type_column(lammps_file)
    
    # Only pass types if 'type' column is present
    kwargs = dict(
        filename=lammps_file,
        timestep=timestep * 1000 if timestep else None,  # Convert ps to fs for SAMOS
        thermo_file=thermo_file,
        f_conv=Conv_factor,
        e_conv=Conv_factor,
        quiet=False
    )
    
    if has_type and elements:
        kwargs['types'] = elements
    
    trajectory = read_lammps_dump(**kwargs)
    
    # Extract trajectory data
    positions = trajectory.get_positions()  # Shape: (n_frames, n_atoms, 3)
    cells = trajectory.get_cells()  # Shape: (n_frames, 3, 3)
    n_frames, n_atoms, _ = positions.shape
    print(f"Trajectory shape: {positions.shape} (frames, atoms, xyz)")
    print(f"Number of frames: {n_frames}")
    print(f"Number of atoms: {n_atoms}")
    
    # Time array generation (SAMOS uses femtoseconds internally)
    try:
        times = trajectory.get_times() / 1000.0  # Convert fs to ps for CPDyAna
        print(f"Time array from trajectory: {times[:5]}...{times[-5:]}")
    except AttributeError:
        if timestep is None:
            raise ValueError("Timestep not provided and trajectory.get_times() is not available")
        times = np.arange(n_frames) * timestep
        print(f"Generated time array with timestep {timestep} ps")
    
    # Calculate time parameters
    dt_full = np.diff(times) if len(times) > 1 else np.array([timestep or 0.001])
    time_difference = dt_full[0] if len(dt_full) > 0 else (timestep or 0.001)
    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} ps")
    print(f"Time step: {time_difference:.6f} ps ({time_difference*1000:.3f} fs)")
    print(f"Total simulation time: {times[-1] - times[0]:.3f} ps")
    
    # Element mapping using SAMOS's extracted types
    try:
        atom_types = trajectory.get_types()
        print(f"Retrieved atom types from trajectory: {set(atom_types)}")
    except AttributeError:
        print("Warning: Could not retrieve atom types from trajectory")
        atom_types = None
    
    # Create element array using enhanced mapping
    if element_mapping and atom_types is not None:
        print("Using element mapping to assign elements...")
        inp_array = []
        for atom_type in atom_types:
            element = element_mapping.get(atom_type, 'H')
            inp_array.append(element)
        print(f"Element assignment complete. Sample: {inp_array[:10]}...")
        print(f"Element distribution: {dict(zip(*np.unique(inp_array, return_counts=True)))}")
    elif elements and len(elements) == n_atoms:
        print("Using provided elements array (length matches atom count)")
        inp_array = elements
    elif elements:
        print(f"Element count mismatch: Expected {n_atoms}, got {len(elements)}")
        print("Extending element list to match atom count...")
        inp_array = []
        element_count = len(elements)
        for i in range(n_atoms):
            element_idx = i % element_count
            inp_array.append(elements[element_idx])
        print(f"Extended element assignment complete")
    else:
        print("⚠️ Warning: No elements or mapping provided. Defaulting to 'H' for all atoms.")
        inp_array = ['H'] * n_atoms
    
    # SAMOS handles unwrapped positions internally if specified in dump file
    print("Using SAMOS's position handling (unwrapping if specified in dump).")
    
    # Convert to CPDyAna format for compatibility with other functions
    pos_full = np.transpose(positions, (1, 0, 2))  # (atoms, frames, xyz)
    cell_param_full = cells.reshape(n_frames, 9)  # Flatten 3x3 to 9-element vectors
    
    # Thermodynamic data generation
    print("\n=== CREATING ENHANCED TRAJECTORY ARRAYS ===")
    t_full = times
    steps_full = np.arange(n_frames)
    ke_elec_full = np.zeros(n_frames)
    cell_temp_full = np.ones(n_frames) * 300.0  # Default 300K
    ion_temp_full = np.ones(n_frames) * 300.0
    tot_energy_full = np.zeros(n_frames)
    enthalpy_full = np.zeros(n_frames)
    tot_energy_ke_ion_full = np.zeros(n_frames)
    tot_energy_ke_ion_ke_elec_full = np.zeros(n_frames)
    vol_full = np.array([np.linalg.det(cell) for cell in cells])
    pressure_full = np.zeros(n_frames)
    print(f"pos_full shape: {pos_full.shape} (atoms, frames, xyz)")
    print(f"cell_param_full shape: {cell_param_full.shape} (frames, 9_params)")
    print(f"Volume range: {vol_full.min():.2f} to {vol_full.max():.2f} Å³")
    
    # Analysis parameter recommendations
    if show_recommendations:
        generate_analysis_recommendations(times, time_difference, n_frames, inp_array)
    
    # Optional trajectory verification export
    if export_verification:
        export_verification_trajectory(positions, cells, inp_array, times)
    
    # Extract thermodynamic data
    thermo_data = {}
    if hasattr(trajectory, 'get_pot_energies'):
        try:
            thermo_data['potential_energy'] = trajectory.get_pot_energies()
        except:
            pass
    
    volumes = [np.linalg.det(cell) for cell in cells]
    print(f"\n=== ENHANCED PROCESSING COMPLETE ===")
    return (pos_full, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array)

def read_ion_file_universal(ion_file_or_elements):
    """
    Universal ion reader that handles both .in files and element lists.
    
    Args:
        ion_file_or_elements: Either path to .in file or list of elements
        
    Returns:
        list: Ion symbols
    """
    if isinstance(ion_file_or_elements, str):
        # Original QE .in file reader
        return read_ion_file(ion_file_or_elements)
    elif isinstance(ion_file_or_elements, list):
        # Direct element list from LAMMPS
        return ion_file_or_elements
    else:
        raise ValueError("Invalid ion file or elements specification")

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