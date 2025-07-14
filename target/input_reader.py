"""
Input File Reader Module for CPDyAna
====================================

This module provides functions for reading and parsing various input file formats
used in molecular dynamics simulations, particularly those from Quantum ESPRESSO,
LAMMPS, and BOMD (.trj).

Supported file formats:
- .pos files: Position trajectories with atomic coordinates
- .cel files: Unit cell parameters and lattice vectors
- .evp files: Energy, volume, and pressure data
- .in files: Ion definitions and atomic species information
- LAMMPS dump files: Trajectories from LAMMPS
- BOMD .trj files: Born-Oppenheimer MD trajectories

The module handles unit conversions (typically Bohr to Angstrom) and provides
structured data arrays suitable for analysis functions.

Functions:
    detect_trajectory_format: Auto-detects trajectory format in a directory.
    generate_analysis_recommendations: Prints recommended analysis parameters.
    export_verification_trajectory: Writes a short trajectory for verification.
    has_type_column: Checks if LAMMPS dump file has a 'type' column.
    read_lammps_trajectory: Reads and parses LAMMPS trajectory files.
    read_bomd_trajectory: Reads and parses BOMD .trj trajectory files.
    read_ion_file_universal: Reads ion info from file or list.
    read_ion_file: Reads atomic species from .in file.
    read_cel_file: Reads cell parameters from .cel file.
    read_evp_file: Reads thermodynamic data from .evp file.
    read_pos_file: Reads position trajectory from .pos file.

Author: CPDyAna Development Team
Version: 01-02-2024
"""

import numpy as np
import pandas as pd
import os
import glob

from target.io import read_lammps_dump, iter_lammps_dump

from ase import Atoms
from ase.io import read, write

def detect_trajectory_format(data_dir):
    """
    Automatically detect trajectory file formats in the data directory.

    Args:
        data_dir (str): Path to directory containing trajectory files.

    Returns:
        dict: Format information with file paths and detected format.
    """
    format_info = {
        'format': None,
        'pos_files': [],
        'cel_files': [],
        'evp_files': [],
        'ion_files': [],
        'lammps_files': [],
        'xsf_files': [],
        'vasp_files': [],
        'cp2k_files': [],
        'gromacs_files': [],
        'bomd_files': [],
        'trajectory_files': []
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
    
    # BOMD .trj or .trajectory files
    bomd_patterns = ['*.trj', '*.trajectory', '*.mdtrj']
    for pattern in bomd_patterns:
        bomd_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if bomd_files:
            format_info['format'] = 'bomd'
            format_info['bomd_files'] = bomd_files
            return format_info

    # XSF detection
    xsf_files = sorted(glob.glob(os.path.join(data_dir, "*.xsf")))
    if xsf_files:
        format_info['format'] = 'xsf'
        format_info['xsf_files'] = xsf_files
        return format_info

    # VASP detection (OUTCAR, XDATCAR, etc.)
    vasp_files = sorted(glob.glob(os.path.join(data_dir, "OUTCAR"))) + \
                 sorted(glob.glob(os.path.join(data_dir, "XDATCAR")))
    if vasp_files:
        format_info['format'] = 'vasp'
        format_info['vasp_files'] = vasp_files
        return format_info

    # CP2K detection (.pdb, .xyz with CP2K header)
    cp2k_files = sorted(glob.glob(os.path.join(data_dir, "*.pdb"))) + \
                 sorted(glob.glob(os.path.join(data_dir, "*.xyz")))
    if cp2k_files:
        format_info['format'] = 'cp2k'
        format_info['cp2k_files'] = cp2k_files
        return format_info

    # GROMACS detection (.gro, .xtc)
    gromacs_files = sorted(glob.glob(os.path.join(data_dir, "*.gro"))) + \
                    sorted(glob.glob(os.path.join(data_dir, "*.xtc")))
    if gromacs_files:
        format_info['format'] = 'gromacs'
        format_info['gromacs_files'] = gromacs_files
        return format_info

    # Generic ASE-compatible detection (fallback)
    ase_patterns = ['*.xyz', '*.extxyz', '*.traj']
    for pattern in ase_patterns:
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if files:
            format_info['format'] = 'ase_compatible'
            format_info['trajectory_files'] = files
            return format_info

    return format_info

def generate_analysis_recommendations(times, time_difference, n_frames, inp_array):
    """
    Generate smart analysis parameter recommendations based on trajectory characteristics.

    Args:
        times (np.ndarray): Array of time values.
        time_difference (float): Time step between frames.
        n_frames (int): Number of frames.
        inp_array (list): List of element symbols.

    Returns:
        None. Prints recommendations to stdout.
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
    Export a short verification trajectory for quality assurance.

    Args:
        positions (np.ndarray): Array of positions (frames, atoms, 3).
        cells (np.ndarray): Array of cell matrices (frames, 3, 3).
        inp_array (list): List of element symbols.
        times (np.ndarray): Array of time values.
        output_path (str): Output file path.

    Returns:
        None. Writes a trajectory file.
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
    """
    Check if a LAMMPS dump file has a 'type' column in its ATOMS section.

    Args:
        lammps_file (str): Path to LAMMPS dump file.

    Returns:
        bool: True if 'type' column is present, False otherwise.
    """
    with open(lammps_file, 'r') as f:
        for line in f:
            if line.startswith('ITEM: ATOMS'):
                # Example: ITEM: ATOMS id type xu yu zu
                header = line.strip().split()[2:]
                return 'type' in header
    return False

# def iter_lammps_trajectory(lammps_file, elements=None, timestep=None,
#                            Conv_factor=1.0, element_mapping=None,
#                            yield_summary=False):
#     """
#     Generator version of read_lammps_trajectory: yields one frame at a time as a dict,
#     including per-frame thermodynamic data (volume, etc.).
#     If yield_summary=True, yields a summary dict at the end with all arrays.
#     """
#     types_list = None
#     if element_mapping:
#         max_type = max(element_mapping)
#         types_list = [element_mapping[i] for i in range(1, max_type+1)]
#     elif elements:
#         types_list = elements

#     # Prepare accumulators for summary
#     positions_list = []
#     cells_list = []
#     volumes_list = []
#     timesteps_list = []
#     symbols = None

#     print(f"\n=== LAZY LAMMPS TRAJECTORY PROCESSING WITH SAMOS ===")
#     print(f"Reading LAMMPS file: {lammps_file}")
#     print(f"Element types provided: {elements}")
#     print(f"Timestep: {timestep} ps")
#     print(f"Element mapping provided: {element_mapping}")

#     for frame in iter_lammps_dump(
#         filename=lammps_file,
#         elements=elements,
#         types=types_list,
#         timestep=timestep * 1000 if timestep else None,
#         f_conv=Conv_factor,
#         quiet=False
#     ):
#         positions = frame["positions"]
#         cell = frame["cell"]
#         symbols = frame["symbols"]
#         vol = np.linalg.det(cell)
#         tstep = frame.get("timestep", None)

#         # Accumulate for summary
#         positions_list.append(positions)
#         cells_list.append(cell)
#         volumes_list.append(vol)
#         timesteps_list.append(tstep)

#         # Per-frame thermodynamic data (can be expanded as needed)
#         frame["volume"] = vol
#         frame["cell_temp"] = 300.0  # Default
#         frame["ion_temp"] = 300.0   # Default
#         frame["pressure"] = 0.0     # Default
#         frame["tot_energy"] = 0.0   # Default
#         frame["enthalpy"] = 0.0     # Default

#         yield frame

#     if yield_summary:
#         # After all frames, yield a summary dict
#         positions_arr = np.array(positions_list)  # (n_frames, n_atoms, 3)
#         cells_arr = np.array(cells_list)          # (n_frames, 3, 3)
#         n_frames = len(positions_list)
#         n_atoms = positions_arr.shape[1] if n_frames > 0 else 0
#         pos_full = np.transpose(positions_arr, (1, 0, 2))  # (atoms, frames, xyz)
#         cell_param_full = cells_arr.reshape(n_frames, 9)
#         dt_full = np.array([timestep] * (n_frames - 1)) if timestep else np.ones(n_frames - 1)
#         t_full = np.concatenate(([0], np.cumsum(dt_full)))
#         vol_full = np.array(volumes_list)
#         inp_array = symbols if symbols is not None else ['H'] * n_atoms

#         print("\n=== CREATING ENHANCED TRAJECTORY ARRAYS ===")
#         print(f"pos_full shape: {pos_full.shape} (atoms, frames, xyz)")
#         print(f"cell_param_full shape: {cell_param_full.shape} (frames, 9_params)")
#         print(f"Volume range: {vol_full.min():.2f} to {vol_full.max():.2f} Å³")
#         summary = {
#             "pos_full": pos_full,
#             "n_frames": n_frames,
#             "dt_full": dt_full,
#             "t_full": t_full,
#             "cell_param_full": cell_param_full,
#             "volumes": vol_full,
#             "inp_array": inp_array,
#         }
#         yield summary

def read_lammps_trajectory(lammps_file, elements=None, timestep=None,
                           thermo_file=None, Conv_factor=1.0, element_mapping=None,
                           export_verification=False, show_recommendations=False):
    """
    Read and parse a LAMMPS trajectory file and convert to CPDyAna format.

    Args:
        lammps_file (str): Path to LAMMPS dump file.
        elements (list): List of element symbols for atom types.
        timestep (float): Timestep in ps.
        thermo_file (str): Optional, path to LAMMPS thermo file.
        Conv_factor (float): Conversion factor for units.
        element_mapping (dict): Mapping from type index to element symbol.
        export_verification (bool): Export verification trajectory.
        show_recommendations (bool): Print analysis recommendations.

    Returns:
        tuple: (pos_full, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array)
    """
    print(f"\n=== ENHANCED LAMMPS TRAJECTORY PROCESSING WITH SAMOS ===")
    print(f"Reading LAMMPS file: {lammps_file}")
    print(f"Element types provided: {elements}")
    print(f"Timestep: {timestep} ps")
    print(f"Element mapping provided: {element_mapping}")

    # Check if the LAMMPS dump file has a 'type' column
    has_type = has_type_column(lammps_file)
    
    # Prepare keyword arguments for read_lammps_dump
    kwargs = dict(
        filename=lammps_file,
        timestep=timestep * 1000 if timestep else None,  # Convert ps to fs
        thermo_file=thermo_file,
        f_conv=Conv_factor,
        e_conv=Conv_factor,
        quiet=False
    )
    
    # Assign types based on element_mapping or elements
    if has_type and element_mapping:
        # Convert mapping dict {1:'Li', 2:'La', ...} to list ['Li', 'La', ...]
        # Ensure order is by type number (1-based)
        max_type = max(element_mapping)
        types_list = [element_mapping[i] for i in range(1, max_type+1)]
        kwargs['types'] = types_list
        print("Using element mapping for types in trajectory reading (converted to list)")
    elif has_type and elements:
        kwargs['types'] = elements
        print("Using elements list for types in trajectory reading")

    # Read the trajectory using SAMOS
    trajectory = read_lammps_dump(**kwargs)
    
    # Extract trajectory data
    positions = trajectory.get_positions()  # Shape: (n_frames, n_atoms, 3)
    cells = trajectory.get_cells()          # Shape: (n_frames, 3, 3)
    n_frames, n_atoms, _ = positions.shape
    print(f"Trajectory shape: {positions.shape} (frames, atoms, xyz)")
    print(f"Number of frames: {n_frames}")
    print(f"Number of atoms: {n_atoms}")
    try:
        atom_types = trajectory.get_types()
        print(f"Retrieved atom types from trajectory: {set(atom_types)}")
    except AttributeError:
        print("Warning: Could not retrieve atom types from trajectory")
        atom_types = ['H'] * n_atoms
    
    # Use atom_types directly as inp_array (no re-mapping needed)
    inp_array = atom_types
    print(f"Element assignment complete. Sample: {inp_array[:10]}...")
    print(f"Element distribution: {dict(zip(*np.unique(inp_array, return_counts=True)))}")
    
    # Fallback logic for element assignment if necessary
    if not has_type:
        if elements and len(elements) == n_atoms:
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
            print("Warning: No elements or mapping provided. Defaulting to 'H' for all atoms.")
            inp_array = ['H'] * n_atoms

    # Convert to CPDyAna format
    pos_full = np.transpose(positions, (1, 0, 2))  # Shape: (atoms, frames, xyz)
    cell_param_full = cells.reshape(n_frames, 9)   # Flatten 3x3 to 9-element vectors
    
    # Generate time array
    dt_full = np.array([timestep] * (n_frames - 1)) if timestep else np.ones(n_frames - 1)
    t_full = np.concatenate(([0], np.cumsum(dt_full)))  # Cumulative time starting at 0
    
    # Thermodynamic data generation
    print("\n=== CREATING ENHANCED TRAJECTORY ARRAYS ===")
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
        generate_analysis_recommendations(t_full, dt_full[0] if len(dt_full) > 0 else 1.0, n_frames, inp_array)
    
    # Optional trajectory verification export
    if export_verification:
        export_verification_trajectory(positions, cells, inp_array, t_full)
    
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

# def iter_bomd_trajectory(bomd_file, elements=None):
#     """
#     Generator version of read_bomd_trajectory: yields one frame at a time as a dict,
#     including per-frame thermodynamic data (volume, etc.).
#     """
#     bohr_to_ang = 0.529177
#     with open(bomd_file, 'r') as f:
#         lines = f.readlines()
#     idx = 0
#     n_lines = len(lines)
#     # Auto-detect n_atoms
#     while idx < n_lines:
#         if lines[idx].strip().startswith("TIME"):
#             idx0 = idx
#             break
#         idx += 1
#     idx += 1  # Move to cell
#     cell_lines = lines[idx:idx+3]
#     idx += 3
#     # Count atom lines
#     atom_start = idx
#     while idx < n_lines and not lines[idx].strip().startswith("TIME"):
#         idx += 1
#     n_atoms = idx - atom_start
#     idx = 0
#     while idx < n_lines:
#         if not lines[idx].strip().startswith("TIME"):
#             idx += 1
#             continue
#         header = lines[idx].strip().split()
#         time_ps = float(header[0].replace("TIME(ps)", "")) if "TIME" in header[0] else float(header[1])
#         step_idx = int(header[1]) if "TIME" in header[0] else int(header[2])
#         energy_ry = float(header[2]) if "TIME" in header[0] else float(header[3])
#         idx += 1
#         cell = []
#         for _ in range(3):
#             cell.append([float(x) * bohr_to_ang for x in lines[idx].split()])
#             idx += 1
#         frame_pos = []
#         for _ in range(n_atoms):
#             frame_pos.append([float(x) * bohr_to_ang for x in lines[idx].split()])
#             idx += 1
#         # Assign elements
#         if elements and len(elements) == n_atoms:
#             inp_array = elements
#         else:
#             inp_array = ['H'] * n_atoms

#         cell_np = np.array(cell)
#         vol = np.linalg.det(cell_np)
#         # Add per-frame thermodynamic data (default values for temperature, pressure, etc.)
#         yield {
#             "positions": np.array(frame_pos),
#             "cell": cell_np,
#             "timestep": time_ps,
#             "energy_ry": energy_ry,
#             "symbols": inp_array,
#             "volume": vol,
#             "cell_temp": 300.0,   # Default temperature
#             "ion_temp": 300.0,    # Default temperature
#             "pressure": 0.0,      # Default pressure
#             "tot_energy": 0.0,    # Default total energy
#             "enthalpy": 0.0       # Default enthalpy
#         }

# def read_bomd_trajectory(bomd_file, elements=None, timestep=None, export_verification=False):
#     """
#     Read BOMD trajectory file (.trj, .mdtrj, .trajectory) in Quantum ESPRESSO format (no TIME header).
#     Args:
#         bomd_file (str): Path to BOMD trajectory file.
#         elements (list): List of element symbols (length = n_atoms).
#         timestep (float): Optional, time step in ps.
#         export_verification (bool): Export verification trajectory.
#     Returns:
#         tuple: (positions, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array)
#     """
#     bohr_to_ang = 0.529177
#     positions = []
#     cells = []
#     times = []
#     energies = []
#     with open(bomd_file, 'r') as f:
#         lines = [line for line in f if line.strip() != '']
#     # Detect n_atoms from first frame
#     n_lines = len(lines)
#     # Each frame: 1 header + 3 cell + N atoms
#     # Find n_atoms by reading first frame
#     header_idx = 0
#     cell_start = header_idx + 1
#     atom_start = cell_start + 3
#     # Find next header (should be at atom_start + n_atoms)
#     # Assume all frames have same number of atoms
#     # Find first header line of next frame
#     for i in range(atom_start + 1, n_lines):
#         parts = lines[i].split()
#         if len(parts) == 3:
#             try:
#                 float(parts[0]); float(parts[1]); float(parts[2])
#                 n_atoms = i - atom_start
#                 break
#             except Exception:
#                 continue
#     else:
#         # If only one frame, use all remaining lines
#         n_atoms = n_lines - atom_start
#     frame_len = 1 + 3 + n_atoms
#     n_frames = n_lines // frame_len
#     # Now parse all frames
#     for fidx in range(n_frames):
#         base = fidx * frame_len
#         header = lines[base].split()
#         time_ps = float(header[0])
#         # temp = float(header[1])  # Not used
#         energy_ry = float(header[2])
#         times.append(time_ps)
#         energies.append(energy_ry)
#         # Cell
#         cell = []
#         for i in range(3):
#             cell.append([float(x) for x in lines[base + 1 + i].split()])
#         cells.append(np.array(cell))
#         # Positions
#         frame_pos = []
#         for i in range(n_atoms):
#             frame_pos.append([float(x) for x in lines[base + 4 + i].split()])
#         positions.append(frame_pos)
#     positions = np.array(positions) * bohr_to_ang  # (frames, atoms, 3)
#     positions = np.transpose(positions, (1, 0, 2))  # (atoms, frames, 3)
#     cells = np.array(cells) * bohr_to_ang  # (frames, 3, 3)
#     cell_param_full = cells.reshape(cells.shape[0], 9)
#     n_frames = positions.shape[1]
#     dt_full = np.diff(times) if len(times) > 1 else np.ones(n_frames-1)
#     t_full = np.array(times)
#     volumes = [np.linalg.det(c) for c in cells]
#     thermo_data = {'total_energy_ry': np.array(energies)}
#     # Element assignment
#     if elements and len(elements) == n_atoms:
#         inp_array = elements
#     else:
#         inp_array = ['H'] * n_atoms
#     print(f"Element assignment for BOMD: {inp_array}")
#     print(f"Unique elements: {set(inp_array)}")
#     # Optional verification export
#     if export_verification:
#         export_verification_trajectory(positions, cells, inp_array, t_full)
#     return (positions, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array)

def read_bomd_files(bomd_in_file, bomd_mdtrj_file, Conv_factor=0.529177249):
    """
    Read BOMD input and trajectory files to form trajectory arrays.

    Args:
        bomd_in_file (str): Path to the BOMD input file (e.g., bomd.in).
        bomd_mdtrj_file (str): Path to the BOMD trajectory file (e.g., bomd.mdtrj).
        Conv_factor (float): Conversion factor from Bohr to Angstrom (default: 0.529177249).

    Returns:
        tuple: (pos_full, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array)
            - pos_full: (n_atoms, n_frames, 3) array of atomic positions in Angstrom.
            - n_frames: int, number of frames.
            - dt_full: (n_frames-1,) array of time differences in ps.
            - t_full: (n_frames,) array of times in ps.
            - cell_param_full: (n_frames, 9) array of flattened lattice vectors in Angstrom.
            - thermo_data: dict with {'total_energy_ry': (n_frames,)} containing energies in Rydberg.
            - volumes: (n_frames,) array of cell volumes in Angstrom^3.
            - inp_array: list of atom symbols in the order of the trajectory.
    """
    # Read atom symbols from bomd.in
    with open(bomd_in_file, 'r') as f:
        lines = f.readlines()
    atomic_positions = False
    inp_array = []
    for line in lines:
        line = line.strip()
        if 'ATOMIC_POSITIONS' in line:
            atomic_positions = True
            continue
        if atomic_positions and line:
            if not line[0].isalpha():
                break  # Stop at the end of ATOMIC_POSITIONS block
            inp_array.append(line.split()[0])
    n_atoms = len(inp_array)
    if n_atoms == 0:
        raise ValueError("No atoms found in bomd.in ATOMIC_POSITIONS section")

    # Read bomd.mdtrj
    with open(bomd_mdtrj_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    frame_size = 1 + 3 + n_atoms  # Header + 3 cell lines + n_atoms position lines
    if len(lines) % frame_size != 0:
        raise ValueError(f"Inconsistent number of lines in {bomd_mdtrj_file}. Expected multiple of {frame_size}, got {len(lines)}")
    
    n_frames = len(lines) // frame_size
    pos_full = []
    cell_param_full = []
    t_full = []
    energies = []

    for i in range(n_frames):
        base = i * frame_size
        
        # Parse header line: time (ps), step, energy (Ry)
        header = lines[base].split()
        if len(header) != 3:
            raise ValueError(f"Invalid header format at line {base+1} in {bomd_mdtrj_file}")
        t_full.append(float(header[0]))
        energies.append(float(header[2]))
        
        # Parse lattice vectors (3 lines)
        cell = []
        for j in range(3):
            cell_line = lines[base + 1 + j].split()
            if len(cell_line) != 3:
                raise ValueError(f"Invalid cell vector format at line {base+2+j} in {bomd_mdtrj_file}")
            cell.append([float(x) for x in cell_line])
        cell = np.array(cell) * Conv_factor  # Convert to Angstrom
        cell_param_full.append(cell.flatten())
        
        # Parse atomic positions (n_atoms lines)
        pos = []
        for j in range(n_atoms):
            pos_line = lines[base + 4 + j].split()
            if len(pos_line) != 3:
                raise ValueError(f"Invalid position format at line {base+5+j} in {bomd_mdtrj_file}")
            pos.append([float(x) for x in pos_line])
        pos = np.array(pos) * Conv_factor  # Convert to Angstrom
        pos_full.append(pos)
    
    # Convert lists to numpy arrays
    pos_full = np.array(pos_full)  # Shape: (n_frames, n_atoms, 3)
    pos_full = np.transpose(pos_full, (1, 0, 2))  # Shape: (n_atoms, n_frames, 3)
    cell_param_full = np.array(cell_param_full)  # Shape: (n_frames, 9)
    t_full = np.array(t_full)  # Shape: (n_frames,)
    energies = np.array(energies)  # Shape: (n_frames,)
    dt_full = np.diff(t_full)  # Shape: (n_frames-1,)
    volumes = np.array([np.linalg.det(cell.reshape(3, 3)) for cell in cell_param_full])  # Shape: (n_frames,)
    thermo_data = {'total_energy_ry': energies}

    return pos_full, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array

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
        ion_file (str): Path to the ion definition file (.in format).

    Returns:
        list: List of atomic symbols (e.g., ['Li', 'Al', 'P', 'S'])
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
        cel_file (str): Path to the cell parameter file (.cel format).
        Length_conversion_factor (float): Unit conversion factor (e.g., 0.529177249 for Bohr to Angstrom).

    Returns:
        np.ndarray: Array with shape (n_frames, 9) containing lattice vectors:
            [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z, a3_x, a3_y, a3_z]
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
        evp_file (str): Path to the EVP file (.evp format).
        Length_conversion_factor (float): Unit conversion factor for energy and volume.

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
        pos_file (str): Path to the position file (.pos format).
        total_ions (list): Ion information from read_ion_file().
        Length_conversion_factor (float): Unit conversion factor (Bohr to Angstrom).
        number_of_time_frames (int): Number of frames to read.
        time_difference (float): Time step between frames in ps.

    Returns:
        tuple: Contains:
            - positions (np.ndarray): Position array with shape (n_ions, n_frames, 3)
            - n_steps (int): Number of time steps
            - dt (np.ndarray): Time differences between steps
            - time (np.ndarray): Absolute time values in picoseconds
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