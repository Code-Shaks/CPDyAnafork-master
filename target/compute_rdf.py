#!/usr/bin/env python3

"""
compute_rdf.py

Compute Radial Distribution Functions (RDFs) from Quantum ESPRESSO or LAMMPS molecular dynamics output files.
This script builds atomic trajectories using ASE, converts them to pymatgen structures, and calculates RDFs for various atom pairs using pymatgen's efficient implementation.

Flexible CLI options allow user control over RDF parameters and plotting.

Usage:
    python compute_rdf.py --in-file LiAlPS.in --pos-file LiAlPS.pos --cel-file LiAlPS.cel
    # For more options:
    python compute_rdf.py --help

Author: CPDyAna Development Team
Version: 2025-06-25
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunctionFast
from . import input_reader as inp
import math
import os
import argparse
import sys

def build_lammps_trajectory(
    lammps_file, lammps_elements, element_mapping, lammps_timestep,
    num_frames=100, stride=1
):
    """
    Build ASE trajectory frames from a LAMMPS dump file using CPDyAna input_reader.

    Args:
        lammps_file (str): Path to LAMMPS trajectory file.
        lammps_elements (list): List of atom-type symbols.
        element_mapping (dict): Mapping from atom type index to element symbol.
        lammps_timestep (float): Timestep in ps.
        num_frames (int): Number of frames to extract.
        stride (int): Stride for frame extraction.

    Returns:
        list: List of ASE Atoms objects representing the trajectory frames.
    """
    # Use input_reader.read_lammps_trajectory to get trajectory arrays and atom types
    pos_full, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array = inp.read_lammps_trajectory(
        lammps_file,
        elements=lammps_elements,
        timestep=lammps_timestep,
        element_mapping=element_mapping,
        export_verification=False,
        show_recommendations=False
    )

    # Limit frames if requested
    if num_frames > 0:
        frame_indices = list(range(0, min(num_frames, n_frames), stride))
    else:
        frame_indices = list(range(0, n_frames, stride))

    atoms_list = []
    for i in frame_indices:
        symbols = [str(s) for s in inp_array]
        positions = pos_full[:, i, :]
        cell = cell_param_full[i].reshape((3, 3))
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms_list.append(atoms)

    print(f"Built {len(atoms_list)} ASE Atoms frames from LAMMPS trajectory.")
    return atoms_list

def build_ase_trajectory(
    in_file, pos_file, cel_file, time_after_start=60, num_frames=100, time_interval=0.00193511
):
    """
    Build ASE trajectory frames from Quantum ESPRESSO output files.

    Args:
        in_file (str): Path to the .in input file.
        pos_file (str): Path to the .pos position file.
        cel_file (str): Path to the .cel cell parameter file.
        time_after_start (float): Time in ps after which to start extracting frames.
        num_frames (int): Number of frames to extract.
        time_interval (float): Time interval between frames in ps.

    Returns:
        list: List of ASE Atoms objects representing the trajectory frames.
    """
    ang = 0.529177249  # Bohr to Angstrom conversion

    # Read atomic species from input file
    symbols = inp.read_ion_file(in_file)

    # Dictionary for converting element symbols to atomic numbers
    symbol_to_number = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36
    }
    atomic_numbers = [symbol_to_number.get(symbol, 1) for symbol in symbols]

    # Read cell parameters and positions with unit conversion
    cell_data = inp.read_cel_file(cel_file, ang)
    cell_matrices = cell_data.reshape(-1, 3, 3)  # Reshape to (n_steps, 3, 3)

    pos_data, n_steps, dt, time_array = inp.read_pos_file(pos_file, symbols, ang, len(cell_matrices), time_interval)

    # Determine frame range to extract
    min_frames = min(len(cell_matrices), pos_data.shape[1])
    starting_frame_number = math.ceil(time_after_start / time_interval)
    ending_frame_number = starting_frame_number + num_frames

    # Build ASE Atoms objects for each frame
    frames = []
    for i in range(starting_frame_number, min(ending_frame_number, min_frames)):
        tlat = cell_matrices[i].transpose()  # Transpose for ASE format
        tpos = pos_data[:, i, :]  # Positions at time step i

        # Create ASE Atoms object with periodic boundary conditions
        atoms = Atoms(cell=tlat, numbers=atomic_numbers, positions=tpos, pbc=True)
        frames.append(atoms)

    return frames

def build_bomd_trajectory(
    trj_file, in_file=None, num_frames=0, stride=1
):
    """
    Build ASE trajectory frames from BOMD (.mdtrj) output files using input_reader.read_bomd_files.

    Args:
        trj_file (str): Path to the BOMD .mdtrj trajectory file.
        in_file (str): Path to the BOMD .in input file (for atom order).
        num_frames (int): Number of frames to extract (0 for all).
        stride (int): Stride for frame extraction.

    Returns:
        list: List of ASE Atoms objects representing the trajectory frames.
    """
    import os
    from ase import Atoms
    from . import input_reader as inp
    import numpy as np

    # Auto-detect .in file if not provided
    if in_file is None:
        trj_dir = os.path.dirname(trj_file)
        in_files = [f for f in os.listdir(trj_dir) if f.endswith('.in')]
        if not in_files:
            raise FileNotFoundError("No BOMD .in file found in the trajectory directory.")
        in_file = os.path.join(trj_dir, in_files[0])

    # Read BOMD trajectory and atom order
    pos_full, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array = inp.read_bomd_files(
        in_file, trj_file
    )

    # Determine frame indices
    if num_frames > 0:
        frame_indices = list(range(0, min(num_frames, n_frames), stride))
    else:
        frame_indices = list(range(0, n_frames, stride))

    # Prepare positions and cells
    pos_arr = np.transpose(pos_full, (1, 0, 2))  # (frames, atoms, 3)
    cell_arr = cell_param_full.reshape(-1, 3, 3)  # (frames, 3, 3)
    atoms_list = []
    for i in frame_indices:
        atoms = Atoms(symbols=inp_array, positions=pos_arr[i], cell=cell_arr[i], pbc=True)
        atoms_list.append(atoms)
    print(f"BOMD trajectory built: {len(atoms_list)} frames (stride={stride})")
    return atoms_list

def compute_rdf(
    trajectory,
    output_prefix='rdf_plot',
    time_after_start=60,
    central_atoms=None,
    pair_atoms=None,
    ngrid=1001,
    rmax=10.0,
    sigma=0.2,
    xlim=(1.5, 8.0)
):
    """
    Compute RDF for various atom pairs and generate plots with RDF integrals.

    Args:
        trajectory (list): List of ASE Atoms objects representing the trajectory.
        output_prefix (str): Prefix for output plot filenames.
        time_after_start (float): Time after start for filename generation.
        central_atoms (list): List of central atom types (default: ['Li', 'Al', 'P', 'S']).
        pair_atoms (list): List of pair atom types (default: same as central_atoms).
        ngrid (int): Number of radial grid points.
        rmax (float): Maximum distance for RDF calculation (Å).
        sigma (float): Gaussian broadening parameter.
        xlim (tuple): x-axis limits for plot.

    Returns:
        dict: Contains r (distance array) and RDF arrays for different atom pairs.
    """
    adaptor = AseAtomsAdaptor()
    structures = []
    for i, frame in enumerate(trajectory):
        try:
            structure = adaptor.get_structure(frame)
            if structure.lattice.volume < 1.0:
                continue
            structures.append(structure)
        except Exception:
            continue

    if not structures:
        print("No valid structures for RDF calculation.")
        return None

    try:
        rdf_func = RadialDistributionFunctionFast(
            structures=structures,
            ngrid=ngrid,
            rmax=rmax,
            sigma=sigma
        )
    except Exception as e:
        print(f"Error initializing RDF function: {e}")
        return None

    # Set up atom types
    if central_atoms is None:
        central_atoms = ['Li', 'Al', 'P', 'S']
    if pair_atoms is None:
        pair_atoms = central_atoms

    results = {}
    for central in central_atoms:
        try:
            plt.figure(figsize=(12, 8))
            ax1 = plt.gca()
            ax2 = ax1.twinx()  # Create a second y-axis for RDF integrals
            
            ax1.set_xlabel('Distance (Å)', fontsize=12)
            ax1.set_ylabel('g(r)', fontsize=12, color='black')
            ax2.set_ylabel('∫ 4πr²g(r)dr', fontsize=12, color='black')
            
            max_rdf = 0
            max_integral = 0
            min_significant_x = float('inf')
            max_significant_x = 0
            
            linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Cycle through these for different pairs
            colors = plt.cm.tab10.colors
            
            for i, pair in enumerate(pair_atoms):
                r, rdf_val = rdf_func.get_rdf(central, [pair])
                if np.any(np.isnan(rdf_val)):
                    continue
                
                # Calculate RDF integral: n(r) = ∫ 4πr²ρg(r)dr
                # Since we already have normalized RDF, we calculate cumulative integral
                dr = r[1] - r[0]
                # Get volume element 4πr²dr and calculate integral
                rdf_integral = np.cumsum(4 * np.pi * r**2 * rdf_val * dr)
                
                # Find significant range of the data (where RDF > 1% of max)
                threshold = np.max(rdf_val) * 0.01
                significant_indices = np.where(rdf_val > threshold)[0]
                if len(significant_indices) > 0:
                    min_significant_x = min(min_significant_x, r[significant_indices[0]])
                    max_significant_x = max(max_significant_x, r[significant_indices[-1]])
                
                # Plot RDF on left axis
                color = colors[i % len(colors)]
                line1, = ax1.plot(r, rdf_val, 
                                 label=f'{central}-{pair} g(r)', 
                                 linewidth=2, 
                                 color=color)
                
                # Plot RDF integral on right axis with dashed line
                line2, = ax2.plot(r, rdf_integral, 
                                 label=f'{central}-{pair} integral', 
                                 linewidth=1.5, 
                                 linestyle='--',
                                 color=color)
                
                max_rdf = max(max_rdf, np.max(rdf_val))
                max_integral = max(max_integral, np.max(rdf_integral))
                results[(central, pair)] = (r, rdf_val, rdf_integral)
            
            if max_rdf == 0:
                plt.close()
                continue
                
            # Set y-axis limits
            ax1.set_ylim(0, max_rdf * 1.2)
            ax2.set_ylim(0, max_integral * 1.2)
            
            # Set dynamic x-axis limits based on significant data range
            if min_significant_x != float('inf') and max_significant_x > 0:
                # Add some padding to the limits (20% extra space)
                padding = (max_significant_x - min_significant_x) * 0.2
                x_min = max(0, min_significant_x - padding)
                x_max = min(rmax, max_significant_x + padding)
                ax1.set_xlim(x_min, x_max)
            else:
                # Fallback to user-provided xlim
                ax1.set_xlim(*xlim)
            
            # Create combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      loc='upper right', fontsize=10, frameon=True, 
                      fancybox=True, shadow=True)
            
            ax1.grid(True, alpha=0.3)
            plt.tight_layout(pad=2.0)
            output_filename = f'{output_prefix}_{central}_time{time_after_start}ps_with_integral.png'
            plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created RDF plot with integral for {central} at {output_filename}")
        except Exception as e:
            print(f"Error computing RDF for {central}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results

def main():
    """
    Command-line interface for computing RDF from QE or LAMMPS files.

    Parses arguments, builds trajectories, computes RDFs, and generates plots.
    """
    parser = argparse.ArgumentParser(description='Compute RDF from QE or LAMMPS files')
    # QE arguments
    parser.add_argument('--in-file', help='Path to .in file')
    parser.add_argument('--pos-file', help='Path to .pos file')
    parser.add_argument('--cel-file', help='Path to .cel file')
    parser.add_argument('--time-after-start', type=float, default=60.0,
                        help='Time (ps) after which to start extracting frames')
    parser.add_argument('--num-frames', type=int, default=0,
                        help='Number of frames to extract for RDF (default: 0 for all frames)')
    parser.add_argument('--time-interval', type=float, default=0.00193511,
                        help='Time interval between frames (ps)')
    # LAMMPS-specific arguments
    parser.add_argument('--lammps-file', help='LAMMPS trajectory file (.lammpstrj)')
    parser.add_argument('--lammps-elements', nargs='+', help='LAMMPS atom-type symbols (e.g., Li La Ti O)')
    parser.add_argument('--element-mapping', nargs='+', help='LAMMPS type-to-element map (e.g., 1:Li 2:La 3:Ti 4:O)')
    parser.add_argument('--lammps-timestep', type=float, help='LAMMPS timestep (ps)')
    parser.add_argument('--stride', type=int, default=1, help='Stride for reading frames (LAMMPS only)')
    # BOMD-specific arguments
    parser.add_argument('--bomd-trj', help='BOMD trajectory file (.trj)')
    # parser.add_argument('--bomd-cell', help='BOMD cell file (optional, text file with 3x3 cell per frame)')
    parser.add_argument('--bomd-in', help='BOMD input file (.in) for atom order')
    # Common RDF arguments
    parser.add_argument('--output-prefix', default='rdf_plot', help='Prefix for output plot filename')
    parser.add_argument('--central-atoms', nargs='+', default=['Li'],
                        help='List of central atom types for RDF')
    parser.add_argument('--pair-atoms', nargs='+', default=None, help='List of pair atom types for RDF')
    parser.add_argument('--ngrid', type=int, default=1001, help='Number of radial grid points for RDF')
    parser.add_argument('--rmax', type=float, default=10.0, help='Maximum distance for RDF calculation (Å)')
    parser.add_argument('--sigma', type=float, default=0.2, help='Gaussian broadening parameter for RDF')
    parser.add_argument('--xlim', nargs=2, type=float, default=[1.5, 8.0], help='x-axis limits for RDF plot')
    args = parser.parse_args()

    # --- LAMMPS branch ---
    if args.lammps_file:
        # Parse element mapping if provided
        element_mapping = None
        if args.element_mapping:
            element_mapping = {}
            for mapping in args.element_mapping:
                k, v = mapping.split(':')
                element_mapping[int(k)] = v

        # Validate required LAMMPS arguments
        if not args.lammps_elements or not args.lammps_timestep:
            print("Error: --lammps-elements and --lammps-timestep are required for LAMMPS input.")
            sys.exit(1)

        # Build trajectory from LAMMPS dump
        frames = build_lammps_trajectory(
            args.lammps_file,
            lammps_elements=args.lammps_elements,
            element_mapping=element_mapping,
            lammps_timestep=args.lammps_timestep,
            num_frames=args.num_frames,
            stride=args.stride
        )

        # Compute RDF and plot
        compute_rdf(
            frames,
            output_prefix=args.output_prefix,
            time_after_start=args.time_after_start,
            central_atoms=args.central_atoms,
            pair_atoms=args.pair_atoms if args.pair_atoms is not None else args.central_atoms,
            ngrid=args.ngrid,
            rmax=args.rmax,
            sigma=args.sigma,
            xlim=tuple(args.xlim)
        )
        print("\n=== RDF Calculation Complete (LAMMPS) ===")
        return
    
    #  --- BOMD branch ---
    if args.bomd_trj:
        import os
        if not os.path.exists(args.bomd_trj):
            print(f"Error: BOMD .mdtrj file not found: {args.bomd_trj}")
            sys.exit(1)
        # Try to find .in file if not provided
        in_file = args.bomd_in
        if not in_file:
            trj_dir = os.path.dirname(args.bomd_trj)
            in_files = [f for f in os.listdir(trj_dir) if f.endswith('.in')]
            if not in_files:
                print("Error: No BOMD .in file found in the trajectory directory.")
                sys.exit(1)
            in_file = os.path.join(trj_dir, in_files[0])

        frames = build_bomd_trajectory(
            args.bomd_trj,
            in_file=in_file,
            num_frames=args.num_frames,
            stride=args.stride
        )

        compute_rdf(
            frames,
            output_prefix=args.output_prefix,
            time_after_start=args.time_after_start,
            central_atoms=args.central_atoms,
            pair_atoms=args.pair_atoms if args.pair_atoms is not None else args.central_atoms,
            ngrid=args.ngrid,
            rmax=args.rmax,
            sigma=args.sigma,
            xlim=tuple(args.xlim)
        )
        print("\n=== RDF Calculation Complete (BOMD) ===")
        return

    # --- QE branch ---
    if args.in_file and args.pos_file and args.cel_file:
        # Validate that all required input files exist
        for file_path in [args.in_file, args.pos_file, args.cel_file]:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
        try:
            # Step 1 & 2: Build trajectory frames in memory
            extracted_frames = build_ase_trajectory(
                args.in_file, args.pos_file, args.cel_file,
                time_after_start=args.time_after_start,
                num_frames=args.num_frames,
                time_interval=args.time_interval
            )
            # Step 3: Compute RDF and generate plots
            compute_rdf(
                extracted_frames,
                output_prefix=args.output_prefix,
                time_after_start=args.time_after_start,
                central_atoms=args.central_atoms,
                pair_atoms=args.pair_atoms if args.pair_atoms is not None else args.central_atoms,
                ngrid=args.ngrid,
                rmax=args.rmax,
                sigma=args.sigma,
                xlim=tuple(args.xlim)
            )
            print("\n=== RDF Calculation Complete (QE) ===")
        except Exception as e:
            print(f"Error during calculation: {e}")
            sys.exit(1)
        return

    print("Error: Please provide either QE or LAMMPS input files.")
    sys.exit(1)

if __name__ == "__main__":
    main()