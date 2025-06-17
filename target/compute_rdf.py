#!/usr/bin/env python3

"""
compute_rdf.py

Compute Radial Distribution Functions (RDFs) from Quantum ESPRESSO molecular dynamics output files (.in, .pos, .cel).
This script builds atomic trajectories using ASE, converts them to pymatgen structures, and calculates RDFs for various atom pairs using pymatgen's efficient implementation.

Flexible CLI options allow user control over RDF parameters and plotting.

Usage:
    python compute_rdf.py --in-file LiAlPS.in --pos-file LiAlPS.pos --cel-file LiAlPS.cel
    # For more options:
    python compute_rdf.py --help
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunctionFast
from . import input_reader as inp
import math
import os
import argparse
import sys

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
        bohr_to_angstrom (float): Conversion factor from Bohr to Angstrom.

    Returns:
        list: List of ASE Atoms objects representing the trajectory frames.
    """
    ang = 0.529177249

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
    Compute RDF for various atom pairs and generate plots.

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
        ylim_factor (float): Factor to multiply max RDF for y-axis limit.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        tuple: Contains r (distance array) and RDF arrays for different atom pairs.
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
            plt.xlabel('Distance (Å)', fontsize=12)
            plt.ylabel('g(r)', fontsize=12)
            max_rdf = 0
            for pair in pair_atoms:
                r, rdf_val = rdf_func.get_rdf(central, [pair])
                if np.any(np.isnan(rdf_val)):
                    continue
                plt.plot(r, rdf_val, label=f'{central}-{pair}', linewidth=2)
                max_rdf = max(max_rdf, np.max(rdf_val))
                results[(central, pair)] = (r, rdf_val)
            if max_rdf == 0:
                plt.close()
                continue
            plt.ylim(0, max_rdf * 1.5)
            plt.xlim(*xlim)
            plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout(pad=2.0)
            output_filename = f'{output_prefix}_LiAlPS_{central}_time{time_after_start}ps_variable_cell.png'
            plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error computing RDF for {central}: {e}")
            continue

    return results

def main():
    """
    Main function with integrated .traj conversion approach.
    Handles command-line argument parsing, file validation,
    and orchestrates the complete RDF calculation workflow from Quantum
    ESPRESSO output files.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Compute RDF from QE files via .traj conversion')
    parser.add_argument('--in-file', required=True, help='Path to .in file')
    parser.add_argument('--pos-file', required=True, help='Path to .pos file')
    parser.add_argument('--cel-file', required=True, help='Path to .cel file')
    parser.add_argument('--time-after-start', type=float, default=60.0,
                        help='Time (ps) after which to start extracting frames')
    parser.add_argument('--num-frames', type=int, default=100,
                        help='Number of frames to extract for RDF')
    parser.add_argument('--time-interval', type=float, default=0.00193511,
                        help='Time interval between frames (ps)')
    parser.add_argument('--output-prefix', default='rdf_plot',
                        help='Prefix for output plot filename')
    parser.add_argument('--central-atoms', nargs='+', default=['Li', 'Al', 'P', 'S'],
                        help='List of central atom types for RDF (default: Li Al P S)')
    parser.add_argument('--pair-atoms', nargs='+', default=None,
                        help='List of pair atom types for RDF (default: same as central-atoms)')
    parser.add_argument('--ngrid', type=int, default=1001,
                        help='Number of radial grid points for RDF')
    parser.add_argument('--rmax', type=float, default=10.0,
                        help='Maximum distance for RDF calculation (Å)')
    parser.add_argument('--sigma', type=float, default=0.2,
                        help='Gaussian broadening parameter for RDF')
    parser.add_argument('--xlim', nargs=2, type=float, default=[1.5, 8.0],
                        help='x-axis limits for RDF plot (default: 1.5 8.0)')

    args = parser.parse_args()

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
        print("\n=== RDF Calculation Complete ===")

    except Exception as e:
        print(f"Error during calculation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Display usage information if no arguments provided
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python compute_rdf.py --in-file LiAlPS.in --pos-file LiAlPS.pos --cel-file LiAlPS.cel")
        print("\nFor help with all options:")
        print("python compute_rdf.py --help")
        sys.exit(0)
    main()
