#!/usr/bin/env python3

"""
Complete RDF Calculation Script for Quantum ESPRESSO Output Files

This module provides functionality to compute Radial Distribution Functions (RDFs)
from Quantum ESPRESSO molecular dynamics simulation output files. It processes
.in, .pos, and .cel files to extract atomic trajectories and calculate pair
distribution functions using pymatgen's efficient RDF implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunctionFast
from . import input_reader as inp
import math
import os
import argparse
import sys

def build_ase_trajectory(in_file, pos_file, cel_file, time_after_start=60, num_frames=100, time_interval=0.00193511):
    """
    Build ASE trajectory frames from Quantum ESPRESSO output files.
    
    This function combines atomic species information from .in file, positions
    from .pos file, and cell parameters from .cel file to create ASE Atoms objects
    for each time step.
    
    Args:
        in_file (str): Path to the .in input file
        pos_file (str): Path to the .pos position file
        cel_file (str): Path to the .cel cell parameter file
        time_after_start (float): Time in ps after which to start extracting frames (default: 60)
        num_frames (int): Number of frames to extract (default: 100)
        time_interval (float): Time interval between frames in ps (default: 0.00193511)
        
    Returns:
        list: List of ASE Atoms objects representing the trajectory frames
    """
    ang = 0.529177249  # Bohr to Angstrom conversion factor
    
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
    
    print(f"Extracting frames {starting_frame_number} to {ending_frame_number-1}")
    
    # Build ASE Atoms objects for each frame
    frames = []
    for i in range(starting_frame_number, min(ending_frame_number, min_frames)):
        tlat = cell_matrices[i].transpose()  # Transpose for ASE format
        tpos = pos_data[:, i, :]  # Positions at time step i
        
        # Create ASE Atoms object with periodic boundary conditions
        atoms = Atoms(cell=tlat, numbers=atomic_numbers, positions=tpos, pbc=True)
        frames.append(atoms)
        
    print(f"Successfully built {len(frames)} ASE frames in memory")
    return frames

def compute_rdf(trajectory, output_prefix='rdf_plot', time_after_start=60):
    """
    Compute RDF with proper validation and improved visualization.
    
    This function calculates Radial Distribution Functions for various atom pairs
    using pymatgen's efficient RDF implementation and creates publication-quality plots.
    
    Args:
        trajectory (list): List of ASE Atoms objects representing the trajectory
        output_prefix (str): Prefix for output plot filenames (default: 'rdf_plot')
        time_after_start (float): Time after start for filename generation (default: 60)
        
    Returns:
        tuple: Contains r (distance array) and RDF arrays for different atom pairs
               (r, rdf_li, rdf_al, rdf_p, rdf_s)
    """
    print("Converting to pymatgen structures with PBC validation...")
    adaptor = AseAtomsAdaptor()
    
    # Convert ASE to pymatgen with validation
    structures = []
    for i, frame in enumerate(trajectory):
        try:
            structure = adaptor.get_structure(frame)
            # Validate lattice volume to ensure proper cell definition
            if structure.lattice.volume < 1.0:
                raise ValueError(f"Invalid lattice volume in frame {i}")
            structures.append(structure)
        except Exception as e:
            print(f"Error converting frame {i}: {e}")
            continue
    
    print(f"Successfully converted {len(structures)} structures")
    
    # Initialize RDF calculator with optimized parameters
    try:
        rdf_func = RadialDistributionFunctionFast(
            structures=structures,
            ngrid=1001,        # Number of radial grid points
            rmax=10.0,         # Maximum distance for RDF calculation (Å)
            sigma=0.2          # Gaussian broadening parameter
        )
    except Exception as e:
        print(f"Error initializing RDF function: {e}")
        return None
    
    print("Computing RDF with proper distance calculations...")
    
    # Calculate RDFs for different central atoms
    central_atoms = ['Li', 'Al', 'P', 'S']
    for central in central_atoms:
        try:
            # Compute pair distribution functions for central atom with all species
            r, rdf_li = rdf_func.get_rdf(central, ['Li'])
            r, rdf_al = rdf_func.get_rdf(central, ['Al'])
            r, rdf_p = rdf_func.get_rdf(central, ['P'])
            r, rdf_s = rdf_func.get_rdf(central, ['S'])
            
            # Validate RDF results for NaN values
            if np.any(np.isnan(rdf_li)) or np.any(np.isnan(rdf_al)):
                print(f"Warning: NaN values detected in RDF for {central}")
                continue
            
            # Create publication-quality plot
            plt.figure(figsize=(12, 8))
            plt.xlabel('Distance (Å)', fontsize=12)
            plt.ylabel('g(r)', fontsize=12)
            
            # Plot RDFs with improved styling
            plt.plot(r, rdf_li, label=f'{central}-Li', linewidth=2)
            plt.plot(r, rdf_al, label=f'{central}-Al', linewidth=2)
            plt.plot(r, rdf_p, label=f'{central}-P', linewidth=2)
            plt.plot(r, rdf_s, label=f'{central}-S', linewidth=2)
            
            # Dynamic y-axis limits based on data range
            max_rdf = max(np.max(rdf_li), np.max(rdf_al), np.max(rdf_p), np.max(rdf_s))
            plt.ylim(0, max_rdf * 1.15)  # Add 15% headroom above highest peak
            
            # Set x-axis range to focus on meaningful RDF region
            plt.xlim(1.5, 8.0)
            
            # Enhanced plot formatting
            plt.legend(loc='upper right', fontsize=10, frameon=True, 
                      fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout(pad=2.0)
            
            # Generate descriptive output filename
            output_filename = f'{output_prefix}_LiAlPS_600_{central}_time{time_after_start}ps_variable_cell.png'
            plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"RDF plot saved as: {output_filename}")
            
        except Exception as e:
            print(f"Error computing RDF for {central}: {e}")
            continue
    
    print("RDF calculation complete with improved visualization")
    return r, rdf_li, rdf_al, rdf_p, rdf_s


def main():
    """
    Main function with integrated .traj conversion approach.
    
    This function handles command-line argument parsing, file validation,
    and orchestrates the complete RDF calculation workflow from Quantum
    ESPRESSO output files.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Compute RDF from QE files via .traj conversion')
    parser.add_argument('--in_file', required=True, help='Path to .in file')
    parser.add_argument('--pos_file', required=True, help='Path to .pos file')
    parser.add_argument('--cel_file', required=True, help='Path to .cel file')
    parser.add_argument('--time_after_start', type=float, default=60.0,
                        help='Time (ps) after which to start extracting frames')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='Number of frames to extract for RDF')
    parser.add_argument('--time_interval', type=float, default=0.00193511,
                        help='Time interval between frames (ps)')
    parser.add_argument('--output_prefix', default='rdf_plot',
                        help='Prefix for output plot filename')
    
    args = parser.parse_args()

    # Validate that all required input files exist
    for file_path in [args.in_file, args.pos_file, args.cel_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    # Display calculation parameters
    print("=== Quantum ESPRESSO RDF Calculation via .traj Conversion ===")
    print(f"Input files:")
    print(f" .in file: {args.in_file}")
    print(f" .pos file: {args.pos_file}")
    print(f" .cel file: {args.cel_file}")
    print(f"Parameters:")
    print(f" Time after start: {args.time_after_start} ps")
    print(f" Number of frames: {args.num_frames}")
    print()

    try:
        # Step 1 & 2: Build trajectory frames in memory
        print("Step 1 & 2: Building ASE trajectory frames in memory...")
        extracted_frames = build_ase_trajectory(
            args.in_file, args.pos_file, args.cel_file,
            time_after_start=args.time_after_start,
            num_frames=args.num_frames,
            time_interval=args.time_interval
        )
        
        # Step 3: Compute RDF and generate plots
        print("Step 3: Computing RDF using trajectory methodology...")
        r, rdf_li, rdf_al, rdf_p, rdf_s = compute_rdf(
            extracted_frames,
            output_prefix=args.output_prefix,
            time_after_start=args.time_after_start
        )
        print("\n=== RDF Calculation Complete ===")
        
    except Exception as e:
        print(f"Error during calculation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Display usage information if no arguments provided
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python compute_rdf_final.py --in_file LiAlPS.in --pos_file LiAlPS.pos --cel_file LiAlPS.cel")
        print("\nFor help with all options:")
        print("python compute_rdf_final.py --help")
        sys.exit(0)
    main()
