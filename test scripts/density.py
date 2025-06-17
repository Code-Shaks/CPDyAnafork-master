#!/usr/bin/env python3
"""
Corrected Probability Density Calculation Script
Uses the EXACT same methodology as cptotraj.py + probdensity.py but keeps data in memory
"""

import numpy as np
import argparse
from ase import Atoms
from ase.io.trajectory import Trajectory as ASE_Trajectory
from samos.trajectory import Trajectory
from samos.analysis.get_gaussian_density import get_gaussian_density
from target import input_reader as inp

def divide_chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def convert_symbols_to_atomic_numbers(element_symbols):
    """
    Convert element symbols to atomic numbers.
    
    Args:
        element_symbols (list): List of element symbols (e.g., ['Li', 'Al', 'P', 'S'])
        
    Returns:
        list: List of corresponding atomic numbers
    """
    # Mapping of element symbols to atomic numbers
    element_to_atomic_number = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103
    }
    
    # Convert symbols to atomic numbers
    atomic_numbers = []
    for symbol in element_symbols:
        if symbol in element_to_atomic_number:
            atomic_numbers.append(element_to_atomic_number[symbol])
        else:
            raise ValueError(f"Unknown element symbol: {symbol}")
    
    # Print summary using element counts
    from collections import Counter
    element_counts = Counter(element_symbols)
    print(f"Species breakdown from input file:")
    for element, count in element_counts.items():
        print(f"  {element}: {count} atoms (Z={element_to_atomic_number[element]})")
    print(f"Total atoms: {len(element_symbols)}")
    
    return atomic_numbers

def build_trajectory(in_file, pos_file, cel_file, time_after_start=60.0, num_frames=100, time_interval=0.00193511):
    """
    Build trajectory.
    Returns list of ASE Atoms objects instead of writing to file
    """

    ang = 0.529177249
    
    element_symbols = inp.read_ion_file(in_file)
    
    # Convert element symbols to atomic numbers
    sp = convert_symbols_to_atomic_numbers(element_symbols)
    
    print(f"Using hardcoded species: {len(sp)} atoms total")
    print(f"Species breakdown: 22 Li + 2 Al + 4 P + 24 S = {len(sp)} atoms")
    
    try:
        cellfile = open(cel_file).read().splitlines()
        posfile = open(pos_file).read().splitlines()
    except FileNotFoundError as e:
        print(f"Error: Could not find required files. Make sure {cel_file} and {pos_file} exist.")
        raise e
    
    lats = list(divide_chunks(cellfile, 4))
    coords = list(divide_chunks(posfile, len(sp)+1))  # This is key: len(sp)+1 = 53
    
    print(f"Found {len(lats)} lattice chunks and {len(coords)} coordinate chunks")
    
    atoms_list = []
    
    for i in range(0, len(lats)):
        try:
            tlat = np.loadtxt(lats[i], skiprows=1).transpose() * ang
             
            tpos = np.loadtxt(coords[i], skiprows=1) * ang
            
            atoms = Atoms(cell=tlat, numbers=sp, positions=tpos, pbc=True)
            atoms_list.append(atoms)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} frames...")
                
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue
    
    print(f"Successfully built {len(atoms_list)} frames in memory")
    return atoms_list

def calculate_probability_density(atoms_list, element='Li', sigma=0.5, n_sigma=3.0, density=0.1, outputfile=None):
    """
    Calculate probability density from in-memory atoms list
    Same as probdensity.py but using atoms_list instead of loading .traj file
    """
    
    if not atoms_list:
        raise ValueError("No atoms found in trajectory")
    
    print(f"Converting {len(atoms_list)} ASE frames to SAMOS trajectory...")
    
    traj = Trajectory.from_atoms(atoms_list)

    params = {
        'element': element,
        'sigma': sigma,
        'n_sigma': n_sigma,
        'density': density,
        'outputfile': '{}_Density.xsf'.format(element)
    }
    
    print(f"Calculating {element} density with parameters: {params}")
    get_gaussian_density(traj, **params)
    
    print(f"Probability density calculation complete. Output: {params['outputfile']}")
    return params['outputfile']

def main():
    parser = argparse.ArgumentParser(description='Calculate probability density from QE files')
    parser.add_argument('--in_file', required=True, help='Input .in file')
    parser.add_argument('--pos_file', required=True, help='Position .pos file')
    parser.add_argument('--cel_file', required=True, help='Cell .cel file')
    parser.add_argument('--time_after_start', type=float, default=60.0, help='Time after start (ps) (default: 60.0)')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames (default: 100)')
    parser.add_argument('--time_interval', type=float, default=0.00193511, help='Time interval (default: 0.00193511)')
    parser.add_argument('--element', default='Li', help='Element for density calculation (default: Li)')
    parser.add_argument('--sigma', type=float, default=0.5, help='Gaussian sigma (default: 0.5)')
    parser.add_argument('--n_sigma', type=float, default=3.0, help='Number of sigma for cutoff (default: 3.0)')
    parser.add_argument('--density', type=float, default=1.0, help='Grid density (default: 1.0)')
    parser.add_argument('--output', default=None, help='Output filename (default: None)')
    
    args = parser.parse_args()
    
    try:
        print("=== Building trajectory in memory using cptotraj.py method ===")
        atoms_list = build_trajectory(
            in_file=args.in_file,
            pos_file=args.pos_file,
            cel_file=args.cel_file,
            time_after_start=args.time_after_start,
            num_frames=args.num_frames,
            time_interval=args.time_interval
        )
        
        print("\n=== Calculating probability density using probdensity.py method ===")
        output_file = calculate_probability_density(
            atoms_list,
            element=args.element,
            sigma=args.sigma,
            n_sigma=args.n_sigma,
            density=args.density,
            outputfile=args.output
        )
        
        print(f"\n=== SUCCESS: Output saved to {output_file} ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
