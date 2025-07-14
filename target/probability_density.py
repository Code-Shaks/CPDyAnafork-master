#!/usr/bin/env python3
"""
Probability Density Calculation Module for CPDyAna
=================================================

This module provides functions to compute the 3D probability density of ions
from molecular dynamics trajectories. It supports Quantum ESPRESSO (.in/.pos/.cel),
LAMMPS (.lammpstrj), and BOMD (.trj) trajectory formats as parsed by CPDyAna's input_reader.

Features:
- Gaussian density estimation for selected elements
- Atom masking, recentering, and custom bounding box support
- Compatible with QE, LAMMPS, and BOMD workflows

Usage (CLI):
    python probability_density.py --in-file LiAlPS.in --pos-file LiAlPS.pos --cel-file LiAlPS.cel --element Li
    python probability_density.py --lammps-file traj.lammpstrj --element Li
    python probability_density.py --bomd-trj traj.trj --bomd-elements Li O Ti --element Li

Author: CPDyAna Development Team
Version: 2025-07-09
"""

import numpy as np
import argparse
from ase import Atoms
from ase.io import read as ase_read
from target.trajectory import Trajectory
from target.analysis import get_gaussian_density
from target import input_reader as inp

def divide_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def convert_symbols_to_atomic_numbers(element_symbols):
    """
    Convert element symbols to atomic numbers.

    Args:
        element_symbols (list): List of element symbols (str).

    Returns:
        list: List of atomic numbers (int).
    """
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
    atomic_numbers = []
    for symbol in element_symbols:
        if symbol in element_to_atomic_number:
            atomic_numbers.append(element_to_atomic_number[symbol])
        else:
            raise ValueError(f"Unknown element symbol: {symbol}")
    return atomic_numbers

def build_trajectory(
    in_file, pos_file, cel_file, time_after_start=0.0, num_frames=None, time_interval=0.00193511, stride=1
):
    """
    Build an in-memory trajectory from .in, .pos, .cel files (Quantum ESPRESSO).
    Returns a list of ASE Atoms objects.
    Supports frame stride for subsampling.

    Args:
        in_file (str): Path to .in file (species).
        pos_file (str): Path to .pos file (positions).
        cel_file (str): Path to .cel file (cell).
        time_after_start (float): Time (ps) after which to start.
        num_frames (int): Number of frames to extract (None or 0 for all).
        time_interval (float): Time between frames (ps).
        stride (int): Stride for frame extraction.

    Returns:
        list: List of ASE Atoms objects.
    """
    ang = 0.529177249  # Bohr to Angstrom conversion factor

    # 1. Read atomic species from .in file
    element_symbols = inp.read_ion_file(in_file)
    sp = convert_symbols_to_atomic_numbers(element_symbols)

    # 2. Read cell and position files
    cellfile = open(cel_file).read().splitlines()
    posfile = open(pos_file).read().splitlines()
    lats = list(divide_chunks(cellfile, 4))
    coords = list(divide_chunks(posfile, len(sp)+1))

    total_frames = min(len(lats), len(coords))
    start_frame = int(time_after_start / time_interval)
    if num_frames is None or num_frames == 0:
        end_frame = total_frames
    else:
        end_frame = min(start_frame + num_frames, total_frames)

    atoms_list = []
    for i in range(start_frame, end_frame, stride):
        tlat = np.loadtxt(lats[i], skiprows=1).transpose() * ang
        tpos = np.loadtxt(coords[i], skiprows=1) * ang
        atoms = Atoms(cell=tlat, numbers=sp, positions=tpos, pbc=True)
        atoms_list.append(atoms)
    print(f"Trajectory built: {len(atoms_list)} frames from {start_frame} to {end_frame-1} (stride={stride})")
    return atoms_list

def build_lammps_trajectory(lammps_file, stride=1, max_frames=None):
    """
    Build a trajectory from a LAMMPS .lammpstrj file using ASE.
    Returns a list of ASE Atoms objects.

    Args:
        lammps_file (str): Path to LAMMPS trajectory file.
        stride (int): Stride for frame extraction.
        max_frames (int): Maximum number of frames to extract.

    Returns:
        list: List of ASE Atoms objects.
    """
    all_frames = ase_read(lammps_file, index=":", format="lammps-dump-text")
    if max_frames is not None:
        all_frames = all_frames[:max_frames]
    atoms_list = [frame for idx, frame in enumerate(all_frames) if idx % stride == 0]
    print(f"LAMMPS trajectory built: {len(atoms_list)} frames (stride={stride})")
    return atoms_list

def build_bomd_trajectory(trj_file, in_file=None, elements=None, num_frames=0, stride=1):
    """
    Build a trajectory from a BOMD .mdtrj file using CPDyAna's input_reader.

    Args:
        trj_file (str): Path to BOMD .mdtrj trajectory file.
        in_file (str): Path to BOMD .in input file (for atom order).
        elements (list): List of element symbols (order must match .mdtrj, optional).
        num_frames (int): Number of frames to extract (0 for all).
        stride (int): Stride for frame extraction.

    Returns:
        list: List of ASE Atoms objects.
    """
    import os

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

def parse_mask(mask_str, total_atoms):
    """
    Parse a mask string like '0,1,2,5-10' into a list of atom indices.

    Args:
        mask_str (str): Mask string.
        total_atoms (int): Total number of atoms.

    Returns:
        list: List of atom indices.
    """
    if not mask_str:
        return None
    indices = set()
    for part in mask_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.update(range(start, end+1))
        else:
            indices.add(int(part))
    # Ensure indices are within bounds
    indices = [i for i in indices if 0 <= i < total_atoms]
    return indices

def calculate_probability_density(
    atoms_list, element='Li', sigma=0.3, n_sigma=3.0, density=0.1, outputfile=None,
    mask=None, recenter=False, bbox=None
):
    """
    Calculate the Gaussian probability density for a given element in the trajectory.
    Supports atom masking, recentering, and custom bounding box.

    Args:
        atoms_list (list): List of ASE Atoms objects.
        element (str): Element symbol to compute density for.
        sigma (float): Gaussian width (Å).
        n_sigma (float): Cutoff in sigma units.
        density (float): Grid density (points/Å).
        outputfile (str): Output file name.
        mask (list): Atom indices to include.
        recenter (bool): Whether to recenter trajectory.
        bbox (np.ndarray): Bounding box for grid.

    Returns:
        None. Writes density to file.
    """
    traj = Trajectory.from_atoms(atoms_list)
    params = {
        'element': element,
        'sigma': sigma,
        'n_sigma': n_sigma,
        'density': density,
        'outputfile': outputfile if outputfile else f'{element}_Density_samos.xsf'
    }
    if mask is not None:
        params['mask'] = mask
    if recenter:
        params['recenter'] = True
    if bbox is not None:
        params['bbox'] = bbox
    print(f"Calculating probability density for {element} (sigma={sigma}, n_sigma={n_sigma}, density={density})")
    if mask is not None:
        print(f"  Using atom mask: {mask}")
    if recenter:
        print("  Recentering trajectory before density calculation.")
    if bbox is not None:
        print(f"  Using custom bounding box: {bbox}")
    get_gaussian_density(traj, **params)
    print(f"Density written to {params['outputfile']}")

def main():
    """
    Command-line interface for probability density calculation.

    Supports QE, LAMMPS, and BOMD input modes. Builds trajectory and computes
    the probability density for the specified element.

    Returns:
        None
    """
    import os
    parser = argparse.ArgumentParser(
        description="Ionic Density Calculator (QE, LAMMPS, or BOMD)"
    )
    # QE-style inputs
    parser.add_argument('--in-file',    help='Input .in file for QE/BOMD trajectory')
    parser.add_argument('--pos-file',   help='Input .pos file for QE trajectory')
    parser.add_argument('--cel-file',   help='Input .cel file for QE trajectory')
    parser.add_argument('--lammps-file', nargs='?', help='LAMMPS dump file (.lammpstrj)')
    parser.add_argument('--lammps-elements', nargs='+',
                        help='LAMMPS atom-type symbols (e.g., Li La Ti O)')
    parser.add_argument('--element-mapping', nargs='+',
                        help='LAMMPS type-to-element map (e.g., 1:Li 2:La 3:Ti 4:O)')
    parser.add_argument('--lammps-timestep', type=float,
                        help='LAMMPS timestep (ps)')
    parser.add_argument('--bomd-trj', help='BOMD trajectory file (.mdtrj)')
    parser.add_argument('--bomd-elements', nargs='+', help='Element symbols for BOMD atom order (e.g., Li O Ti)')
    parser.add_argument('--element',    default='Li', help='Species for density calc')
    parser.add_argument('--sigma',      type=float, default=0.3, help='Gaussian σ (Å)')
    parser.add_argument('--n-sigma',    type=float, default=4.0, help='Cutoff (σ units)')
    parser.add_argument('--density',    type=float, default=0.2, help='Grid density (pts/Å)')
    parser.add_argument('--step-skip',  type=int,   default=1,   help='Frame stride')
    parser.add_argument('--num-frames', type=int,   default=0,   help='Max frames (0=all)')
    parser.add_argument('--mask', default=None, help="Atom mask (e.g. '0,1,2,5-10') for density calculation")
    parser.add_argument('--recenter', action='store_true', help="Recenter trajectory before density calculation")
    parser.add_argument('--bbox', default=None, help="Bounding box for grid as 'xmin,xmax,ymin,ymax,zmin,zmax' (comma-separated)")
    parser.add_argument('--output',     default='density.xsf', help='Output XSF file')
    args = parser.parse_args()

    atoms_list = None

    # --- BOMD branch ---
    if args.bomd_trj:
        # Use BOMD elements if provided, else fallback to None
        bomd_elements = args.bomd_elements if args.bomd_elements else None
        # Try to find .in file if not provided
        in_file = args.in_file if args.in_file else None
        atoms_list = build_bomd_trajectory(
            args.bomd_trj,
            in_file=in_file,
            elements=bomd_elements,
            num_frames=args.num_frames,
            stride=args.step_skip
        )

    # --- LAMMPS branch ---
    elif args.lammps_file:
        mapping = {}
        if args.element_mapping:
            for m in args.element_mapping:
                tid, sym = m.split(':')
                mapping[int(tid)] = sym

        pos_full, n_frames, dt_full, t_full, cell_full, thermo, volumes, types = inp.read_lammps_trajectory(
            args.lammps_file,
            elements=args.lammps_elements,
            timestep=args.lammps_timestep,
            element_mapping=mapping
        )
        atoms_list = []
        for i in range(n_frames):
            cell = cell_full[i].reshape(3,3)
            coords = pos_full[:, i, :]
            atoms_list.append(Atoms(cell=cell, symbols=types, positions=coords, pbc=True))

    # --- QE branch ---
    elif args.in_file and args.pos_file and args.cel_file:
        atoms_list = build_trajectory(
            args.in_file, args.pos_file, args.cel_file,
            time_after_start=0.0,
            num_frames=args.num_frames,
            time_interval=None,
            stride=args.step_skip
        )
    else:
        parser.error("Provide --bomd-trj, --lammps-file, or all of --in-file, --pos-file, --cel-file")

    # Compute and write density
    traj = Trajectory.from_atoms(atoms_list)
    params = {
        'element': args.element,
        'sigma': args.sigma,
        'n_sigma': args.n_sigma,
        'density': args.density,
        'outputfile': args.output
    }
    if args.mask:
        params['mask'] = args.mask
    if args.recenter:
        params['recenter'] = True
    if args.bbox:
        vals = list(map(float, args.bbox.split(',')))
        params['bbox'] = np.array(vals).reshape(3,2)
    get_gaussian_density(traj, **params)
    print(f"Density file: {args.output}")

if __name__ == '__main__':
    main()