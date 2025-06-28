# #!/usr/bin/env python3
# """
# Ultimate Probability Density Calculator (Extensible Version)
# Integrates CPDyAna and SAMOS methodologies for direct use with .in, .pos, .cel, .evp files.
# Now supports advanced options: frame stride, atom masking, recentering, and custom bounding box.
# """

# import numpy as np
# import argparse
# from ase import Atoms
# from samos_modules.samos_trajectory import Trajectory
# from samos_modules.samos_analysis import get_gaussian_density
# from target import input_reader as inp

# def divide_chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]

# def convert_symbols_to_atomic_numbers(element_symbols):
#     """Convert element symbols to atomic numbers."""
#     element_to_atomic_number = {
#         'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
#         'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
#         'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
#         'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
#         'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
#         'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
#         'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
#         'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
#         'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
#         'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
#         'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
#         'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
#         'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103
#     }
#     atomic_numbers = []
#     for symbol in element_symbols:
#         if symbol in element_to_atomic_number:
#             atomic_numbers.append(element_to_atomic_number[symbol])
#         else:
#             raise ValueError(f"Unknown element symbol: {symbol}")
#     return atomic_numbers

# def build_trajectory(
#     in_file, pos_file, cel_file, time_after_start=0.0, num_frames=None, time_interval=0.00193511, stride=1
# ):
#     """
#     Build an in-memory trajectory from .in, .pos, .cel files.
#     Returns a list of ASE Atoms objects.
#     Supports frame stride for subsampling.
#     """
#     ang = 0.529177249  # Bohr to Angstrom conversion factor

#     # 1. Read atomic species from .in file
#     element_symbols = inp.read_ion_file(in_file)
#     sp = convert_symbols_to_atomic_numbers(element_symbols)

#     # 2. Read cell and position files
#     cellfile = open(cel_file).read().splitlines()
#     posfile = open(pos_file).read().splitlines()
#     lats = list(divide_chunks(cellfile, 4))
#     coords = list(divide_chunks(posfile, len(sp)+1))

#     total_frames = min(len(lats), len(coords))
#     start_frame = int(time_after_start / time_interval)
#     if num_frames is None or num_frames == 0:
#         end_frame = total_frames
#     else:
#         end_frame = min(start_frame + num_frames, total_frames)

#     atoms_list = []
#     for i in range(start_frame, end_frame, stride):
#         tlat = np.loadtxt(lats[i], skiprows=1).transpose() * ang
#         tpos = np.loadtxt(coords[i], skiprows=1) * ang
#         atoms = Atoms(cell=tlat, numbers=sp, positions=tpos, pbc=True)
#         atoms_list.append(atoms)
#     print(f"Trajectory built: {len(atoms_list)} frames from {start_frame} to {end_frame-1} (stride={stride})")
#     return atoms_list

# def parse_mask(mask_str, total_atoms):
#     """
#     Parse a mask string like '0,1,2,5-10' into a list of atom indices.
#     """
#     if not mask_str:
#         return None
#     indices = set()
#     for part in mask_str.split(','):
#         if '-' in part:
#             start, end = map(int, part.split('-'))
#             indices.update(range(start, end+1))
#         else:
#             indices.add(int(part))
#     # Ensure indices are within bounds
#     indices = [i for i in indices if 0 <= i < total_atoms]
#     return indices

# def calculate_probability_density(
#     atoms_list, element='Li', sigma=0.3, n_sigma=3.0, density=0.1, outputfile=None,
#     mask=None, recenter=False, bbox=None
# ):
#     """
#     Calculate the Gaussian probability density for a given element in the trajectory.
#     Supports atom masking, recentering, and custom bounding box.
#     """
#     traj = Trajectory.from_atoms(atoms_list)
#     params = {
#         'element': element,
#         'sigma': sigma,
#         'n_sigma': n_sigma,
#         'density': density,
#         'outputfile': outputfile if outputfile else f'{element}_Density_samos.xsf'
#     }
#     if mask is not None:
#         params['mask'] = mask
#     if recenter:
#         params['recenter'] = True
#     if bbox is not None:
#         params['bbox'] = bbox
#     print(f"Calculating probability density for {element} (sigma={sigma}, n_sigma={n_sigma}, density={density})")
#     if mask is not None:
#         print(f"  Using atom mask: {mask}")
#     if recenter:
#         print("  Recentering trajectory before density calculation.")
#     if bbox is not None:
#         print(f"  Using custom bounding box: {bbox}")
#     get_gaussian_density(traj, **params)
#     print(f"Density written to {params['outputfile']}")

# def main():
#     parser = argparse.ArgumentParser(
#         description="Probability Density Calculator for Molecular Dynamics Trajectories"
#     )
#     parser.add_argument('--in-file', required=True, help='Input .in file')
#     parser.add_argument('--pos-file', required=True, help='Input .pos file')
#     parser.add_argument('--cel-file', required=True, help='Input .cel file')
#     parser.add_argument('--evp-file', required=False, help='Input .evp file (optional)')
#     parser.add_argument('--element', default='Li', help='Element for density calculation')
#     parser.add_argument('--sigma', type=float, default=0.3, help='Gaussian sigma (Å)')
#     parser.add_argument('--n-sigma', type=float, default=3.0, help='Cutoff in units of sigma')
#     parser.add_argument('--density', type=float, default=0.1, help='Grid density (points/Å)')
#     parser.add_argument('--output', default=None, help='Output XSF filename')
#     parser.add_argument('--time-after-start', type=float, default=0.0, help='Start time (ps)')
#     parser.add_argument('--num-frames', type=int, default=0, help='Number of frames (0 for all)')
#     parser.add_argument('--time-interval', type=float, default=0.00193511, help='Time interval between frames (ps)')
#     parser.add_argument('--step-skip', type=int, default=1, help='Number of steps to skip between frames')
#     parser.add_argument('--mask', default=None, help="Atom mask (e.g. '0,1,2,5-10') for density calculation")
#     parser.add_argument('--recenter', action='store_true', help="Recenter trajectory before density calculation")
#     parser.add_argument('--bbox', default=None, help="Bounding box for grid as 'xmin,xmax,ymin,ymax,zmin,zmax' (comma-separated)")

#     args = parser.parse_args()

#     # Build trajectory
#     atoms_list = build_trajectory(
#         args.in_file, args.pos_file, args.cel_file,
#         time_after_start=args.time_after_start,
#         num_frames=args.num_frames,
#         time_interval=args.time_interval,
#         stride=args.step_skip
#     )

#     # Parse mask and bounding box if provided
#     mask = None
#     if args.mask:
#         mask = parse_mask(args.mask, len(atoms_list[0]))

#     bbox = None
#     if args.bbox:
#         try:
#             bbox_vals = [float(x) for x in args.bbox.split(',')]
#             if len(bbox_vals) == 6:
#                 bbox = np.array(bbox_vals).reshape((3,2))
#             else:
#                 print("Bounding box must have 6 comma-separated values (xmin,xmax,ymin,ymax,zmin,zmax)")
#         except Exception as e:
#             print(f"Error parsing bounding box: {e}")

#     # Calculate probability density
#     calculate_probability_density(
#         atoms_list,
#         element=args.element,
#         sigma=args.sigma,
#         n_sigma=args.n_sigma,
#         density=args.density,
#         outputfile=args.output,
#         mask=mask,
#         recenter=args.recenter,
#         bbox=bbox
#     )

# if __name__ == '__main__':
#     main()

import numpy as np
import argparse
from ase import Atoms
from ase.io import read as ase_read
from samos_modules.samos_trajectory import Trajectory
from samos_modules.samos_analysis import get_gaussian_density
from target import input_reader as inp

def divide_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def convert_symbols_to_atomic_numbers(element_symbols):
    """Convert element symbols to atomic numbers."""
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
    Build an in-memory trajectory from .in, .pos, .cel files.
    Returns a list of ASE Atoms objects.
    Supports frame stride for subsampling.
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
    """
    all_frames = ase_read(lammps_file, index=":", format="lammps-dump-text")
    if max_frames is not None:
        all_frames = all_frames[:max_frames]
    atoms_list = [frame for idx, frame in enumerate(all_frames) if idx % stride == 0]
    print(f"LAMMPS trajectory built: {len(atoms_list)} frames (stride={stride})")
    return atoms_list

def parse_mask(mask_str, total_atoms):
    """
    Parse a mask string like '0,1,2,5-10' into a list of atom indices.
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
    parser = argparse.ArgumentParser(
        description="Ionic Density Calculator (QE or LAMMPS)"
    )
    # QE-style inputs
    parser.add_argument('--in-file',    help='Input .in file for QE trajectory')
    parser.add_argument('--pos-file',   help='Input .pos file for QE trajectory')
    parser.add_argument('--cel-file',   help='Input .cel file for QE trajectory')
    parser.add_argument('--lammps-file', nargs='?', help='LAMMPS dump file (.lammpstrj)')
    parser.add_argument('--lammps-elements', nargs='+',
                        help='LAMMPS atom-type symbols (e.g., Li La Ti O)')
    parser.add_argument('--element-mapping', nargs='+',
                        help='LAMMPS type-to-element map (e.g., 1:Li 2:La 3:Ti 4:O)')
    parser.add_argument('--lammps-timestep', type=float,
                        help='LAMMPS timestep (ps)')
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

    # Determine which input mode to use
    if args.lammps_file:
        # Build mapping dict
        mapping = {}
        if args.element_mapping:
            for m in args.element_mapping:
                tid, sym = m.split(':')
                mapping[int(tid)] = sym

        # Read LAMMPS trajectory
        pos_full, n_frames, dt_full, t_full, cell_full, thermo, volumes, types = inp.read_lammps_trajectory(
            args.lammps_file,
            elements=args.lammps_elements,
            timestep=args.lammps_timestep,
            element_mapping=mapping
        )
        # Assemble ASE Atoms list
        atoms_list = []
        for i in range(n_frames):
            cell = cell_full[i].reshape(3,3)
            coords = pos_full[:, i, :]
            atoms_list.append(Atoms(cell=cell, symbols=types, positions=coords, pbc=True))

    elif args.in_file and args.pos_file and args.cel_file:
        # QE trajectory
        from target.probability_density import build_trajectory
        atoms_list = build_trajectory(
            args.in_file, args.pos_file, args.cel_file,
            time_after_start=0.0,
            num_frames=args.num_frames,
            time_interval=None,
            stride=args.step_skip
        )
    else:
        parser.error("Provide either --lammps-file or all of --in-file, --pos-file, --cel-file")

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
        # parse bbox string into [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        vals = list(map(float, args.bbox.split(',')))
        params['bbox'] = np.array(vals).reshape(3,2)
    get_gaussian_density(traj, **params)
    print(f"Density file: {args.output}")

if __name__ == '__main__':
    main()