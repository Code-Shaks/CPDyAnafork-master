#!/usr/bin/env python3
"""
lammps_to_qe.py – A comprehensive tool to read LAMMPS dump files and convert them to Quantum ESPRESSO input files.

This script combines detailed parsing of LAMMPS dump files with the ability to write Quantum ESPRESSO input files using ASE. It supports various element mapping methods, optional data like velocities and forces, and flexible output options.

Usage Examples
--------------
# Convert LAMMPS dump to QE input files (one per snapshot):
python lammps_to_qe.py traj.lammpstrj output_%05d.in --split --units metal --map 1:Li 2:O

# Convert with a specific timestep and save only the first frame:
python lammps_to_qe.py traj.lammpstrj output.in --units real --dt 0.5 --map 1:Li

# Use an elements file for mapping and skip initial frames:
python lammps_to_qe.py traj.lammpstrj output_%05d.in --split --elements-file elements.txt --skip 10
"""

from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from ase import Atoms
from ase.io import write
import os
from ase.data import atomic_masses, chemical_symbols

# Unit conversion factors for LAMMPS
TIME_CONV = {
    'metal': 41341.37333656114,  # 1 ps → a.u.
    'real': 41.34137333656114,   # 1 fs → a.u.
    'si': 4.134137333656114e16,  # 1 s → a.u.
    'nano': 41.34137333656114,   # 1 ns* → a.u. (LAMMPS stores fs)
    'lj': 1.0                    # dimensionless
}

LEN_CONV = {
    'metal': 1.0,
    'real': 1.0,
    'si': 1.0e10,  # m → Å
    'nano': 10.0,  # nm → Å
    'lj': 1.0
}

def read_step_info(lines: List[str], lidx: int = 0, start: bool = False, additional_kw: List[str] = [], quiet: bool = False) -> Tuple[int, int, Optional[int], Optional[int], str, Tuple[int, int, int], bool, Tuple[int, ...], bool, Tuple[int, ...], List[int]]:
    """
    Extract information from a LAMMPS dump step header.
    Returns: (nat, atomid_idx, element_idx, type_idx, postype, posids, has_vel, velids, has_frc, frcids, additional_ids)
    """
    if not lines[0].startswith('ITEM: TIMESTEP'):
        raise ValueError(f"Expected 'ITEM: TIMESTEP' at line {lidx}, got {lines[0]}")
    if not lines[2].startswith('ITEM: NUMBER OF ATOMS'):
        raise ValueError(f"Expected 'ITEM: NUMBER OF ATOMS' at line {lidx+2}, got {lines[2]}")
    nat = int(lines[3].strip())
    if not lines[4].startswith('ITEM: BOX BOUNDS'):
        raise ValueError(f"Expected 'ITEM: BOX BOUNDS' at line {lidx+4}, got {lines[4]}")
    if not lines[8].startswith('ITEM: ATOMS'):
        raise ValueError(f"Expected 'ITEM: ATOMS' at line {lidx+8}, got {lines[8]}")
    
    header = lines[8].split()[2:]  # Skip 'ITEM: ATOMS'
    atomid_idx = header.index('id') if 'id' in header else -1
    element_idx = header.index('element') if 'element' in header else None
    type_idx = header.index('type') if 'type' in header else None
    
    postype = 'unknown'
    posids = (-1, -1, -1)
    for ptype in ['x', 'xs', 'xu']:
        if ptype in header:
            postype = ptype
            posids = (header.index(ptype), header.index(ptype.replace('x', 'y')), header.index(ptype.replace('x', 'z')))
            break
    
    has_vel = all(v in header for v in ['vx', 'vy', 'vz'])
    velids = (header.index('vx'), header.index('vy'), header.index('vz')) if has_vel else tuple()
    
    has_frc = all(f in header for f in ['fx', 'fy', 'fz'])
    frcids = (header.index('fx'), header.index('fy'), header.index('fz')) if has_frc else tuple()
    
    additional_ids = [header.index(kw) for kw in additional_kw if kw in header]
    
    if start and not quiet:
        print(f"Detected position type: {postype}")
        if has_vel:
            print("Velocities detected")
        if has_frc:
            print("Forces detected")
        if additional_ids:
            print(f"Additional keywords detected: {[additional_kw[i] for i, kw in enumerate(additional_kw) if header.index(kw) in additional_ids]}")
    
    return (nat, atomid_idx, element_idx, type_idx, postype, posids, has_vel, velids, has_frc, frcids, additional_ids)

def pos_2_absolute(cell: np.ndarray, pos: np.ndarray, postype: str) -> np.ndarray:
    """Convert positions to absolute coordinates based on position type."""
    if postype == 'x':
        return pos
    elif postype == 'xs':
        # Scaled coordinates to absolute
        lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]
        return pos * np.array([lx, ly, lz])
    elif postype == 'xu':
        # Unwrapped coordinates (assume already absolute for simplicity)
        return pos
    else:
        raise NotImplementedError(f"Position type {postype} not supported")

def read_lammps_dump(filename: str, elements: Optional[List[str]] = None, elements_file: Optional[str] = None,
                     types: Optional[List[str]] = None, mass_types: Optional[str] = None,
                     ignore_forces: bool = False, ignore_velocities: bool = False, skip: int = 0,
                     istep: int = 1, additional_keywords_dump: List[str] = [], quiet: bool = False) -> List[dict]:
    """
    Read a LAMMPS dump file and return a list of snapshots with detailed data.
    Adapted from paste-4.txt to focus on data extraction for QE conversion.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"LAMMPS dump file {filename} not found")

    # First pass: Check file structure and determine element mapping
    with open(filename) as f:
        lines = [next(f) for _ in range(9)]
        (nat_must, atomid_idx, element_idx, type_idx, postype, posids, has_vel, velids,
         has_frc, frcids, additional_ids_dump) = read_step_info(
            lines, lidx=0, start=True, additional_kw=additional_keywords_dump, quiet=quiet)
        if ignore_forces:
            has_frc = False
        if ignore_velocities:
            has_vel = False
        
        body = np.array([f.readline().split() for _ in range(nat_must)])
        atomids = np.array(body[:, atomid_idx], dtype=int)
        sorting_key = atomids.argsort()
        
        # Determine symbols for atoms
        if types is not None:
            if type_idx is None:
                raise ValueError("Types specified but not found in file")
            types_in_body = np.array(body[:, type_idx][sorting_key], dtype=int) - 1  # 1-based to 0-based
            symbols = np.array(types, dtype=str)[types_in_body]
        elif element_idx is not None:
            symbols = np.array(body[:, element_idx])[sorting_key]
        elif elements is not None:
            if len(elements) != nat_must:
                raise ValueError(f"Length of elements list ({len(elements)}) does not match number of atoms ({nat_must})")
            symbols = elements[:]
        elif elements_file is not None:
            if not os.path.exists(elements_file):
                raise FileNotFoundError(f"Elements file {elements_file} not found")
            with open(elements_file) as f:
                for line in f:
                    if line:
                        break
                elements = line.strip().split()
                if len(elements) != nat_must:
                    raise ValueError(f"Length of elements list ({len(elements)}) does not match number of atoms ({nat_must})")
                symbols = elements[:]
        elif mass_types is not None:
            if not os.path.exists(mass_types):
                raise FileNotFoundError(f"Mass types file {mass_types} not found")
            with open(mass_types) as fmass:
                masses = []
                reading_masses = False
                for line in fmass:
                    line = line.strip()
                    if not line:
                        continue
                    if reading_masses:
                        try:
                            typ, mass = line.split()[:2]
                            masses.append((int(typ), float(mass)))
                        except ValueError:
                            break
                    elif line.startswith('Masses'):
                        reading_masses = True
            if not masses:
                raise ValueError("No masses found in mass_types file")
            type_indices, mass_values = zip(*masses)
            mass_values = np.array(mass_values, dtype=float)
            type_indices = np.array(type_indices, dtype=int)
            if not np.all(np.arange(1, type_indices.max() + 1) == type_indices):
                raise ValueError("Types are not consecutive in mass_types file")
            types_list = []
            for mass in mass_values:
                idx = np.argmin(np.abs(atomic_masses - mass))
                types_list.append(chemical_symbols[idx])
            if type_idx is None:
                raise ValueError("Type field not found in dump file for mass-based mapping")
            types_in_body = np.array(body[:, type_idx][sorting_key], dtype=int) - 1
            symbols = np.array(types_list, dtype=str)[types_in_body]
        else:
            symbols = ['H'] * nat_must  # Fallback to Hydrogen

    # Second pass: Read full trajectory data
    snaps = []
    lidx = 0
    iframe = 0
    with open(filename) as f:
        while True:
            step_info = [f.readline() for _ in range(9)]
            if ''.join(step_info) == '':
                if not quiet:
                    print(f"End reached at line {lidx}, stopping")
                break
            nat = int(step_info[3].strip())
            timestep_ = int(step_info[1].strip())
            bounds = [list(map(float, step_info[i].split())) for i in range(5, 8)]
            (xlo, xhi, *xt), (ylo, yhi, *yt), (zlo, zhi, *zt) = bounds
            xy, xz, yz = (xt + yt + zt + [0, 0, 0])[:3]
            lidx += 9
            if nat != nat_must:
                print("Changing number of atoms is not supported, breaking")
                break
            
            body = np.array([f.readline().split() for _ in range(nat_must)])
            lidx += nat_must
            atomids = np.array(body[:, atomid_idx], dtype=int)
            sorting_key = atomids.argsort()
            pos = np.array(body[:, posids], dtype=float)[sorting_key]
            
            if iframe >= skip and iframe % istep == 0:
                snap = {
                    'step': timestep_,
                    'natoms': nat,
                    'xlo': xlo, 'xhi': xhi, 'xy': xy,
                    'ylo': ylo, 'yhi': yhi, 'xz': xz,
                    'zlo': zlo, 'zhi': zhi, 'yz': yz,
                    'atoms': []
                }
                lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
                cell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
                absolute_pos = pos_2_absolute(cell, pos, postype)
                for i, idx in enumerate(sorting_key):
                    atom_data = {
                        'type': int(body[idx, type_idx]) if type_idx is not None else 0,
                        'symbol': symbols[i],
                        'x_qe': absolute_pos[i, 0],
                        'y_qe': absolute_pos[i, 1],
                        'z_qe': absolute_pos[i, 2]
                    }
                    if has_vel:
                        vel = np.array(body[idx, velids], dtype=float)
                        atom_data['vx'], atom_data['vy'], atom_data['vz'] = vel
                    if has_frc:
                        frc = np.array(body[idx, frcids], dtype=float)
                        atom_data['fx'], atom_data['fy'], atom_data['fz'] = frc
                    snap['atoms'].append(atom_data)
                snaps.append(snap)
            iframe += 1
    if not quiet:
        print(f"Read trajectory of length {iframe}, created {len(snaps)} snapshots after skipping and stepping")
    return snaps

def cell_matrix(ts: dict) -> np.ndarray:
    """Build a 3x3 cell matrix from snapshot bounds and tilt factors."""
    lx, ly, lz = ts['xhi'] - ts['xlo'], ts['yhi'] - ts['ylo'], ts['zhi'] - ts['zlo']
    return np.array([[lx, 0, 0],
                     [ts['xy'], ly, 0],
                     [ts['xz'], ts['yz'], lz]])

def ts_to_atoms(ts: dict) -> Atoms:
    """Convert a LAMMPS snapshot to an ASE Atoms object."""
    symbols = [a['symbol'] for a in ts['atoms']]
    positions = [[a['x_qe'], a['y_qe'], a['z_qe']] for a in ts['atoms']]
    cell = cell_matrix(ts)
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def write_qe(fname: str, atoms: Atoms):
    """Write an ASE Atoms object to a Quantum ESPRESSO input file with dummy pseudopotentials."""
    inp = {
        'control': {
            'calculation': 'md',
            'restart_mode': 'from_scratch',
            'prefix': 'lmp2qe',
            'pseudo_dir': './pseudo',
            'outdir': './out'
        },
        'system': {
            'ibrav': 0,
            'ecutwfc': 60.0
        },
        'electrons': {},
        'ions': {},
        'cell': {}
    }
    
    # Create a dummy pseudopotential mapping for each unique symbol
    symbols = set(atoms.get_chemical_symbols())
    pseudopotentials = {sym: f"{sym}.dummy.UPF" for sym in symbols}
    print(f"Warning: Using dummy pseudopotential placeholders: {pseudopotentials}. Edit the QE input file to specify actual pseudopotential files if needed.")
    
    write(fname, atoms, format='espresso-in', input_data=inp, pseudopotentials=pseudopotentials)

def dump_to_qe(dump: str, output: str, units: str, dt: Optional[float] = None,
               split: bool = False, elements: Optional[List[str]] = None,
               elements_file: Optional[str] = None, types: Optional[List[str]] = None,
               mass_types: Optional[str] = None, skip: int = 0, istep: int = 1,
               ignore_forces: bool = False, ignore_velocities: bool = False,
               quiet: bool = False):
    """Convert a LAMMPS dump file to Quantum ESPRESSO input files."""
    conv_len = LEN_CONV[units]
    frames = read_lammps_dump(
        dump, elements=elements, elements_file=elements_file, types=types,
        mass_types=mass_types, ignore_forces=ignore_forces,
        ignore_velocities=ignore_velocities, skip=skip, istep=istep, quiet=quiet
    )
    
    if not frames:
        raise ValueError("No frames read from the dump file after skipping and stepping")
    
    if split:
        for idx, ts in enumerate(frames):
            fname = output % idx
            atoms = ts_to_atoms(ts)
            write_qe(fname, atoms)
            if not quiet:
                print(f"wrote {fname}")
    else:
        atoms0 = ts_to_atoms(frames[0])
        write_qe(output, atoms0)
        if not quiet:
            print(f"wrote {output} (first snapshot only)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LAMMPS dump → Quantum ESPRESSO input files")
    ap.add_argument("dump", help="LAMMPS text dump file (dump ... custom ...)")
    ap.add_argument("output", help="Output file name; use %05d with --split for multiple files")
    ap.add_argument("--units", choices=LEN_CONV, default="metal", help="LAMMPS units keyword (default metal)")
    ap.add_argument("--dt", type=float, help="Timestep in original units (for reference only)")
    ap.add_argument("--split", action="store_true", help="Write one QE file per snapshot")
    ap.add_argument("--map", nargs="+", help="Type:symbol list, e.g. --map 1:Li 2:O")
    ap.add_argument("--elements-file", help="File with space-separated element symbols")
    ap.add_argument("--skip", type=int, default=0, help="Number of initial steps to skip")
    ap.add_argument("--istep", type=int, default=1, help="Frequency of steps to take from trajectory")
    ap.add_argument("--ignore-forces", action="store_true", help="Ignore force data if present")
    ap.add_argument("--ignore-velocities", action="store_true", help="Ignore velocity data if present")
    ap.add_argument("--quiet", action="store_true", help="Suppress output messages")
    
    args = ap.parse_args()
    
    # Process element mapping from --map if provided
    types_map = None
    if args.map:
        types_map = []
        max_type = 0
        for m in args.map:
            try:
                t, s = m.split(":")
                t_int = int(t)
                max_type = max(max_type, t_int)
            except ValueError:
                ap.error(f"Bad mapping: {m}")
        types_map = ['X'] * max_type  # Placeholder for unused types
        for m in args.map:
            t, s = m.split(":")
            types_map[int(t) - 1] = s  # Adjust for 0-based indexing
    
    dump_to_qe(
        args.dump, args.output, args.units, args.dt, args.split,
        elements=None, elements_file=args.elements_file, types=types_map,
        mass_types=None, skip=args.skip, istep=args.istep,
        ignore_forces=args.ignore_forces, ignore_velocities=args.ignore_velocities,
        quiet=args.quiet
    )
