#!/usr/bin/env python3
"""
compute_vaf.py

Read QE .in/.pos/.cel[/.evp], build ASE+Velocities, compute VAF with SAMOS, and plot.

This script reads Quantum ESPRESSO trajectory files, constructs ASE Atoms objects with velocities,
computes the velocity autocorrelation function (VAF) for specified elements using the SAMOS library,
and plots/saves the results.

Usage example:
    python vaf.py --in_file LiAlPS.in --pos_file LiAlPS.pos --cel_file LiAlPS.cel --element Li Na
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import glob
from ase import Atoms
from target.trajectory import Trajectory
from target.analysis import DynamicsAnalyzer
from plotting import plot_vaf_isotropic

# Add target directory to sys.path for input_reader import
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'target')))
from target import input_reader as inp

# Mapping from element symbol to atomic number
element_to_atomic_number = {
    'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,
    'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,
    'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,
    'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,
    'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
    'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,
    'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,
    'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,
    'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,
    'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,
    'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,
    'Ac':89,'Th':90,'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,
    'Bk':97,'Cf':98,'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103
}

def convert_symbols_to_atomic_numbers(symbols):
    """
    Convert a list of element symbols to atomic numbers.

    Args:
        symbols (list): List of element symbols (str).

    Returns:
        list: List of atomic numbers (int).
    """
    try:
        return [element_to_atomic_number[s] for s in symbols]
    except KeyError as e:
        raise ValueError(f"Unknown element symbol: {e}") from e

def read_cel(cel_file):
    """
    Read cell parameters from a .cel file.

    Args:
        cel_file (str): Path to .cel file.

    Returns:
        list: List of 4-line blocks (cell info per frame).
    """
    lines = open(cel_file).read().splitlines()
    return [lines[i:i+4] for i in range(0, len(lines), 4)]

def read_pos(pos_file, natoms):
    """
    Read atomic positions from a .pos file.

    Args:
        pos_file (str): Path to .pos file.
        natoms (int): Number of atoms.

    Returns:
        list: List of (natoms+1)-line blocks (positions per frame).
    """
    lines = open(pos_file).read().splitlines()
    return [lines[i:i+natoms+1] for i in range(0, len(lines), natoms+1)]

def read_evp(evp_file):
    """
    Read time steps from a .evp file.

    Args:
        evp_file (str): Path to .evp file.

    Returns:
        list: List of times (float).
    """
    times = []
    try:
        for L in open(evp_file):
            parts = L.split()
            if len(parts) >= 2 and parts[0].isdigit():
                times.append(float(parts[1]))
    except FileNotFoundError:
        pass
    return times

def finite_diff_velocities(pos_arr, times):
    """
    Compute velocities using finite differences from positions and times.

    Args:
        pos_arr (np.ndarray): Array of shape (Nframes, natoms, 3) with positions.
        times (list): List of time values (ps), length Nframes.

    Returns:
        np.ndarray: Array of velocities, same shape as pos_arr.
    """
    N, M, _ = pos_arr.shape
    v = np.zeros_like(pos_arr)
    # Use uniform timestep if possible
    if len(times) >= 2:
        dt0 = (times[1] - times[0]) * 1000  # convert ps to fs
    else:
        dt0 = 1.0
    # Central differences for interior points
    for i in range(1, N-1):
        v[i] = (pos_arr[i+1] - pos_arr[i-1]) / (2 * dt0)
    # Forward/backward for endpoints
    v[0]   = (pos_arr[1] - pos_arr[0])  / dt0
    v[-1]  = (pos_arr[-1] - pos_arr[-2]) / dt0
    return v

def build_trajectory(in_file, pos_file, cel_file,
                     evp_file=None,
                     start=0.0, nframes=0, stride=1,
                     time_interval=0.00193511):
    """
    Build a trajectory as a list of ASE Atoms objects with velocities.

    Args:
        in_file (str): Path to .in file (species).
        pos_file (str): Path to .pos file (positions).
        cel_file (str): Path to .cel file (cell parameters).
        evp_file (str, optional): Path to .evp file (timing).
        start (float): Start time in ps.
        nframes (int): Number of frames to use (0=all).
        stride (int): Stride for frames.
        time_interval (float): Default time between frames (ps) if no .evp.

    Returns:
        tuple: (frames, dt)
            frames: list of ASE Atoms objects with velocities.
            dt: time step in ps.
    """
    BOHR2ANG = 0.529177249
    # 1) Read species symbols
    syms = inp.read_ion_file(in_file)
    # 2) Convert to atomic numbers
    sp = convert_symbols_to_atomic_numbers(syms)
    # 3) Read cell and position blocks
    cel_chunks = read_cel(cel_file)
    pos_chunks = read_pos(pos_file, len(syms))
    # 4) Read times from .evp if available, else use uniform spacing
    times_full = read_evp(evp_file) if evp_file else []
    total = min(len(cel_chunks), len(pos_chunks))

    # Determine start/end indices
    if len(times_full) >= 2:
        dt = times_full[1] - times_full[0]
        start_idx = int(start / dt)
        time_list = times_full[start_idx:total:stride]
    else:
        dt = time_interval
        start_idx = int(start / dt)
        time_list = [i * dt for i in range(start_idx, total, stride)]

    start_idx = max(0, min(start_idx, total-1))
    end_idx = total if nframes <= 0 else min(start_idx + nframes, total)

    # Collect positions and cells
    pos_list = []
    cell_list = []
    for i in range(start_idx, end_idx, stride):
        block_c = cel_chunks[i]
        cell = np.loadtxt(block_c[1:], dtype=float).T * BOHR2ANG
        block_p = pos_chunks[i]
        coords = np.loadtxt(block_p[1:], dtype=float) * BOHR2ANG
        pos_list.append(coords)
        cell_list.append(cell)
    pos_arr = np.array(pos_list)  # (Nf, nat, 3)
    # Compute velocities
    vel_arr = finite_diff_velocities(pos_arr, time_list)

    # Build ASE Atoms objects with velocities
    frames = []
    for i, (coords, cell) in enumerate(zip(pos_arr, cell_list)):
        a = Atoms(numbers=sp, positions=coords, cell=cell, pbc=True)
        a.set_velocities(vel_arr[i])
        frames.append(a)
    print(f"Built {len(frames)} frames with velocities.")
    return frames, dt

def compute_vaf(frames, dt, element, blocks, out_prefix, in_file,
                t_start_fit_ps=0, stepsize_t=1, stepsize_tau=1, t_end_fit_ps=10):
    """
    Compute the velocity autocorrelation function (VAF) for a given element.

    Args:
        frames (list): List of ASE Atoms objects with velocities.
        dt (float): Time step in ps.
        element (str): Element symbol (e.g. 'Li').
        blocks (int): Number of statistical blocks.
        out_prefix (str): Output file prefix.
        in_file (str): Path to .in file (for species).
        t_start_dt (int): Start index for VAF calculation.
        stepsize_t (int): Stride for t in VAF.
        stepsize_tau (int): Stride for tau in VAF.
        t_end_fit_ps (float): End of fit in ps.

    Returns:
        tuple: (t, vaf, vint)
            t: time array (ps)
            vaf: VAF array
            vint: VAF integral array
    """
    traj = Trajectory.from_atoms(frames)
    traj.set_attr('timestep_fs', dt * 1000.0)
    da = DynamicsAnalyzer()
    da.set_trajectories(traj)
    # Map species to indices
    all_syms = inp.read_ion_file(in_file)
    indices = [i for i, s in enumerate(all_syms) if s == element]
    # Compute VAF
    ts = da.get_vaf(
        integration='trapezoid',
        species_of_interest=[element],
        nr_of_blocks=blocks,
        t_start_fit_ps=t_start_fit_ps,
        stepsize_t=stepsize_t,
        stepsize_tau=stepsize_tau,
        t_end_fit_ps=t_end_fit_ps
    )
    # Extract mean arrays
    key_mean = f"vaf_isotropic_{element}_mean"
    key_int  = f"vaf_integral_isotropic_{element}_mean"
    vaf  = ts.get_array(key_mean)
    vint = ts.get_array(key_int)
    t = np.arange(len(vaf)) * dt
    # Write VAF and integral to disk
    out_vaf = f"{out_prefix}_{element}_vaf.dat"
    out_int = f"{out_prefix}_{element}_vint.dat"
    np.savetxt(out_vaf,  np.column_stack((t, vaf)),  header="t(ps)  VAF")
    np.savetxt(out_int, np.column_stack((t, vint)), header="t(ps)  integrationVAF")
    print("Wrote", out_vaf, out_int)
    attrs = ts.get_attrs()
    if element in attrs and "diffusion_mean_cm2_s" in attrs[element]:
        D = attrs[element]["diffusion_mean_cm2_s"]
        print(f"Diffusion coeff (cm²/s): {D:.3e}")
    else:
        print("Diffusion coefficient not found in VAF results. Available keys:", list(attrs.keys()))
    return t, vaf, vint

if __name__ == '__main__':
    # Argument parsing for CLI usage
    p = argparse.ArgumentParser(description="Compute and plot VAF from QE trajectory files.")
    p.add_argument('--data-dir', help="Directory containing trajectory files (QE or LAMMPS)")
    p.add_argument('--lammps-elements', nargs='+', help="Element symbols for LAMMPS atom types (e.g., Li S Al P O)")
    p.add_argument('--element-mapping', nargs='+', help="LAMMPS type to element mapping (e.g., 1:Li 2:S 3:Al)")
    p.add_argument('--lammps-timestep', type=float, help="LAMMPS timestep in picoseconds")
    p.add_argument('--element',   nargs='+', required=True,
                   help="Atom symbol(s) for VAF (e.g. Li Na)")
    p.add_argument('--start',   type=float, default=0.0,
                   help="Time (ps) to start analysis")
    p.add_argument('--nframes', type=int,   default=0,
                   help="Number of frames (0=all)")
    p.add_argument('--stride',  type=int,   default=1,
                   help = "Stride for frames (1=all, 2=every other, etc.)")
    p.add_argument('--blocks',  type=int,   default=1,
                   help="Number of blocks for error estimates")
    p.add_argument('--out-prefix', default='vaf',
                   help="Prefix for output files")
    p.add_argument('--time-interval', type=float, default=0.00193511,
                   help='Default time between frames (ps) if no .evp file')
    p.add_argument('--t-start-fit-ps', type=float, default=0, help="Start of the fit in ps (for SAMOS, default: 0)")
    p.add_argument('--stepsize-t', type=int, default=1, help="Stride for t in VAF")
    p.add_argument('--stepsize-tau', type=int, default=1, help="Stride for tau in VAF")
    p.add_argument('--t-end-fit-ps', type=float, default=50, help="End of the fit in ps (required by SAMOS)")
    args = p.parse_args()

    files = os.listdir(args.data_dir)
    files_lower = [f.lower() for f in files]
    # QE detection
    is_qe = any(f.endswith('.pos') for f in files_lower) and any(f.endswith('.cel') for f in files_lower)
    # LAMMPS detection
    is_lammps = any(f.endswith('.dump') or f.endswith('.lammpstrj') or f.endswith('.extxyz') for f in files_lower)

    if is_qe:
        # Find files
        pos_file = glob.glob(os.path.join(args.data_dir, '*.pos'))[0]
        cel_file = glob.glob(os.path.join(args.data_dir, '*.cel'))[0]
        in_file = glob.glob(os.path.join(args.data_dir, '*.in'))[0]
        evp_file = glob.glob(os.path.join(args.data_dir, '*.evp'))[0] if glob.glob(os.path.join(args.data_dir, '*.evp')) else None
        # Build trajectory with velocities
        frames, dt = build_trajectory(
            in_file, pos_file, cel_file,
            evp_file=evp_file,
            start=args.start,
            nframes=args.nframes,
            stride=args.stride,
            time_interval=args.time_interval
        )

        # Compute and plot VAF for each requested element
        for elem in args.element:
            traj = Trajectory.from_atoms(frames)
            traj.set_attr('timestep_fs', dt * 1000.0)
            da = DynamicsAnalyzer()
            da.set_trajectories(traj)
            ts = da.get_vaf(
                integration='trapezoid',
                species_of_interest=[elem],
                nr_of_blocks=args.blocks,
                stepsize_t=args.stepsize_t,
                stepsize_tau=args.stepsize_tau,
                t_start_fit_ps=args.t_start_fit_ps,            
                t_end_fit_ps=args.t_end_fit_ps
            )

            # Plot VAF for this element
            plot_vaf_isotropic(ts)
            plt.xlim(0, args.t_end_fit_ps * 1000)  # fs, matches process file
            save_path = f'{args.out_prefix}_{elem}_vaf_upto_{int(args.t_end_fit_ps)}psframes.png'
            print(f"Saving VAF plot to: {os.path.abspath(save_path)}")
            plt.savefig(save_path)
            plt.show()
            plt.close()

    elif is_lammps:
        # Find LAMMPS file
        lammps_file = None
        for ext in ('*.dump', '*.lammpstrj', '*.extxyz'):
            found = glob.glob(os.path.join(args.data_dir, ext))
            if found:
                lammps_file = found[0]
                break
        if not lammps_file:
            raise RuntimeError("No LAMMPS trajectory file found in data-dir.")

        element_map = {}
        if args.element_mapping:
            for mapping in args.element_mapping:
                type_id, element = mapping.split(':')
                element_map[int(type_id)] = element
        # Read LAMMPS trajectory
        pos_full, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array = inp.read_lammps_trajectory(
            lammps_file,
            elements=args.lammps_elements,
            timestep=args.lammps_timestep,
            Conv_factor=1.0,
            element_mapping=element_map if element_map else None,
            export_verification=False,
            show_recommendations=False
        )
        # Build velocities (finite diff)
        pos_arr = np.transpose(pos_full, (1, 0, 2))  # (frames, atoms, 3)
        dt = dt_full[0] if dt_full is not None and len(dt_full) > 0 else (args.lammps_timestep or 1.0)
        t_list = t_full
        vel_arr = finite_diff_velocities(pos_arr, t_list)
        # Build ASE Atoms frames
        frames = []
        for i, (coords, cell) in enumerate(zip(pos_arr, cell_param_full)):
            a = Atoms(symbols=inp_array, positions=coords, cell=cell.reshape(3,3), pbc=True)
            a.set_velocities(vel_arr[i])
            frames.append(a)
        # Compute and plot VAF for each requested element
        for elem in args.element:
            traj = Trajectory.from_atoms(frames)
            traj.set_attr('timestep_fs', dt * 1000.0)
            da = DynamicsAnalyzer()
            da.set_trajectories(traj)
            ts = da.get_vaf(
                integration='trapezoid',
                species_of_interest=[elem],
                nr_of_blocks=args.blocks,
                stepsize_t=args.stepsize_t,
                stepsize_tau=args.stepsize_tau,
                t_start_fit_ps=args.t_start_fit_ps,            
                t_end_fit_ps=args.t_end_fit_ps
            )
            plot_vaf_isotropic(ts)
            plt.xlim(0, args.t_end_fit_ps * 1000)
            save_path = f'{args.out_prefix}_{elem}_vaf_upto_{int(args.t_end_fit_ps)}psframes.png'
            print(f"Saving VAF plot to: {os.path.abspath(save_path)}")
            plt.savefig(save_path)
            plt.show()
            plt.close()
    
    else:
        raise RuntimeError("Could not detect QE or LAMMPS trajectory files in data-dir.")