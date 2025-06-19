#!/usr/bin/env python3
"""
Compute and plot VDOS from .in/.pos/.cel[/.evp] using SAMOS.

This script reads Quantum ESPRESSO trajectory files, constructs ASE Atoms objects with velocities,
computes the vibrational density of states (VDOS) using the SAMOS library, and plots/saves the results.

Usage example:
    python vdos.py --in_file LiAlPS.in --pos_file LiAlPS.pos --cel_file LiAlPS.cel --elements Li Al P S
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from samos_modules.samos_trajectory import Trajectory
from samos_modules.samos_analysis import DynamicsAnalyzer
from samos_modules.samos_plotting import plot_power_spectrum

# Import your input_reader
sys.path.append(os.path.abspath(os.path.join(__file__, '..')))
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
    return [element_to_atomic_number[s] for s in symbols]

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
    if len(times) >= 2:
        dt0 = (times[1] - times[0])*1000
    else:
        dt0 = 1.0
    for i in range(1, N-1):
        v[i] = (pos_arr[i+1] - pos_arr[i-1])/(2*dt0)
    v[0]   = (pos_arr[1] - pos_arr[0])  / dt0
    v[-1]  = (pos_arr[-1] - pos_arr[-2])/dt0
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
        list: List of ASE Atoms objects with velocities.
    """
    BOHR2ANG = 0.529177249
    syms = inp.read_ion_file(in_file)
    sp   = convert_symbols_to_atomic_numbers(syms)
    cel_chunks = read_cel(cel_file)
    pos_chunks = read_pos(pos_file, len(syms))
    times_full = read_evp(evp_file) if evp_file else []
    total = min(len(cel_chunks), len(pos_chunks))
    if len(times_full) >= 2:
        dt = times_full[1] - times_full[0]
        start_idx = int(start/dt)
        time_list = times_full[start_idx:total:stride]
    else:
        dt = time_interval
        start_idx = int(start/dt)
        time_list = [i * dt for i in range(start_idx, total, stride)]
    start_idx = max(0, min(start_idx, total-1))
    end_idx   = total if nframes<=0 else min(start_idx+nframes, total)
    pos_list = []; cell_list = []
    for i in range(start_idx, end_idx, stride):
        block_c = cel_chunks[i]
        cell = np.loadtxt(block_c[1:], dtype=float).T * BOHR2ANG
        block_p = pos_chunks[i]
        coords = np.loadtxt(block_p[1:], dtype=float) * BOHR2ANG
        pos_list.append(coords)
        cell_list.append(cell)
    pos_arr = np.array(pos_list)
    vel_arr = finite_diff_velocities(pos_arr, time_list)
    frames = []
    for i, (coords, cell) in enumerate(zip(pos_arr, cell_list)):
        a = Atoms(numbers=sp, positions=coords, cell=cell, pbc=True)
        a.set_velocities(vel_arr[i])
        frames.append(a)
    print(f"Built {len(frames)} frames with velocities.")
    return frames

def compute_plot_vdos(frames, prefix, elements=None, time_interval=0.00193511):
    """
    Compute and plot the vibrational density of states (VDOS) for the trajectory.

    Args:
        frames (list): List of ASE Atoms objects with velocities.
        prefix (str): Output file prefix for plots.
        elements (list or None): List of elements to plot VDOS for (default: Li, Al, P, S).
        time_interval (float): Time interval between frames in ps.

    Returns:
        None
    """
    traj = Trajectory.from_atoms(frames)
    # Set the timestep in femtoseconds for SAMOS
    traj._attrs['timestep_fs'] = time_interval * 1000
    da = DynamicsAnalyzer(trajectories=[traj])
    # Plot total VDOS using SAMOS's built-in plot
    res = da.get_power_spectrum()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    plot_power_spectrum(res, ax=ax1)
    ax1.set_xlabel('Frequency (THz)', fontsize=12)
    ax1.set_ylabel('Signal (A$^2$ fs$^{-1}$)', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.set_xlim(-2, 22)
    ax1.legend(fontsize=10)
    fig1.tight_layout()
    save_path1 = f'{prefix}_1.png'
    print(f"Saving VDOS plot to: {os.path.abspath(save_path1)}")
    plt.savefig(save_path1, dpi=300)
    plt.show()
    plt.close(fig1)

    # Custom plot for each element
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(
        plt.GridSpec(1, 1, left=0.15, top=0.95, bottom=0.2, right=0.95)[0]
    )
    color_cycle = plt.cm.tab10.colors
    for smoothing in [100]:
        res = da.get_power_spectrum(smothening=smoothing)
        freq_THz = res.get_array('frequency_0')
        if elements is None:
            elements = ['Li', 'Al', 'P', 'S']
        for idx, el in enumerate(elements):
            key = f'periodogram_{el}_mean'
            vdos = res.get_array(key)
            if vdos is not None:
                ax2.plot(freq_THz, vdos, label=f'{el} (smooth={smoothing})', color=color_cycle[idx % len(color_cycle)], linewidth=2)
    ax2.legend(fontsize=11, loc='upper right', frameon=True)
    ax2.set_yticks([])
    ax2.set_ylabel('Signal (A$^2$ fs$^{-1}$)', fontsize=12)
    ax2.set_xlabel('Frequency (THz)', fontsize=12)
    ax2.set_xlim(-2, 22)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    save_path2 = f'{prefix}_2.png'
    print(f"Saving custom VDOS plot to: {os.path.abspath(save_path2)}")
    plt.savefig(save_path2, dpi=300)
    plt.show()
    plt.close(fig2)

def main():
    """
    Main function for VDOS computation and plotting.
    Parses command-line arguments, builds trajectory, and generates VDOS plots.
    """
    parser = argparse.ArgumentParser(description="Compute and plot VDOS using SAMOS")
    parser.add_argument('--in_file', required=True)
    parser.add_argument('--pos_file', required=True)
    parser.add_argument('--cel_file', required=True)
    parser.add_argument('--evp_file', default=None)
    parser.add_argument('--out_prefix', default='vdos')
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--nframes', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--time_interval', type=float, default=0.00193511)
    parser.add_argument('--elements', nargs='+', default=None, help="Elements to plot (default: Li Al P S)")
    args = parser.parse_args()

    frames = build_trajectory(
        args.in_file, args.pos_file, args.cel_file,
        evp_file=args.evp_file,
        start=args.start,
        nframes=args.nframes,
        stride=args.stride,
        time_interval=args.time_interval
    )
    compute_plot_vdos(frames, args.out_prefix, elements=args.elements, time_interval=args.time_interval)

if __name__ == '__main__':
    main()