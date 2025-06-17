#!/usr/bin/env python3
"""
compute_vaf.py

Read QE .in/.pos/.cel[/.evp], build ASE+Velocities, compute VAF with SAMOS, and plot.
"""
import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from ase import Atoms
from samos.trajectory import Trajectory
from samos.analysis.dynamics import DynamicsAnalyzer

# point at your input_reader
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'target')))
from target import input_reader as inp

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
    """Map element symbols to atomic numbers."""
    try:
        return [element_to_atomic_number[s] for s in symbols]
    except KeyError as e:
        raise ValueError(f"Unknown element symbol: {e}") from e

def read_cel(cel_file):
    lines = open(cel_file).read().splitlines()
    return [lines[i:i+4] for i in range(0, len(lines), 4)]

def read_pos(pos_file, natoms):
    lines = open(pos_file).read().splitlines()
    return [lines[i:i+natoms+1] for i in range(0, len(lines), natoms+1)]

def read_evp(evp_file):
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
    pos_arr: (Nframes, natoms, 3)
    times:   list of length Nframes (ps)
    returns velocities array same shape
    """
    N, M, _ = pos_arr.shape
    v = np.zeros_like(pos_arr)
    # assume uniform timesteps, compute dt0 from first two entries
    if len(times) >= 2:
        dt0 = times[1] - times[0]
    else:
        dt0 = 1.0
    # central differences
    for i in range(1, N-1):
        v[i] = (pos_arr[i+1] - pos_arr[i-1])/(2*dt0)
    # endpoints: forward/backward
    v[0]   = (pos_arr[1] - pos_arr[0])  / dt0
    v[-1]  = (pos_arr[-1] - pos_arr[-2])/dt0
    return v

def build_trajectory(in_file, pos_file, cel_file,
                     evp_file=None,
                     start=0.0, nframes=0, stride=1,
                     time_interval=0.00193511):
    """
    Returns list of ASE.Atoms with velocities set.
    """
    BOHR2ANG = 0.529177249
    # 1) read species symbols
    syms = inp.read_ion_file(in_file)
    # 2) convert to atomic numbers locally
    sp   = convert_symbols_to_atomic_numbers(syms)

    # 3) read cell & pos chunks
    cel_chunks = read_cel(cel_file)
    pos_chunks = read_pos(pos_file, len(syms))
    # 4) optional times from .evp, else use uniform spacing
    times_full = read_evp(evp_file) if evp_file else []
    total = min(len(cel_chunks), len(pos_chunks))

    # determine start/end indices
    if len(times_full) >= 2:
        # slice out only the frames we will use
        dt = times_full[1] - times_full[0]
        # find start_idx based on time
        start_idx = int(start/dt)
        time_list = times_full[start_idx:total:stride]
    else:
        dt = time_interval
        start_idx = int(start/dt)
        time_list = [i * dt for i in range(start_idx, total, stride)]

    start_idx = max(0, min(start_idx, total-1))
    end_idx   = total if nframes<=0 else min(start_idx+nframes, total)

    # collect positions & cells first
    pos_list = []; cell_list = []
    for i in range(start_idx, end_idx, stride):
        block_c = cel_chunks[i]
        cell = np.loadtxt(block_c[1:], dtype=float).T * BOHR2ANG
        block_p = pos_chunks[i]
        coords = np.loadtxt(block_p[1:], dtype=float) * BOHR2ANG
        pos_list.append(coords)
        cell_list.append(cell)
    pos_arr = np.array(pos_list)  # (Nf, nat, 3)
    # build velocity array with a valid time_list (no None)
    vel_arr = finite_diff_velocities(pos_arr, time_list)

    frames = []
    for i, (coords, cell) in enumerate(zip(pos_arr, cell_list)):
        a = Atoms(numbers=sp, positions=coords, cell=cell, pbc=True)
        a.set_velocities(vel_arr[i])
        frames.append(a)
    print(f"Built {len(frames)} frames with velocities.")
    return frames, dt

def compute_vaf(frames, dt, element, blocks, prefix, in_file,
                t_start_dt=0, stepsize_t=1, stepsize_tau=1, t_end_fit_ps=10):
    """
    element: symbol (e.g. 'Li')  → pick indices in trajectory
    blocks:  number of statistical blocks
    """
    traj = Trajectory.from_atoms(frames)
    traj.set_attr('timestep_fs', dt * 1000.0)
    da = DynamicsAnalyzer()
    da.set_trajectories(traj)
    # map species→indices
    all_syms = inp.read_ion_file(in_file)
    indices = [i for i, s in enumerate(all_syms) if s == element]
    # get vaf
    ts = da.get_vaf(
        integration='trapezoid',
        species_of_interest=[element],
        nr_of_blocks=blocks,
        t_start_dt=t_start_dt,
        stepsize_t=stepsize_t,
        stepsize_tau=stepsize_tau,
        t_end_fit_ps=t_end_fit_ps
    )
    # extract mean arrays
    key_mean = f"vaf_isotropic_{element}_mean"
    key_int  = f"vaf_integral_isotropic_{element}_mean"
    vaf  = ts.get_array(key_mean)
    vint = ts.get_array(key_int)
    t = np.arange(len(vaf)) * dt
    # write to disk
    out_vaf = f"{prefix}_{element}_vaf.dat"
    out_int = f"{prefix}_{element}_vint.dat"
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

def plot_vaf(t, vaf, vint, element, prefix):
    """
    Plot VAF and integrated VAF vs time, with units in cm²/s² and cm²/s.
    """
    # Convert units
    t_s = t * 1e-12  # ps to s
    vaf_cms2 = vaf * 1e8  # Å²/ps² to cm²/s²
    vint_cms = vint * 1e4  # Å²/ps to cm²/s

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(t_s, vaf_cms2, '-', color='C0', label='VAF')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('VAF (cm²/s²)', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')

    ax2 = ax1.twinx()
    ax2.plot(t_s, vint_cms, '--', color='C1', label='∫VAF')
    ax2.set_ylabel('Integrated VAF (cm²/s)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    plt.title(f'{element} VAF and ∫VAF')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    out_png = f'{prefix}_{element}_vaf.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f'Plot saved to {out_png}')
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in_file',   required=True)
    p.add_argument('--pos_file',  required=True)
    p.add_argument('--cel_file',  required=True)
    p.add_argument('--evp_file',  default=None)
    p.add_argument('--element',   nargs='+', required=True,
                   help="Atom symbol(s) for VAF (e.g. Li Na)")
    p.add_argument('--start',   type=float, default=0.0,
                   help="Time (ps) to start analysis")
    p.add_argument('--nframes', type=int,   default=0,
                   help="Number of frames (0=all)")
    p.add_argument('--stride',  type=int,   default=1,
                   help = "Stride for frames (1=all, 2=every other, etc.)")
    p.add_argument('--blocks',  type=int,   default=1,   # changed from 4 to 1
                   help="Number of blocks for error estimates")
    p.add_argument('--out_prefix', default='vaf',
                   help="Prefix for output files")
    p.add_argument('--time_interval', type=float, default=0.00193511,
                   help='Default time between frames (ps) if no .evp file')
    p.add_argument('--t_start_dt', type=int, default=0, help="Start frame index for VAF")
    p.add_argument('--stepsize_t', type=int, default=1, help="Stride for t in VAF")
    p.add_argument('--stepsize_tau', type=int, default=1, help="Stride for tau in VAF")
    p.add_argument('--t_end_fit_ps', type=float, default=50, help="End of the fit in ps (required by SAMOS)")  # changed from 10 to 50
    args = p.parse_args()

    frames, dt = build_trajectory(
        args.in_file, args.pos_file, args.cel_file,
        evp_file=args.evp_file,
        start=args.start,
        nframes=args.nframes,
        stride=args.stride,
        time_interval=args.time_interval
    )

    for elem in args.element:
        t, vaf, vint = compute_vaf(
            frames, dt, elem, args.blocks, args.out_prefix, args.in_file,
            t_start_dt=args.t_start_dt,
            stepsize_t=args.stepsize_t,
            stepsize_tau=args.stepsize_tau,
            t_end_fit_ps=args.t_end_fit_ps
        )
        plot_vaf(t, vaf, vint, elem, args.out_prefix)
