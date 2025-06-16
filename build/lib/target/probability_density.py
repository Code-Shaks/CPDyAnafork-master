"""
This script shows how to use a QE input file and a CP trajectory to plot the density of the Li-ions and of the rigid sublattice in LGPS.

NB: The files are not provided here; the goal is to showcase the code.

The positions of the QE input file and of the CP trajectory are aligned by removing the drift of the center of the rigid sublattice.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from samos.trajectory import Trajectory
from samos.utils.constants import *
from samos.analysis.get_gaussian_density import get_gaussian_density
from samos.lib.mdutils import recenter_positions
# from input_reader import inp

def load_start_configuration_from_qe_file(file_scf_input):
    """
    Read starting positions and nat from a QE input file
    """
    with open(file_scf_input) as finit:
        finit_lines = finit.readlines()
    iat = -1
    types = []
    start_positions = []
    for l in finit_lines:
        if len(l.split()) > 0:
            if l.split('=')[0].strip() == 'nat':
                snat = l.split('=')[1].strip()
                if snat[-1] == ',':
                    snat = snat[:-1]
                print(snat)
                nat = int(snat)
                print(f'Detected {nat} atoms')
            if l.split()[0] == 'ATOMIC_POSITIONS':
                iat += 1
                if len(l.split()) > 0 and l.split()[1].strip() in ['bohr', '(bohr)']:
                    print('Detected bohr we move to angstrom units')
                    factor = bohr_to_ang
                else:
                    print('We assume angstrom units')
                    factor = 1
            elif iat >= 0 and iat < nat:
                if not l[0] == '#':
                    split = l.split()
                    typ = split[0]
                    types.append(typ)
                    iat += 1
                    pos = np.array([float(split[1]) * factor, float(split[2]) * factor, float(split[3]) * factor])
                    start_positions.append(pos)
    start_positions = np.array(start_positions)
    assert len(types) == nat
    assert start_positions.shape == (nat, 3)
    return nat, start_positions

def load_trajectory_from_cp_file(file_traj, nat, format='bohr'):
    """
    Read trajectory from a trajectory output from cp.x
    """
    with open(file_traj) as ftraj:
        ftraj_lines = ftraj.readlines()
        nt = int(len(ftraj_lines) / (nat + 1)) if nat else 0  # Placeholder, requires nat
        positionsArr = np.zeros((nt, nat, 3), dtype=float) if nat else None
        for it in range(0, nt):
            every_nstep_pos = []
            for line in ftraj_lines[((nat + 1) * it) + 1:(nat + 1) * (it + 1)]:
                y = line.split()
                y = np.array(y, dtype=float)
                every_nstep_pos.append(y)
            if format == 'bohr':
                positionsArr[it, :, :] = np.array(every_nstep_pos, dtype=float) * bohr_to_ang
            else:
                positionsArr[it, :, :] = np.array(every_nstep_pos, dtype=float)
    delta = np.abs(float(ftraj_lines[0].split()[1]) - float(ftraj_lines[0 + nat + 1].split()[1]))
    delta = delta * 1000  # Convert to femtoseconds
    timestep = delta
    return positionsArr, timestep

def evaluate_com(conf, t, mode=None):
    """
    Evaluate the center of mass for the rigid sublattice
    """
    indices_rigid_z = np.array(
        list(t.get_indices_of_species('Al', start=0)) +
        list(t.get_indices_of_species('P', start=0)) +
        list(t.get_indices_of_species('S', start=0))
    )
    masses = t.atoms.get_masses()
    if mode == 'geometric':
        masses = [1.0] * len(masses)
    num = np.sum(np.array([conf[i, :] * masses[i] for i in indices_rigid_z]), axis=0)
    den = np.sum(np.array([masses[i] for i in indices_rigid_z]))
    return num / den