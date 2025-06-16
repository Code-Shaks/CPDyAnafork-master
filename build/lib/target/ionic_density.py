import numpy as np
from ase import Atoms
from samos.trajectory import Trajectory
from samos.utils.constants import bohr_to_ang
from samos.analysis.get_gaussian_density import get_gaussian_density

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
                nat = int(snat)
            if l.split()[0] == 'ATOMIC_POSITIONS':
                iat += 1
                if len(l.split()) > 0 and (l.split()[1].strip() == 'bohr' or l.split()[1].strip() == '(bohr)'):
                    factor = bohr_to_ang
                else:
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
    assert (len(types) == nat)
    assert (start_positions.shape == (nat, 3))
    return nat, start_positions

def load_trajectory_from_cp_file(file_traj, nat, format='bohr'):
    """
    Read trajectory from a trajectory output from cp.x
    """
    with open(file_traj) as ftraj:
        ftraj_lines = ftraj.readlines()
        nt = int(len(ftraj_lines) / (nat + 1))
        positionsArr = np.zeros((nt, nat, 3), dtype=float)
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
    delta = delta * 1000  # fs
    timestep = delta
    return positionsArr, timestep

def evaluate_com(conf, t, mode=None):
    indices_rigid_z = np.array(list(t.get_indices_of_species('Ge', start=0)) +
                               list(t.get_indices_of_species('P', start=0)) +
                               list(t.get_indices_of_species('S', start=0)))
    masses = t.atoms.get_masses()
    if mode == 'geometric':
        masses = [1.0] * len(masses)
    num = np.sum(np.array([conf[i, :] * masses[i] for i in indices_rigid_z]), axis=0)
    den = np.sum(np.array([masses[i] for i in indices_rigid_z]))
    return num / den

def main(
    file_scf_input,
    file_traj,
    formula='Li20Ge2P4S24',
    rigid_lattice=['Ge', 'P', 'S'],
    outputfile_li='out_li_all_test_2.xsf',
    outputfile_rigid='out_rigid_all_test_2.xsf'
):
    # Cell parameters (edit as needed)
    a = 16.4294 * bohr_to_ang
    b = a
    c = 16.4294 * 1.44919 * bohr_to_ang
    simulation_cell = [[a, 0, 0], [0, b, 0], [0, 0, c]]

    # Load initial configuration and trajectory
    nat, start_positions = load_start_configuration_from_qe_file(file_scf_input)
    positionsArr, timestep = load_trajectory_from_cp_file(file_traj, nat)

    # Initialize atoms and trajectory object
    atoms = Atoms(formula)
    atoms.set_positions(start_positions)
    atoms.cell = np.array(simulation_cell)
    t = Trajectory()
    t.set_timestep(timestep)
    t.set_atoms(atoms)
    t.set_positions(np.array(positionsArr))

    # Recenter so that the centre of the rigid sublattice is at zero
    print('We recenter so that the centre of the rigid sublattice is at zero')
    t.recenter(rigid_lattice, mode='geometric')

    # Evaluate center of mass of rigid sublattice
    com = evaluate_com(start_positions, t, mode='geometric')

    # Shift all positions so that the com of the rigid sublattice is aligned with the centre
    shift_all = True
    if shift_all:
        pos = t.get_positions()
        nstep, nat, ncoord = pos.shape
        for i in range(nstep):
            for j in range(nat):
                pos[i, j, :] = pos[i, j, :] + com
        t.set_positions(np.array(pos))

    # Get indices for Li and rigid lattice
    indices_li = t.get_indices_of_species('Li', start=1)
    indices_rigid = np.array(list(t.get_indices_of_species('Ge', start=1)) +
                             list(t.get_indices_of_species('P', start=1)) +
                             list(t.get_indices_of_species('S', start=1)))

    # Compute and save Gaussian densities
    get_gaussian_density(t, element=None,
                        outputfile=outputfile_li,
                        indices_i_care=indices_li,
                        indices_exclude_from_plot=[])

    get_gaussian_density(t, element=None,
                        outputfile=outputfile_rigid,
                        indices_i_care=indices_rigid,
                        indices_exclude_from_plot=[])

if __name__ == "__main__":
    # Example usage: update file paths as needed
    file_scf_input = 'LiAlPS.in'
    file_traj = 'LiAlPS.pos'
    main(file_scf_input, file_traj)