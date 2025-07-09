# -*- coding: utf-8 -*-

"""
Core Analysis Module for SAMOS

This module provides the main analysis classes and functions for the SAMOS
(Statistical Analysis of MOlecular Simulations) package.

:author: SAMOS Development Team
:date: 01-02-2024
"""

import numpy as np
import sys
import itertools
from target.utilities import get_terminal_width 
from scipy.stats import linregress
from scipy.signal import convolve
from target.trajectory import check_trajectory_compatibility, Trajectory
from target.utilities import AttributedArray
from target.utilities import InputError
from scipy.spatial.distance import cdist
from abc import ABCMeta, abstractmethod

bohr_to_ang = 0.52917720859

class TimeSeries(AttributedArray):
    """Time series analysis utilities."""
    pass


class DynamicsAnalyzer(object):
    """Comprehensive analyzer for particle dynamics in molecular simulations."""

    def __init__(self, **kwargs):
        self._species_of_interest = None
        self._verbosity = 1
        for key, val in kwargs.items():
            getattr(self, 'set_{}'.format(key))(val)

    def set_trajectories(self, trajectories):
        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]
        types, self._timestep_fs = check_trajectory_compatibility(
            trajectories)
        self._types = np.array(types)
        self._trajectories = trajectories

    def set_species_of_interest(self, species_of_interest):
        if isinstance(species_of_interest, str):
            self._species_of_interest = [species_of_interest]
        elif isinstance(species_of_interest, (tuple, set, list)):
            self._species_of_interest = list(species_of_interest)
        else:
            raise TypeError(
                'Species of interest has to be a list of'
                ' strings with the atomic symbol')

    def set_verbosity(self, verbosity):
        if not isinstance(verbosity, int):
            raise TypeError('Verbosity is an integer')
        self._verbosity = verbosity

    def get_species_of_interest(self):
        types = self._types
        if self._species_of_interest is None:
            return sorted(set(types))
        else:
            return self._species_of_interest

    def _get_running_params(self, timestep_fs, **kwargs):
        species_of_interest = kwargs.pop(
            'species_of_interest', self.get_species_of_interest())

        stepsize_t = int(kwargs.pop('stepsize_t', 1))
        stepsize_tau = int(kwargs.pop('stepsize_tau', 1))

        keywords_provided = list(kwargs.keys())
        for mutually_exclusive_keys in (
                ('t_start_fs', 't_start_ps', 't_start_dt'),
                ('t_end_fs', 't_end_ps', 't_end_dt'),
                ('block_length_fs', 'block_length_ps',
                 'block_length_dt', 'nr_of_blocks'),
                ('t_start_fit_fs', 't_start_fit_ps', 't_start_fit_dt'),
                ('t_end_fit_fs', 't_end_fit_ps', 't_end_fit_dt'),
                ('t_long_end_fs', 't_long_end_ps',
                 't_long_end_dt', 't_long_factor'),
        ):
            keys_provided_this_group = [
                k for k in mutually_exclusive_keys if k in keywords_provided]
            if len(keys_provided_this_group) > 1:
                raise InputError(
                    'This keywords are mutually exclusive: {}'.format(
                        ', '.join(keys_provided_this_group)))

        if 't_start_fit_fs' in keywords_provided:
            arg = kwargs.pop('t_start_fit_fs')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_start_fit_dt = np.rint(
                    np.array(arg, dtype=float) / timestep_fs).astype(int)
            else:
                t_start_fit_dt = int(float(arg) / timestep_fs)
        elif 't_start_fit_ps' in keywords_provided:
            arg = kwargs.pop('t_start_fit_ps')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_start_fit_dt = np.rint(
                    1000 * np.array(arg, dtype=float) / timestep_fs
                ).astype(int)
            else:
                t_start_fit_dt = int(1000 * float(arg) / timestep_fs)
        elif 't_start_fit_dt' in keywords_provided:
            arg = kwargs.pop('t_start_fit_dt')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_start_fit_dt = np.array(arg, dtype=int)
            else:
                t_start_fit_dt = int(arg)
        else:
            t_start_fit_dt = 0

        if not np.all(np.array(t_start_fit_dt >= 0)):
            raise InputError('t_start_fit_dt is not positive or 0')

        if 't_end_fit_fs' in keywords_provided:
            arg = kwargs.pop('t_end_fit_fs')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_end_fit_dt = np.rint(
                    np.array(arg, dtype=float) / timestep_fs).astype(int)
            else:
                t_end_fit_dt = int(float(arg) / timestep_fs)
        elif 't_end_fit_ps' in keywords_provided:
            arg = kwargs.pop('t_end_fit_ps')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_end_fit_dt = np.rint(
                    1000 * np.array(arg, dtype=float)
                    / timestep_fs
                ).astype(int)
            else:
                t_end_fit_dt = int(1000 * float(arg) / timestep_fs)
        elif 't_end_fit_dt' in keywords_provided:
            arg = kwargs.pop('t_end_fit_dt')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_end_fit_dt = np.array(arg, dtype=int)
            else:
                t_end_fit_dt = int(arg)
        else:
            raise InputError('Provide a time to end fitting the time series')

        if not np.all(t_end_fit_dt > t_start_fit_dt):
            raise InputError('t_end_fit_dt must be larger than '
                             't_start_fit_dt')

        if not isinstance(t_start_fit_dt, int):
            if isinstance(t_end_fit_dt, int):
                raise InputError(
                    't_start_fit_dt and t_end_fit_dt'
                    ' must be both integers or lists')
            elif (len(t_start_fit_dt) != len(t_end_fit_dt)):
                raise InputError(
                    't_start_fit_dt and t_end_fit_dt must '
                    'be of the same size')
        elif not isinstance(t_end_fit_dt, int):
            raise InputError(
                't_start_fit_dt and t_end_fit_dt must be both '
                'integers or lists')

        if 't_start_fs' in keywords_provided:
            t_start_dt = int(float(kwargs.pop('t_start_fs')) / timestep_fs)
        elif 't_start_ps' in keywords_provided:
            t_start_dt = int(
                1000 * float(kwargs.pop('t_start_ps')) / timestep_fs)
        elif 't_start_dt' in keywords_provided:
            t_start_dt = int(kwargs.pop('t_start_dt'))
        else:
            t_start_dt = 0

        if not (t_start_dt >= 0):
            raise InputError('t_start_dt is not positive or 0')
        if t_start_dt > 0:
            raise NotImplementedError('t_start has not been implemented yet!')

        if 't_end_fs' in keywords_provided:
            t_end_dt = int(float(kwargs.pop('t_end_fs')) / timestep_fs)
        elif 't_end_ps' in keywords_provided:
            t_end_dt = int(1000 * float(kwargs.pop('t_end_ps')) / timestep_fs)
        elif 't_end_dt' in keywords_provided:
            t_end_dt = int(kwargs.pop('t_end_dt'))
        else:
            t_end_dt = int(np.max(t_end_fit_dt))

        if not (t_end_dt > t_start_dt):
            raise InputError('t_end_dt is not larger than t_start_dt')
        if not (t_end_dt >= np.max(t_end_fit_dt)):
            raise InputError('t_end_dt must be larger than t_end_fit_dt')

        nr_of_t = (t_end_dt - t_start_dt) // stepsize_t

        if 'block_length_fs' in keywords_provided:
            block_length_dt = int(
                float(kwargs.pop('block_length_fs')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_ps' in keywords_provided:
            block_length_dt = int(
                1000 * float(kwargs.pop('block_length_ps')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_dt' in keywords_provided:
            block_length_dt = int(kwargs.pop('block_length_dt'))
            nr_of_blocks = None
        elif 'nr_of_blocks' in keywords_provided:
            nr_of_blocks = int(kwargs.pop('nr_of_blocks'))
            block_length_dt = None
        else:
            nr_of_blocks = 1
            block_length_dt = None

        do_com = kwargs.pop('do_com', False)

        do_long = kwargs.pop('do_long', False)
        if 't_long_end_fs' in keywords_provided:
            t_long_end_dt = int(
                float(kwargs.pop('t_long_end_fs')) / timestep_fs)
            t_long_factor = None
        elif 't_long_end_ps' in keywords_provided:
            t_long_end_dt = int(
                1000 * float(kwargs.pop('t_long_end_ps')) / timestep_fs)
            t_long_factor = None
        elif 't_long_end_dt' in keywords_provided:
            t_long_end_dt = int(kwargs.pop('t_long_end_dt'))
            t_long_factor = None
        elif 't_long_factor' in keywords_provided:
            t_long_factor = float(kwargs.pop('t_long_factor'))
            t_long_end_dt = None
        else:
            t_long_end_dt = None
            t_long_factor = None

        if kwargs:
            raise InputError(
                'Uncrecognized keywords: {}'.format(list(kwargs.keys())))

        return (species_of_interest, nr_of_blocks, t_start_dt,
                t_end_dt, t_start_fit_dt, t_end_fit_dt, nr_of_t,
                stepsize_t, stepsize_tau, block_length_dt, do_com,
                do_long, t_long_end_dt, t_long_factor)

    def get_vaf(self, arrayname=None, integration='trapezoid', **kwargs):
        from samos_modules.mdutils import (
            calculate_vaf_specific_atoms,
            get_com_velocities)
        try:
            timestep_fs = self._timestep_fs
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        (species_of_interest, nr_of_blocks, t_start_dt, t_end_dt,
         t_start_fit_dt, t_end_fit_dt, nr_of_t,
            stepsize_t, stepsize_tau, block_length_dt,
            do_com, do_long, t_long_end_dt,
            _) = self._get_running_params(timestep_fs, **kwargs)
        if do_long:
            raise NotImplementedError('Do_long is not implemented for VAF')

        vaf_time_series = TimeSeries()

        results_dict = dict()

        for atomic_species in species_of_interest:
            vaf_this_species = []
            vaf_integral_this_species = []
            fitted_means_of_integral = []

            for itraj, trajectory in enumerate(trajectories):
                if arrayname:
                    velocities = trajectory.get_array(arrayname)
                else:
                    velocities = trajectory.get_velocities()
                if do_com:
                    masses = self._atoms.get_masses()
                    factors = [1]*len(masses)
                    velocities = get_com_velocities(
                        velocities, masses, factors)
                    indices_of_interest = [1]
                    prefactor = len(trajectory.get_indices_of_species(
                        atomic_species, start=0))
                else:
                    indices_of_interest = trajectory.get_indices_of_species(
                        atomic_species, start=1)
                    prefactor = 1

                nstep, nat, _ = velocities.shape
                if nr_of_blocks > 0:
                    block_length_dt_this_traj = (
                        nstep - t_end_dt) // nr_of_blocks
                    nr_of_blocks_this_traj = nr_of_blocks
                elif block_length_dt > 0:
                    block_length_dt_this_traj = block_length_dt
                    nr_of_blocks_this_traj = (
                        nstep - t_end_dt) // block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_ft is '
                        'specified')

                nat_of_interest = len(indices_of_interest)

                if self._verbosity > 0:
                    print((
                        '\n    ! Calculating VAF for atomic species {} '
                        'in trajectory {}\n'
                        '      Structure contains {} atoms of type {}\n'
                        '      I will calculate {} block(s)'
                        ''.format(atomic_species, itraj,
                                  nat_of_interest, atomic_species,
                                  nr_of_blocks)
                    ))

                vaf, vaf_integral = calculate_vaf_specific_atoms(
                    velocities, indices_of_interest, stepsize_t, stepsize_tau,
                    nr_of_t, nr_of_blocks_this_traj, block_length_dt_this_traj,
                    timestep_fs*stepsize_t,
                    integration, nstep, nat, nat_of_interest)
                vaf_integral *= 0.1/3. * prefactor

                for iblock in range(nr_of_blocks_this_traj):
                    vaf_this_species.append(vaf[iblock])
                    vaf_integral_this_species.append(vaf_integral[iblock])
                    data_ = vaf_integral[
                        iblock,
                        t_start_fit_dt//stepsize_t:t_end_fit_dt//stepsize_t]
                    fitted_means_of_integral.append(data_.mean())

                vaf_time_series.set_array(
                    'vaf_isotropic_{}_{}'.format(atomic_species, itraj),
                    vaf)
                vaf_time_series.set_array(
                    'vaf_integral_isotropic_{}_{}'.format(
                        atomic_species, itraj),
                    vaf_integral)

            for arr, name in ((vaf_this_species, 'vaf_isotropic'),
                              (vaf_integral_this_species,
                               'vaf_integral_isotropic')):
                arr = np.array(arr)

                arr_mean = np.mean(arr, axis=0)
                arr_std = np.std(arr, axis=0)
                arr_sem = arr_std / np.sqrt(arr.shape[0] - 1)
                vaf_time_series.set_array(
                    '{}_{}_mean'.format(name, atomic_species),
                    arr_mean)
                vaf_time_series.set_array(
                    '{}_{}_std'.format(name, atomic_species),
                    arr_std)
                vaf_time_series.set_array(
                    '{}_{}_sem'.format(name, atomic_species),
                    arr_sem)

            fitted_means_of_integral = np.array(fitted_means_of_integral)
            results_dict[atomic_species] = dict(
                diffusion_mean_cm2_s=fitted_means_of_integral.mean(),
                diffusion_std_cm2_s=fitted_means_of_integral.std())

            results_dict[atomic_species]['diffusion_sem_cm2_s'] = (
                results_dict[atomic_species]['diffusion_std_cm2_s']
                / np.sqrt(len(fitted_means_of_integral) - 1))

        results_dict.update({
            't_start_fit_dt':   t_start_fit_dt,
            't_end_fit_dt':   t_end_fit_dt,
            't_start_dt':   t_start_dt,
            't_end_dt':   t_end_dt,
            'nr_of_trajectories':   len(trajectories),
            'stepsize_t':   stepsize_t,
            'species_of_interest':   species_of_interest,
            'timestep_fs':   timestep_fs,
            'nr_of_t':   nr_of_t, })

        for k, v in results_dict.items():
            vaf_time_series.set_attr(k, v)
        return vaf_time_series

    def get_kinetic_energies(self, stepsize=1, decompose_system=True,
                             decompose_atoms=False,
                             decompose_species=False):
        from target.utilities import amu_kg, kB

        try:
            timestep_fs = self._timestep_fs
            atoms = self._atoms
            masses = atoms.get_masses()
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        prefactor = amu_kg * 1e10 / kB

        if decompose_atoms and decompose_species:
            raise Exception('Cannot decompose atoms and decompose species')

        kinetic_energies_series = TimeSeries()
        kinetic_energies_series.set_attr('stepsize', stepsize)
        kinetic_energies_series.set_attr('timestep_fs', timestep_fs)

        for itraj, t in enumerate(trajectories):
            vel_array = t.get_velocities()
            nstep, nat, _ = vel_array.shape
            steps = list(range(0, nstep, stepsize))

            if decompose_system:
                kinE = np.zeros(len(steps))
                for istep0, istep in enumerate(steps):
                    for iat in range(nat):
                        for ipol in range(3):
                            kinE[istep0] += (
                                prefactor * masses[iat]
                                * vel_array[istep, iat, ipol]**2)
                kinE[:] /= nat*3
                kinetic_energies_series.set_array(
                    'system_kinetic_energy_{}'.format(itraj), kinE)
                kinetic_energies_series.set_attr(
                    'mean_system_kinetic_energy_{}'.format(itraj), kinE.mean())
            if decompose_species:
                species_of_interest = self.get_species_of_interest()
                ntyp = len(species_of_interest)
                steps = list(range(0, nstep, stepsize))
                kinE_species = np.zeros((len(steps), ntyp))
                for ityp, atomic_species in enumerate(species_of_interest):
                    indices_of_interest = t.get_indices_of_species(
                        atomic_species, start=0)
                    for istep0, istep in enumerate(steps):
                        for _, iat in enumerate(indices_of_interest):
                            for ipol in range(3):
                                kinE_species[istep0, ityp] += (
                                    prefactor * masses[iat] *
                                    vel_array[istep, iat, ipol]**2)

                    kinE_species[:, ityp] /= float(len(indices_of_interest)*3)

                kinetic_energies_series.set_array(
                    'species_kinetic_energy_{}'.format(itraj), kinE)
                kinetic_energies_series.set_attr(
                    'species_of_interest', species_of_interest)
                kinetic_energies_series.set_attr(
                    'mean_species_kinetic_energy_{}'.format(itraj),
                    kinE_species.mean(axis=0).tolist())

            if decompose_atoms:
                kinE = np.zeros((len(steps), nat))
                for istep0, istep in enumerate(steps):
                    for iat in range(nat):
                        for ipol in range(3):
                            kinE[istep0, iat] += prefactor * masses[iat] * \
                                vel_array[istep, iat, ipol]**2 / 3.

                kinetic_energies_series.set_array(
                    'atoms_kinetic_energy_{}'.format(itraj), kinE)
                kinetic_energies_series.set_attr(
                    'mean_atoms_kinetic_energy_{}'.format(itraj),
                    kinE.mean(axis=0).tolist())

        return kinetic_energies_series

    def get_power_spectrum(self, arrayname=None, **kwargs):
        from scipy import signal

        try:
            trajectories = self._trajectories
            timestep_fs = self._timestep_fs
            sampling_frequency_THz = 1e3 / timestep_fs
        except AttributeError as e:
            raise Exception(
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        keywords_provided = list(kwargs.keys())
        for mutually_exclusive_keys in (
                ('block_length_fs', 'block_length_ps',
                 'block_length_dt', 'nr_of_blocks'),):
            keys_provided_this_group = [k for k
                                        in mutually_exclusive_keys
                                        if k in keywords_provided]
            if len(keys_provided_this_group) > 1:
                raise InputError(
                    'This keywords are mutually exclusive: '
                    '{}'.format(', '.join(keys_provided_this_group)))
        if 'block_length_fs' in keywords_provided:
            block_length_dt = int(
                float(kwargs.pop('block_length_fs')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_ps' in keywords_provided:
            block_length_dt = int(
                1000*float(kwargs.pop('block_length_ps')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_dt' in keywords_provided:
            block_length_dt = int(kwargs.pop('block_length_dt'))
            nr_of_blocks = None
        elif 'nr_of_blocks' in keywords_provided:
            nr_of_blocks = kwargs.pop('nr_of_blocks')
            block_length_dt = None
        else:
            nr_of_blocks = 1
            block_length_dt = None
        species_of_interest = kwargs.pop(
            'species_of_interest', None) or self.get_species_of_interest()
        smothening = int(kwargs.pop('smothening', 1))
        if kwargs:
            raise InputError(
                'Uncrecognized keywords: {}'.format(list(kwargs.keys())))

        power_spectrum = TimeSeries()

        for index_of_species, atomic_species in enumerate(species_of_interest):
            periodogram_this_species = []

            for itraj, trajectory in enumerate(trajectories):
                if arrayname:
                    vel_array = trajectory.get_array(
                        arrayname)[:, trajectory.get_indices_of_species(
                            atomic_species, start=0), :]
                else:
                    vel_array = trajectory.get_velocities()[
                        :, trajectory.get_indices_of_species(atomic_species,
                                                             start=0), :]
                nstep, _, _ = vel_array.shape

                if nr_of_blocks > 0:
                    nr_of_blocks_this_traj = nr_of_blocks
                    split_number = nstep // nr_of_blocks_this_traj
                elif block_length_dt > 0:
                    nr_of_blocks_this_traj = nstep // block_length_dt
                    split_number = block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_ft '
                        'is specified')

                blocks = np.array(np.split(
                    vel_array[:nr_of_blocks_this_traj*split_number],
                    nr_of_blocks_this_traj, axis=0))
                nblocks = len(blocks)

                freq, pd = signal.periodogram(blocks,
                                              fs=sampling_frequency_THz,
                                              axis=1, return_onesided=True)
                pd_this_species_this_traj = pd.mean(axis=(2, 3))
                if smothening > 1:
                    kernel = np.ones((nblocks, smothening)) / smothening
                    pd_this_species_this_traj = convolve(
                        pd_this_species_this_traj,
                        kernel, mode='same')

                power_spectrum.set_array('periodogram_{}_{}'.format(
                    atomic_species, itraj), pd_this_species_this_traj)
                if not index_of_species:
                    power_spectrum.set_array(
                        'frequency_{}'.format(itraj), freq)
                for block in pd_this_species_this_traj:
                    periodogram_this_species.append(block)
            try:
                length_last_block = len(block)
                for pd in periodogram_this_species:
                    if len(pd) != length_last_block:
                        raise Exception(
                            'Cannot calculate mean signal '
                            ' because of different lengths')
                periodogram_this_species = np.array(
                    periodogram_this_species)
                power_spectrum.set_array('periodogram_{}_mean'.format(
                    atomic_species), periodogram_this_species.mean(axis=0))
                std = periodogram_this_species.std(axis=0)
                power_spectrum.set_array(
                    'periodogram_{}_std'.format(atomic_species), std)
                power_spectrum.set_array('periodogram_{}_sem'.format(
                    atomic_species),
                    std/np.sqrt(len(periodogram_this_species)-1))
            except Exception as e:
                print(e)

        for k, v in (('species_of_interest', species_of_interest),
                     ('nr_of_trajectories', len(trajectories)),):
            power_spectrum.set_attr(k, v)
        return power_spectrum

def write_xsf_header(
        atoms, positions, cell, data,
        vals_per_line=6, outfilename=None, **kwargs):
    if isinstance(outfilename, str):
        f = open(outfilename, 'w')
    elif outfilename is None:
        f = sys.stdout
    else:
        raise Exception('No file')
    if data is not None:
        xdim, ydim, zdim = data.shape
    else:
        xdim = kwargs.get('xdim')
        ydim = kwargs.get('ydim')
        zdim = kwargs.get('zdim')
    f.write(' CRYSTAL\n PRIMVEC\n')
    for row in cell:
        f.write('    {}\n'.format('    '.join(
            ['{:.9f}'.format(r) for r in row])))
    f.write('PRIMCOORD\n       {}        1\n'.format(len(atoms)))
    for atom, pos in zip(atoms, positions):
        f.write('{:<3}     {}\n'.format(
            atom, '   '.join(['{:.9f}'.format(v) for v in pos])))

    f.write("""BEGIN_BLOCK_DATAGRID_3D
3D_PWSCF
DATAGRID_3D_UNKNOWN
        {}         {}         {}
  0.000000  0.000000  0.000000
""".format(*[i+1 for i in (xdim, ydim, zdim)]))

    for row in cell:
        f.write('    {}\n'.format('    '.join(
            ['{:.9f}'.format(item) for item in row])))

    if data is not None:
        col = 1
        for z in range(zdim+1):
            for y in range(ydim+1):
                for x in range(xdim+1):
                    f.write('  {:0.4E}'.format(
                        data[x % xdim, y % ydim, z % zdim]))
                    if col < vals_per_line:
                        col += 1
                    else:
                        f.write('\n')
                        col = 1
        if col:
            f.write('\n')
        f.write('END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n')
    f.close()


def get_gaussian_density(trajectory, element=None, outputfile='out.xsf',
                         sigma=0.3, n_sigma=3.0, density=0.1,
                         istart=1, istop=None, stepsize=1,
                         indices=None, indices_exclude_from_plot=None):
    from samos_modules.gaussian_density import make_gaussian_density

    cell = trajectory.cell
    positions = trajectory.get_positions()

    nstep, nat, _ = positions.shape
    symbols = trajectory.atoms.get_chemical_symbols()
    starting_pos = trajectory.atoms.get_positions()

    if not outputfile.endswith('.xsf'):
        outputfile += '.xsf'

    if indices is None:
        if element:
            indices = trajectory.get_indices_of_species(
                element, start=1)
        else:
            indices = np.array(list(range(1, nat+1)))

    if not len(indices):
        raise Exception(
            'Element {} not found in symbols {}'.format(element, symbols))

    nat_this_species = len(indices)

    if istop is None:
        istop = nstep

    try:
        termwidth = get_terminal_width()
        pbar_frequency = int((istop - istart) / termwidth)
    except Exception as e:
        pbar_frequency = int((istop - istart) / 30)

    pbar_frequency = max([pbar_frequency, 1])
    a, b, c = [np.linalg.norm(cell[i]) for i in range(3)]
    n1, n2, n3 = [int(celldim/density)+1 for celldim in (a, b, c)]

    if indices_exclude_from_plot is None:
        indices_exclude_from_plot = indices
    
    write_xsf_header(
        [s for i, s in enumerate(symbols, start=1)
         if i not in indices_exclude_from_plot],
        [p for i, p in enumerate(starting_pos, start=1)
         if i not in indices_exclude_from_plot],
        cell, None, outfilename=outputfile, xdim=n1, ydim=n2, zdim=n3)

    S = np.matrix(np.diag([1, 1, 1, -(sigma*n_sigma/density)**2]))
    cellT = cell.T
    cellTI = np.matrix(cellT).I
    M = np.matrix(np.r_[np.c_[np.matrix(cellTI), np.zeros(3)], [[0, 0, 0, 1]]])
    R = M * S.I * M.T
    b1 = int(np.abs(
        (R[0, 3] - np.sqrt(R[0, 3]**2 - R[0, 0]*R[3, 3]))
        / R[3, 3]) / density) + 1
    b2 = int(
        abs((R[1, 3] - np.sqrt(R[1, 3]**2 - R[1, 1]*R[3, 3]))
            / R[3, 3]) / density)+1
    b3 = int(
        abs((R[2, 3] - np.sqrt(R[2, 3]**2 - R[2, 2]*R[3, 3]))
            / R[3, 3]) / density)+1

    make_gaussian_density(
        positions, outputfile, n1, n2, n3, b1, b2, b3, istart, istop, stepsize,
        sigma, cell, cellTI, indices, pbar_frequency, nstep, nat,
        nat_this_species
    )

class BaseAnalyzer(object, metaclass=ABCMeta):
    """Abstract base class for molecular dynamics analysis tools."""
    
    def __init__(self, **kwargs):
        for key, val in list(kwargs.items()):
            getattr(self, 'set_{}'.format(key))(val)

    def set_trajectory(self, trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError(
                'You need to pass a {} as trajectory'.format(Trajectory.__name__))
        self._trajectory = trajectory

    @abstractmethod
    def run(*args, **kwargs):
        pass