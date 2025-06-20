# -*- coding: utf-8 -*-

"""
Core Analysis Module for SAMOS

This module provides the main analysis classes and functions for the SAMOS
(Statistical Analysis of MOlecular Simulations) package. It implements
high-performance algorithms for common molecular dynamics analysis tasks.

Key Features:
    - Mean Square Displacement (MSD) calculations
    - Radial Distribution Functions (RDF)
    - Velocity Autocorrelation Functions (VAF)
    - Vibrational Density of States (VDOS)
    - Angular momentum analysis
    - Time series analysis utilities

The module is designed for efficiency with large trajectory datasets and
provides both object-oriented and functional interfaces for flexibility.

Classes:
    - ``BaseAnalyzer``: Abstract base class for analysis tools
    - ``DynamicsAnalyzer``: Comprehensive dynamics analysis
    - ``RDF``: Radial distribution function calculator
    - ``AngularSpectrum``: Angular momentum analysis
    - ``TimeSeries``: Time series analysis utilities

Functions:
    - ``util_msd``: High-level MSD calculation interface
    - ``util_rdf_and_plot``: RDF calculation with plotting
    - ``get_gaussian_density``: Gaussian density field generation

:author: SAMOS Development Team
:date: 01-02-2024
"""

import numpy as np
import sys
import itertools
from samos_modules.samos_utils import get_terminal_width 
from scipy.stats import linregress
from scipy.signal import convolve
from samos_modules.samos_trajectory import check_trajectory_compatibility, Trajectory
from samos_modules.samos_utils import AttributedArray
from samos_modules.samos_utils import InputError
from scipy.spatial.distance import cdist
from abc import ABCMeta, abstractmethod

bohr_to_ang = 0.52917720859

class TimeSeries(AttributedArray):
    """
    Time series analysis utilities.
    
    This class provides tools for analyzing time-dependent properties
    including autocorrelation functions, power spectra, and statistical
    analysis.
    """
    pass


class DynamicsAnalyzer(object):
    """
    Comprehensive analyzer for particle dynamics in molecular simulations.
    
    This class provides a unified interface for calculating various dynamic
    properties including MSD, VAF, and related quantities. It handles multiple
    species analysis and provides statistical error estimation.
    
    Attributes:
        species_of_interest (list): Chemical species to analyze.
        verbosity_level (int): Level of output detail.
        trajectories (list): Multiple trajectory datasets.
        
    Methods:
        get_msd: Calculate mean square displacement
        get_vaf: Calculate velocity autocorrelation function
        get_power_spectrum: Calculate power spectrum from VAF
        get_kinetic_energies: Calculate kinetic energy time series
        set_species_of_interest: Set target species for analysis
    """

    def __init__(self, **kwargs):
        """
        Initialize the dynamics analyzer.

        Parameters
        ----------
        trajectory (list, optional): List of ASE Atoms objects.
        species_of_interest (list, optional): Species to analyze.
        verbose (bool, optional): Enable verbose output. Default is ``False``.
        """
        self._species_of_interest = None
        self._verbosity = 1
        for key, val in kwargs.items():
            getattr(self, 'set_{}'.format(key))(val)

    def set_trajectories(self, trajectories):
        """
        Set multiple trajectories for ensemble analysis.

        Parameters
        ----------
        trajectories (list): List of trajectory datasets.
        """
        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]
        # I check the compatibility. Implicitly,
        # also checks if trajectories are valid instances.
        types, self._timestep_fs = check_trajectory_compatibility(
            trajectories)
        self._types = np.array(types)
        # Setting as attribute of self for analysis
        self._trajectories = trajectories

    def set_species_of_interest(self, species_of_interest):
        """
        Set the chemical species to be analyzed.

        Parameters
        ----------
        species_of_interest (list or str): List of chemical symbols (e.g., ['Li', 'Na']).

        Raises
        ------
        TypeError
            If species_of_interest is not a string or list/tuple/set of strings.
        """
        if isinstance(species_of_interest, str):
            self._species_of_interest = [species_of_interest]
        elif isinstance(species_of_interest, (tuple, set, list)):
            self._species_of_interest = list(species_of_interest)
        else:
            raise TypeError(
                'Species of interest has to be a list of'
                ' strings with the atomic symbol')

    def set_verbosity(self, verbosity):
        """
        Set the verbosity level for output.

        Parameters
        ----------
        verbosity (int): Verbosity level (0=quiet, 1=normal, 2=verbose).

        Raises
        ------
        TypeError
            If verbosity is not an integer.
        """
        if not isinstance(verbosity, int):
            raise TypeError('Verbosity is an integer')
        self._verbosity = verbosity

    def get_species_of_interest(self):
        """
        Get the current list of species being analyzed.

        Returns
        -------
        list
            Current species of interest. If not set, returns sorted unique types from trajectories.
        """
        # Also a good way to check if atoms have been set
        types = self._types
        if self._species_of_interest is None:
            return sorted(set(types))
        else:
            return self._species_of_interest

    def _get_running_params(self, timestep_fs, **kwargs):
        """
        Utility function to parse and validate parameters for time series calculations.

        Parameters
        ----------
        timestep_fs (float): Timestep in femtoseconds.
        **kwargs
            Additional parameters for time series calculation:
            - species_of_interest (list, optional): Species to calculate.
            - stepsize_t (int, optional): Outer-loop step size for trajectory sampling. Default is 1.
            - stepsize_tau (int, optional): Inner-loop step size for sliding window. Default is 1.
            - t_start_fs (float, optional): Minimum sliding window value in femtoseconds.
            - t_start_ps (float, optional): Minimum sliding window value in picoseconds.
            - t_start_dt (int, optional): Minimum sliding window value in timesteps.
            - t_end_fs (float, optional): Maximum sliding window value in femtoseconds.
            - t_end_ps (float, optional): Maximum sliding window value in picoseconds.
            - t_end_dt (int, optional): Maximum sliding window value in timesteps.
            - block_length_fs (float, optional): Block size for trajectory blocking in femtoseconds.
            - block_length_ps (float, optional): Block size for trajectory blocking in picoseconds.
            - block_length_dt (int, optional): Block size for trajectory blocking in timesteps.
            - nr_of_blocks (int, optional): Number of blocks for trajectory splitting. Default is 1.
            - t_start_fit_fs (float, optional): Start time for fitting in femtoseconds.
            - t_start_fit_ps (float, optional): Start time for fitting in picoseconds.
            - t_start_fit_dt (int, optional): Start time for fitting in timesteps.
            - t_end_fit_fs (float, optional): End time for fitting in femtoseconds.
            - t_end_fit_ps (float, optional): End time for fitting in picoseconds.
            - t_end_fit_dt (int, optional): End time for fitting in timesteps.
            - do_long (bool, optional): Perform maximum-statistics calculation. Default is ``False``.
            - t_long_end_fs (float, optional): Maximum sliding window in femtoseconds for max-stats.
            - t_long_end_ps (float, optional): Maximum sliding window in picoseconds for max-stats.
            - t_long_end_dt (int, optional): Maximum sliding window in timesteps for max-stats.
            - t_long_factor (float, optional): Factor for max-stats trajectory length.
            - do_com (bool, optional): Calculate center-of-mass diffusion. Default is ``False``.

        Returns
        -------
        tuple
            Parsed parameters for time series calculation.

        Raises
        ------
        InputError
            If mutually exclusive keywords are provided or parameters are invalid.
        NotImplementedError
            If certain features (e.g., t_start > 0) are not implemented.
        """

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
            # By default I create the time series from the start
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

        # The number of timesteps I will calculate:
        nr_of_t = (t_end_dt - t_start_dt) // stepsize_t

        # Checking if I have to partition the trajectory into blocks
        # (By default just 1 block)
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

        # Asking whether to calculate COM diffusion
        do_com = kwargs.pop('do_com', False)

        # Asking whether to calculate for every trajectory
        # a time series with maximal statistics:
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
            t_long_end_dt = None  # will be adapted to trajectory length!!
            t_long_factor = None  # will be adapted to trajectory length!!

        # Irrespective of whether do_long is false or true,
        # I see whether factors are calculated:

        if kwargs:
            raise InputError(
                'Uncrecognized keywords: {}'.format(list(kwargs.keys())))

        return (species_of_interest, nr_of_blocks, t_start_dt,
                t_end_dt, t_start_fit_dt, t_end_fit_dt, nr_of_t,
                stepsize_t, stepsize_tau, block_length_dt, do_com,
                do_long, t_long_end_dt, t_long_factor)

    def get_msd(self, decomposed=False, atom_indices=None, **kwargs):
        """
        Calculate Mean Square Displacement for specified species.

        This method computes MSD using an efficient algorithm that handles
        large trajectories. It supports both single and multiple species
        analysis with automatic error estimation.

        Parameters
        ----------
        decomposed : bool, optional
            Whether to decompose MSD by directions. Default is ``False``.
        atom_indices : list, optional
            Specific atom indices to analyze. If ``None``, all atoms of the
            specified species are included.
        **kwargs
            Additional parameters for trajectory sampling and fitting:
            - stepsize_t (int, optional): Step size for trajectory sampling (default: 1).
              Use larger values for faster computation on large trajectories.
            - species_of_interest (list, optional): List of species to analyze.
              If ``None``, uses ``self.species_of_interest``.
            - t_start_fit_ps (float, optional): Start time for diffusion coefficient fit in picoseconds.
            - t_end_fit_ps (float, optional): End time for diffusion coefficient fit in picoseconds.
            - nr_of_blocks (int, optional): Number of blocks for statistical analysis.

        Returns
        -------
        TimeSeries
            MSD data containing:
            - t_list_fs: Time array in femtoseconds.
            - msd_isotropic_X_mean: Mean MSD values for species X.
            - msd_isotropic_X_std: Standard deviation of MSD for species X.
            - slope_msd_mean: Fitted slope values (proportional to diffusion coefficient).
            - diffusion_mean_cm2_s: Calculated diffusion coefficients in cm²/s.

        Raises
        ------
        ValueError
            If no trajectory is set or species list is empty.

        Examples
        --------
        >>> analyzer = DynamicsAnalyzer(trajectory, species_of_interest=['Li'])
        >>> msd_data = analyzer.get_msd(stepsize=2)
        >>> print(f"Li diffusion coefficient: {msd_data.get_attr('Li')['diffusion_mean_cm2_s']:.2e}")

        Note
        ----
        MSD is calculated using an efficient FFT-based algorithm for
        improved performance on large datasets.
        """
        from samos_modules.mdutils import (
            calculate_msd_specific_atoms,
            calculate_msd_specific_atoms_decompose_d,
            calculate_msd_specific_atoms_max_stats, get_com_positions)
        try:
            timestep_fs = self._timestep_fs
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        (
            species_of_interest, nr_of_blocks, t_start_dt, t_end_dt,
            t_start_fit_dt, t_end_fit_dt, nr_of_t, stepsize_t,
            stepsize_tau, block_length_dt, do_com, do_long, t_long_end_dt,
            t_long_factor) = self._get_running_params(timestep_fs, **kwargs)
        multiple_params_fit = not isinstance(t_start_fit_dt, int)

        msd = TimeSeries()
        # list of t at which MSD will be computed
        t_list_fs = timestep_fs * stepsize_t * \
            (t_start_dt + np.arange(nr_of_t))
        msd.set_array('t_list_fs', t_list_fs)

        results_dict = {atomic_species: {}
                        for atomic_species in species_of_interest}
        nr_of_t_long_list = []
        t_list_long_fs = []
        # Setting params for calculation of MSD and conductivity
        # Future: Maybe allow for element specific parameter settings?

        for atomic_species in species_of_interest:
            msd_this_species = []  # Here I collect the trajectories
            slopes = []  # That's where I collect slopes
            # for the final estimate of diffusion

            for itraj, trajectory in enumerate(trajectories):
                positions = trajectory.get_positions()
                if do_com:
                    # I replace the array positions with the COM!
                    # Getting the massesfor recentering
                    masses = self._atoms.get_masses()
                    factors = [1]*len(masses)
                    positions = get_com_positions(positions, masses, factors)
                    indices_of_interest = [1]
                    prefactor = len(trajectory.get_indices_of_species(
                        atomic_species, start=0))
                else:
                    indices_of_interest = trajectory.get_indices_of_species(
                        atomic_species, start=1)
                    prefactor = 1
                if atom_indices is not None:
                    indices_of_interest = [
                        i for i in indices_of_interest if i in atom_indices]

                # make blocks
                nstep, nat, _ = positions.shape
                if nr_of_blocks:
                    block_length_dt_this_traj = (
                        nstep - t_end_dt) // nr_of_blocks
                    nr_of_blocks_this_traj = nr_of_blocks
                elif block_length_dt > 0:
                    block_length_dt_this_traj = block_length_dt
                    nr_of_blocks_this_traj = (
                        nstep - t_end_dt) // block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_dt '
                        'was specified')
                if (
                    (nr_of_blocks_this_traj < 0)
                        or (block_length_dt_this_traj < 0)):
                    raise RuntimeError(
                        't_end_dt (or t_end_fit_dt) is bigger'
                        ' than the trajectory length')

                nat_of_interest = len(indices_of_interest)

                #
                # compute MSD (using defined blocks, nstep, nr_of_t, ...)
                if self._verbosity > 0:
                    print((
                        '\n    ! Calculating MSD for atomic species {} '
                        'in trajectory {}\n'
                        '      Structure contains {} atoms of type {}\n'
                        '      I will calculate {} block(s) of size {} '
                        '({} ps)\n'
                        '      I will fit from {} ({} ps) to {} ({} ps)\n'
                        '      Outer stepsize is {}, inner is {}\n'
                        ''.format(
                            atomic_species, itraj, nat_of_interest,
                            atomic_species,
                            nr_of_blocks_this_traj,
                            block_length_dt_this_traj,
                            block_length_dt_this_traj * timestep_fs / 1e3,
                            t_start_fit_dt, t_start_fit_dt * timestep_fs / 1e3,
                            t_end_fit_dt, t_end_fit_dt * timestep_fs / 1e3,
                            stepsize_t, stepsize_tau)
                    ))
                if decomposed:
                    msd_this_species_this_traj = (
                        prefactor * calculate_msd_specific_atoms_decompose_d(
                            positions, indices_of_interest, stepsize_t,
                            stepsize_tau, block_length_dt_this_traj,
                            nr_of_blocks_this_traj,
                            nr_of_t, nstep, nat, nat_of_interest)
                    )
                else:
                    msd_this_species_this_traj = (
                        prefactor * calculate_msd_specific_atoms(
                            positions, indices_of_interest, stepsize_t,
                            stepsize_tau, block_length_dt_this_traj,
                            nr_of_blocks_this_traj, nr_of_t, nstep,
                            nat, nat_of_interest)
                    )
                if self._verbosity > 0:
                    print('      Done\n')

                for iblock, block in enumerate(msd_this_species_this_traj):
                    msd_this_species.append(block)
                msd.set_array('msd_{}_{}_{}'.format(
                    'decomposed' if decomposed else 'isotropic',
                    atomic_species, itraj), msd_this_species_this_traj)

                #
                # linear regression of MSD
                if multiple_params_fit:
                    # using lists of (t_start_fit_dt, t_end_fit_dt):
                    # we will loop over them
                    if decomposed:
                        slopes_intercepts = np.empty(
                            (len(t_start_fit_dt),
                             nr_of_blocks_this_traj, 3, 3, 2))
                    else:
                        slopes_intercepts = np.empty(
                            (len(t_start_fit_dt), nr_of_blocks_this_traj, 2))
                    for istart, (current_t_start_fit_dt,
                                 current_t_end_fit_dt) in enumerate(
                            zip(t_start_fit_dt, t_end_fit_dt)):
                        current_t_list_fit_fs = (
                            timestep_fs * stepsize_t *
                            np.arange(current_t_start_fit_dt//stepsize_t,
                                      current_t_end_fit_dt//stepsize_t))
                        for iblock, block in enumerate(
                                msd_this_species_this_traj):
                            if decomposed:
                                for ipol in range(3):
                                    for jpol in range(3):
                                        data = block[
                                            current_t_start_fit_dt//stepsize_t:
                                            current_t_end_fit_dt//stepsize_t,
                                            ipol, jpol]
                                        slope, intercept, _, _, _ = linregress(
                                            current_t_list_fit_fs, data)
                                        slopes_intercepts[istart, iblock, ipol,
                                                          jpol, 0] = slope
                                        slopes_intercepts[istart, iblock, ipol,
                                                          jpol, 1] = intercept
                                # slopes.append(slopes_intercepts[istart,
                                # iblock, :, :, 0])
                            else:
                                data = block[
                                    (current_t_start_fit_dt-t_start_dt)//stepsize_t:  # noqa: E501
                                    current_t_end_fit_dt//stepsize_t]
                                slope, intercept, _, _, _ = linregress(
                                    current_t_list_fit_fs, data)
                                slopes_intercepts[istart, iblock, 0] = slope
                                slopes_intercepts[istart,
                                                  iblock, 1] = intercept
                                # slopes.append(slope)
                    for iblock, block in enumerate(msd_this_species_this_traj):
                        if decomposed:
                            slopes.append(
                                slopes_intercepts[:, iblock, :, :, 0])
                        else:
                            slopes.append(slopes_intercepts[:, iblock, 0])
                else:
                    # just one value of (t_start_fit_dt, t_end_fit_dt)
                    # TODO: we could avoid this special case by defining
                    # t_start_fit_dt as a lenght-1 list, instead of int.
                    # We keep it for backward-compatibility.
                    if decomposed:
                        slopes_intercepts = np.empty(
                            (nr_of_blocks_this_traj, 3, 3, 2))
                    else:
                        slopes_intercepts = np.empty(
                            (nr_of_blocks_this_traj, 2))
                    t_list_fit_fs = timestep_fs * stepsize_t * \
                        np.arange(t_start_fit_dt//stepsize_t,
                                  t_end_fit_dt//stepsize_t)
                    for iblock, block in enumerate(msd_this_species_this_traj):
                        if decomposed:
                            for ipol in range(3):
                                for jpol in range(3):
                                    slope, intercept, _, _, _ = linregress(
                                        t_list_fit_fs,
                                        block[t_start_fit_dt//stepsize_t:
                                              t_end_fit_dt//stepsize_t,
                                              ipol, jpol])
                                    slopes_intercepts[iblock,
                                                      ipol, jpol, 0] = slope
                                    slopes_intercepts[iblock, ipol,
                                                      jpol, 1] = intercept
                            slopes.append(slopes_intercepts[iblock, :, :, 0])
                        else:
                            slope, intercept, _, _, _ = linregress(
                                t_list_fit_fs,
                                block[(t_start_fit_dt-t_start_dt)//stepsize_t:
                                      t_end_fit_dt//stepsize_t])
                            slopes_intercepts[iblock, 0] = slope
                            slopes_intercepts[iblock, 1] = intercept
                            slopes.append(slopes_intercepts[iblock, 0])

                msd.set_array('slopes_intercepts_{}_{}_{}'.format(
                    'decomposed' if decomposed else 'isotropic',
                    atomic_species, itraj),
                    slopes_intercepts)
                
                # compute MSD with maximal statistics (whole trajectory,
                # no blocks)
                if do_long:
                    # nr_of_t_long may change with the length of the traj
                    # (if not set by user)
                    if t_long_end_dt is not None:
                        nr_of_t_long = t_long_end_dt // stepsize_t
                    elif t_long_factor is not None:
                        nr_of_t_long = int(t_long_factor *
                                           nstep / stepsize_t)
                    else:
                        nr_of_t_long = (nstep - 1) // stepsize_t
                    if nr_of_t_long > nstep:
                        raise RuntimeError(
                            't_long_end_dt is bigger than the '
                            'trajectory length')
                    nr_of_t_long_list.append(nr_of_t_long)
                    t_list_long_fs.append(
                        timestep_fs * stepsize_t * np.arange(nr_of_t_long))
                    msd_this_species_this_traj_max_stats = (
                        prefactor * calculate_msd_specific_atoms_max_stats(
                            positions, indices_of_interest, stepsize_t,
                            stepsize_tau, nr_of_t_long, nstep, nat,
                            nat_of_interest))
                    msd.set_array('msd_long_{}_{}'.format(atomic_species,
                                                          itraj),
                                  msd_this_species_this_traj_max_stats)
            #
            # end of trajectories loop

            # Calculating the mean, std, sem for each point in time
            # (averaging over trajectories)
            msd_mean = np.mean(msd_this_species, axis=0)
            if (len(msd_this_species) > 1):
                msd_std = np.std(msd_this_species, axis=0)
                msd_sem = msd_std / np.sqrt(len(msd_this_species) - 1)
            else:
                msd_std = np.full(msd_mean.shape, np.NaN)
                msd_sem = np.full(msd_mean.shape, np.NaN)
            msd.set_array('msd_{}_{}_mean'.format(
                'decomposed' if decomposed else 'isotropic',
                atomic_species),
                msd_mean)
            msd.set_array('msd_{}_{}_std'.format(
                'decomposed' if decomposed else 'isotropic',
                atomic_species),
                msd_std)
            msd.set_array('msd_{}_{}_sem'.format(
                'decomposed' if decomposed else 'isotropic',
                atomic_species),
                msd_sem)

            slopes = np.array(slopes)  # 0th axis
            results_dict[atomic_species]['slope_msd_mean'] = np.mean(
                slopes, axis=0)
            if (len(msd_this_species) > 1):
                results_dict[atomic_species]['slope_msd_std'] = np.std(
                    slopes, axis=0)
                results_dict[atomic_species]['slope_msd_sem'] = (
                    results_dict[atomic_species]['slope_msd_std']
                    / np.sqrt(len(slopes)-1))
            else:
                results_dict[atomic_species]['slope_msd_std'] = np.full(
                    results_dict[atomic_species]['slope_msd_mean'].shape,
                    np.NaN)
                results_dict[atomic_species]['slope_msd_sem'] = np.full(
                    results_dict[atomic_species]['slope_msd_mean'].shape,
                    np.NaN)

            if decomposed:
                dimensionality_factor = 2.
            else:
                dimensionality_factor = 6.
            results_dict[atomic_species]['diffusion_mean_cm2_s'] = 1e-1 / \
                dimensionality_factor * \
                results_dict[atomic_species]['slope_msd_mean']
            if (len(msd_this_species) > 1):
                results_dict[atomic_species]['diffusion_std_cm2_s'] = 1e-1 / \
                    dimensionality_factor * \
                    results_dict[atomic_species]['slope_msd_std']
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = 1e-1 / \
                    dimensionality_factor * \
                    results_dict[atomic_species]['slope_msd_sem']
            else:
                results_dict[atomic_species]['diffusion_std_cm2_s'] = np.full(
                    results_dict[atomic_species]['diffusion_mean_cm2_s'].shape,
                    np.NaN)
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = np.full(
                    results_dict[atomic_species]['diffusion_mean_cm2_s'].shape,
                    np.NaN)

            # I need to transform to lists, numpy are not json serializable:
            for k in ('slope_msd_mean', 'slope_msd_std', 'slope_msd_sem',
                      'diffusion_mean_cm2_s', 'diffusion_std_cm2_s',
                      'diffusion_sem_cm2_s'):
                if isinstance(results_dict[atomic_species][k],
                              np.ndarray):
                    results_dict[atomic_species][k] = results_dict[
                        atomic_species][k].tolist()
            if self._verbosity > 1:
                print('      Done, these are the results for {}:'.format(
                    atomic_species))
                for key, val in results_dict[atomic_species].items():
                    if not isinstance(val, (tuple, list, dict)):
                        print('          {:<20} {}'.format(key,  val))
        # end of species_of_interest loop
        #

        results_dict.update({
            't_start_fit_dt': (t_start_fit_dt.tolist()
                               if multiple_params_fit
                               else t_start_fit_dt),
            't_end_fit_dt': (t_end_fit_dt.tolist()
                             if multiple_params_fit
                             else t_end_fit_dt),
            't_start_dt':   t_start_dt,
            't_end_dt':   t_end_dt,
            'nr_of_trajectories':   len(trajectories),
            'stepsize_t':   stepsize_t,
            'species_of_interest':   species_of_interest,
            'timestep_fs':   timestep_fs,
            'nr_of_t':   nr_of_t,
            'decomposed':   decomposed,
            'do_long':   do_long,
            'multiple_params_fit':   multiple_params_fit,
        })
        if do_long:
            results_dict['nr_of_t_long_list'] = nr_of_t_long_list
            msd.set_array('t_list_long_fs', t_list_long_fs)
        for k, v in results_dict.items():
            msd.set_attr(k, v)
        return msd

    def get_vaf(self, arrayname=None, integration='trapezoid', **kwargs):
        """
        Calculate Velocity Autocorrelation Function for specified species.

        This method computes VAF which provides insights into particle dynamics
        and can be used to calculate diffusion coefficients and vibrational
        density of states.

        Parameters
        ----------
        arrayname : str, optional
            Name of custom velocity array to use. If ``None``, default velocities are used.
        integration : str, optional
            Integration method for calculating VAF integrals. Options include 'trapezoid', 'simpson', etc.
            Default is 'trapezoid'.
        **kwargs
            Additional parameters for trajectory sampling and fitting:
            - stepsize_t (int, optional): Step size for trajectory sampling. Default is 1.
            - species_of_interest (list, optional): Species to analyze. If ``None``, uses default species.
            - t_start_fit_dt (int, optional): Start time for statistical fitting in timesteps.
            - t_end_fit_dt (int, optional): End time for statistical fitting in timesteps.
            - nr_of_blocks (int, optional): Number of blocks for statistical analysis.

        Returns
        -------
        TimeSeries
            VAF data containing:
            - vaf_isotropic_X_mean: Mean VAF values for species X.
            - vaf_isotropic_X_std: Standard deviation of VAF for species X.
            - vaf_integral_isotropic_X_mean: Integral of VAF (proportional to diffusion).
            - diffusion_mean_cm2_s: Calculated diffusion coefficients in cm²/s.

        Raises
        ------
        NotImplementedError
            If ``do_long`` is requested, as it is not implemented for VAF.
        ValueError
            If no trajectory is set.
        RuntimeError
            If neither ``nr_of_blocks`` nor ``block_length_dt`` is specified.

        Examples
        --------
        >>> analyzer = DynamicsAnalyzer(trajectory, species_of_interest=['Li'])
        >>> vaf_data = analyzer.get_vaf(stepsize_t=1)
        >>> # VAF at time zero should be proportional to temperature
        >>> print(f"VAF(0) = {vaf_data.get_array('vaf_isotropic_Li_mean')[0]}")

        Note
        ----
        The VAF is computed for each species across multiple trajectory blocks for statistical robustness.
        Diffusion coefficients are derived from the integral of VAF using the Green-Kubo relation.
        """

        from samos_modules.mdutils import (
            calculate_vaf_specific_atoms,
            get_com_velocities)
        try:
            timestep_fs = self._timestep_fs
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories '
                'method to set trajectories'
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
                    # I replace the array positions with the COM!
                    # Getting the masses for recentering:
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

                # slopes_intercepts = np.empty((nr_of_blocks_this_traj, 2))

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
                # transforming A^2/fs -> cm^2 /s, dividing by three to get D
                vaf_integral *= 0.1/3. * prefactor

                for iblock in range(nr_of_blocks_this_traj):

                    # ~ D =  0.1 / 3. * prefactor * vaf_integral[iblock]
                    vaf_this_species.append(vaf[iblock])
                    # ~ print vaf[iblock,0]
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
                # ~ print name, arr_mean.shape
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

            if self._verbosity > 1:
                print(
                    ('      Done, these are the results for {}:'.format(
                        atomic_species)))
                for key, val in results_dict[atomic_species].items():
                    if not isinstance(val, (tuple, list, dict)):
                        print(('          {:<20} {}'.format(key,  val)))

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
        """
        Calculate kinetic energy time series for specified species.

        This method computes kinetic energy time series for the system, individual species,
        or individual atoms based on the decomposition options provided. The energies are
        derived from velocity data in the trajectories.

        Parameters
        ----------
        stepsize : int, optional
            Step size for sampling trajectory frames. Default is 1.
        decompose_system : bool, optional
            If ``True``, calculate kinetic energy for the entire system. Default is ``True``.
        decompose_atoms : bool, optional
            If ``True``, calculate kinetic energy for individual atoms. Default is ``False``.
        decompose_species : bool, optional
            If ``True``, calculate kinetic energy for each species. Default is ``False``.

        Returns
        -------
        TimeSeries
            Kinetic energy data containing:
            - system_kinetic_energy_X: Kinetic energy time series for the system (if ``decompose_system=True``) for trajectory X.
            - species_kinetic_energy_X: Kinetic energy time series per species (if ``decompose_species=True``) for trajectory X.
            - atoms_kinetic_energy_X: Kinetic energy time series per atom (if ``decompose_atoms=True``) for trajectory X.
            - mean_system_kinetic_energy_X: Mean kinetic energy for the system for trajectory X.
            - mean_species_kinetic_energy_X: Mean kinetic energy per species for trajectory X.
            - mean_atoms_kinetic_energy_X: Mean kinetic energy per atom for trajectory X.
            - timestep_fs: Timestep in femtoseconds.
            - stepsize: Sampling step size used.

        Raises
        ------
        Exception
            If trajectories are not set using ``set_trajectories`` method.
            If both ``decompose_atoms`` and ``decompose_species`` are set to ``True``, as they are mutually exclusive.

        Examples
        --------
        >>> analyzer = DynamicsAnalyzer(trajectory)
        >>> ke_data = analyzer.get_kinetic_energies(stepsize=2, decompose_system=True, decompose_species=True)
        >>> print(f"Mean system kinetic energy: {ke_data.get_attr('mean_system_kinetic_energy_0'):.2f}")

        Note
        ----
        Kinetic energies are calculated using a prefactor to convert from atomic mass units
        and velocity squared to energy units, normalized by degrees of freedom where applicable.
        """
        from samos_modules.samos_utils import amu_kg, kB

        try:
            timestep_fs = self._timestep_fs
            atoms = self._atoms
            masses = atoms.get_masses()
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        prefactor = amu_kg * 1e10 / kB
        # * 1.06657254018667

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
                kinE[:] /= nat*3  # I devide by the degrees of freedom!
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
                    # ~ print istep0
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
        """
        Calculate power spectrum from velocity autocorrelation function.

        The power spectrum is obtained by Fourier transform of the VAF and
        represents the vibrational density of states (VDOS).

        Parameters
        ----------
        arrayname : str, optional
            Name of custom velocity array to use. If ``None``, default velocities are used.
        **kwargs
            Additional parameters for trajectory sampling and analysis:
            - stepsize_t (int, optional): Step size for trajectory sampling. Default is 1.
            - species_of_interest (list, optional): Species to analyze. If ``None``, uses default species from ``self.get_species_of_interest()``.
            - smoothing (int, optional): Level of smoothing to apply to the periodogram. Default is 1.
            - nr_of_blocks (int, optional): Number of blocks for statistical analysis.
            - block_length_dt (int, optional): Block length in timesteps.
            - block_length_fs (float, optional): Block length in femtoseconds.
            - block_length_ps (float, optional): Block length in picoseconds.

        Returns
        -------
        TimeSeries
            Power spectrum data containing:
            - frequency_X: Frequency arrays in THz for trajectory X.
            - periodogram_X_Y: Power spectrum intensity for species X in trajectory Y.
            - periodogram_X_mean: Mean power spectrum intensity for species X across blocks.
            - periodogram_X_std: Standard deviation of power spectrum intensity for species X.
            - periodogram_X_sem: Standard error of the mean for power spectrum intensity for species X.
            - species_of_interest: List of species analyzed.
            - nr_of_trajectories: Number of trajectories processed.

        Raises
        ------
        Exception
            If trajectories are not set using ``set_trajectories`` method.
        InputError
            If mutually exclusive keywords (e.g., multiple block length units) are provided or unrecognized keywords are used.
        RuntimeError
            If neither ``nr_of_blocks`` nor a block length parameter is specified.

        Examples
        --------
        >>> analyzer = DynamicsAnalyzer(trajectory, species_of_interest=['Li'])
        >>> spectrum = analyzer.get_power_spectrum(smoothing=3)
        >>> # Plot VDOS for Li species
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(spectrum.get_array('frequency_0'), spectrum.get_array('periodogram_Li_mean'))
        >>> plt.xlabel('Frequency (THz)')
        >>> plt.ylabel('VDOS Intensity')
        >>> plt.show()

        Note
        ----
        The power spectrum is computed using ``scipy.signal.periodogram`` with a sampling frequency
        derived from the trajectory timestep. Smoothing is applied via convolution if requested.
        """

        from scipy import signal

        try:
            trajectories = self._trajectories
            timestep_fs = self._timestep_fs
            # Calculating the sampling frequency of the
            # trajectory in THz (the inverse of a picosecond)
            sampling_frequency_THz = 1e3 / timestep_fs
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
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
                    # Use the number of blocks specified by user
                    split_number = nstep // nr_of_blocks_this_traj
                elif block_length_dt > 0:
                    nr_of_blocks_this_traj = nstep // block_length_dt
                    # Use the precise length specified by user
                    split_number = block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_ft '
                        'is specified')

                # I need to have blocks of equal length, and use the
                # split method I need the length of the array to be
                # a multiple of nr_of_blocks_this_traj
                blocks = np.array(np.split(
                    vel_array[:nr_of_blocks_this_traj*split_number],
                    nr_of_blocks_this_traj, axis=0))
                nblocks = len(blocks)
                if self._verbosity > 0:
                    print('nblocks = {}, blocks.shape = {},'
                          ' block_length_ps = {}'.format(
                              nblocks, blocks.shape,
                              blocks.shape[1]*timestep_fs))

                freq, pd = signal.periodogram(blocks,
                                              fs=sampling_frequency_THz,
                                              axis=1, return_onesided=True)
                # I mean over all atoms of this species and directions
                # In the future, maybe consider having a
                # direction resolved periodogram?
                pd_this_species_this_traj = pd.mean(axis=(2, 3))
                # Smothening the array:
                if smothening > 1:
                    # Applying a simple convolution to get the mean
                    kernel = np.ones((nblocks, smothening)) / smothening
                    pd_this_species_this_traj = convolve(
                        pd_this_species_this_traj,
                        kernel, mode='same')

                power_spectrum.set_array('periodogram_{}_{}'.format(
                    atomic_species, itraj), pd_this_species_this_traj)
                if not index_of_species:
                    # I need to save the frequencies only once,
                    # so I save them only for the first species.
                    # I do not see any problem here, but maybe I
                    # missed something.
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
                # Not the end of the world, I just don't calculate the mean
                print(e)

        for k, v in (('species_of_interest', species_of_interest),
                     ('nr_of_trajectories', len(trajectories)),):
            power_spectrum.set_attr(k, v)
        return power_spectrum

def util_msd(trajectory_path, stepsize=1, species=None,
             plot=True, savefig=None, t_start_fit_ps=5,
             t_end_fit_ps=10, timestep=None, nblocks=None):
    """
    High-level interface for MSD calculation with optional plotting.

    This function provides a convenient way to calculate Mean Square Displacement (MSD)
    from trajectory files with automatic plotting and diffusion coefficient extraction.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file. Supports formats like '.extxyz' or others via `Trajectory.load_file`.
    stepsize : int, optional
        Frame sampling interval for trajectory analysis. Default is 1.
    species : list, optional
        List of species to analyze. If ``None``, analyzes all species in the trajectory.
    plot : bool, optional
        Whether to generate and display MSD plots. Default is ``True``. If ``savefig`` is provided, overrides display.
    savefig : str, optional
        Filename to save the generated plots. If provided, saves the plot instead of displaying it.
    t_start_fit_ps : float, optional
        Start time for diffusion coefficient fit in picoseconds. Default is 5.
    t_end_fit_ps : float, optional
        End time for diffusion coefficient fit in picoseconds. Default is 10.
    timestep : float, optional
        Override timestep in femtoseconds. If ``None``, reads timestep from the trajectory file.
    nblocks : int, optional
        Number of blocks for statistical error estimation. If not provided, defaults to a single block.

    Returns
    -------
    TimeSeries
        MSD results containing:
        - t_list_fs: Time array in femtoseconds.
        - msd_isotropic_X_mean: Mean MSD values for species X.
        - msd_isotropic_X_std: Standard deviation of MSD for species X.
        - diffusion_mean_cm2_s: Calculated diffusion coefficients in cm²/s for each species.
        - Additional attributes for fitted slopes and statistical data.

    Raises
    ------
    ValueError
        If the trajectory file cannot be read or if invalid parameters (e.g., negative stepsize) are provided.
    FileNotFoundError
        If the specified `trajectory_path` does not exist.
    Exception
        If trajectory compatibility checks fail or internal analysis errors occur.

    Examples
    --------
    >>> # Calculate MSD for Lithium with plotting and save the figure
    >>> results = util_msd('trajectory.extxyz', stepsize=2, species=['Li'], plot=True, savefig='li_msd.png')
    >>> # Print the diffusion coefficient for Lithium
    >>> print(f"D_Li = {results.get_attr('Li')['diffusion_mean_cm2_s']:.2e} cm²/s")

    Note
    ----
    This function automates trajectory loading, MSD calculation using `DynamicsAnalyzer`, and optional
    visualization. Diffusion coefficients are derived from the linear fit of MSD over the specified
    time range using the Einstein relation.
    """
    if trajectory_path.endswith('.extxyz'):
        from ase.io import read
        aselist = read(trajectory_path, format='extxyz', index=':')
        traj = Trajectory.from_atoms(aselist)
        
    else:
        traj = Trajectory.load_file(trajectory_path)
    if timestep:
        traj.set_timestep(timestep)
    dyn = DynamicsAnalyzer(trajectories=[traj])
    if species is None:
        species = sorted(set(traj.atoms.get_chemical_symbols()))
    print(traj.get_timestep(), species)
    print(t_start_fit_ps, t_end_fit_ps, timestep)
    msd = dyn.get_msd(stepsize_t=stepsize,
                      species_of_interest=species,
                      t_end_fit_ps=t_end_fit_ps,
                      t_start_fit_ps=t_start_fit_ps,
                      nr_of_blocks=nblocks)
    if plot or savefig:
        from samos_modules.samos_plotting import plot_msd_isotropic
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1,1, left=0.18, right=0.95, bottom=0.18, top=0.95)
        fig = plt.figure(figsize=(4,3) )
        ax = fig.add_subplot(gs[0   ])                 
        plot_msd_isotropic(msd, ax=ax)
        if plot:
            plt.show()
        elif savefig:
            plt.savefig(savefig, dpi=240)
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("analysis/plot of a MSD, given a trajectory")
    parser.add_argument('trajectory_path')
    parser.add_argument('-s', '--stepsize', type=int,
                        help='Stepsize over the trajectory, defaults to 1',
                        default=1)
    parser.add_argument('--species', nargs='+',)
    parser.add_argument('--plot', action='store_true',
                        help='Plot the MSD to screen')
    parser.add_argument('-te', '--t-end-fit-ps',
                        help='End of the fit in ps, defaults to 100',
                        type=float, default=10)
    parser.add_argument('-ts', '--t-start-fit-ps',
                        help='End of the fit in ps, defaults to 50',
                        type=float, default=5)
    parser.add_argument('--timestep', type=float,
                        help='Timestep in fs, defaults to 1',
                        default=1)
    parser.add_argument('-n', '--nblocks', type=int, 
                        default=1, help='Number of blocks to use')
    parser.add_argument(
        '--savefig',
        help='Where to save figure (will otherwise show on screen)')
    args = parser.parse_args()
    kwargs = vars(args)
    util_msd(**kwargs)

def write_xsf_header(
        atoms, positions, cell, data,
        vals_per_line=6, outfilename=None, **kwargs):
    """
    Write XSF file header for density data visualization.

    This function writes the header information for XSF files containing 3D scalar field
    data such as electron or ionic densities. It formats structural data (atoms, positions,
    cell) and optional grid data for visualization purposes.

    Parameters
    ----------
    atoms : list
        List of atomic symbols (e.g., ['Li', 'O']).
    positions : np.ndarray
        Array of atomic positions with shape (N, 3) where N is the number of atoms.
    cell : np.ndarray
        Unit cell vectors as a 3x3 array representing the simulation box.
    data : np.ndarray
        3D grid data for the scalar field. If ``None``, grid dimensions must be provided via `kwargs`.
    vals_per_line : int, optional
        Number of values per line in the output file for grid data. Default is 6.
    outfilename : str, optional
        Output filename to write the XSF data. If ``None``, prints to stdout.
    **kwargs
        Additional parameters for grid dimensions if `data` is ``None``:
        - xdim (int, optional): Grid dimension along x-axis.
        - ydim (int, optional): Grid dimension along y-axis.
        - zdim (int, optional): Grid dimension along z-axis.

    Returns
    -------
    None
        Writes the XSF header and data (if provided) to the specified file or stdout.

    Raises
    ------
    Exception
        If `outfilename` is neither a string nor ``None`` (invalid output target).
    ValueError
        If grid dimensions are not provided via `data` or `kwargs` when required.
    IOError
        If there are issues writing to the specified `outfilename`.

    Examples
    --------
    >>> # Write XSF header with density data to a file
    >>> atoms = ['Li', 'O']
    >>> positions = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
    >>> cell = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    >>> density_grid = np.random.rand(10, 10, 10)
    >>> write_xsf_header(atoms, positions, cell, density_grid, outfilename='density.xsf')

    Note
    ----
    XSF (XCrySDen Structure Format) files are used for visualizing crystal structures and
    associated scalar fields. This function handles the header and data formatting according
    to the XSF specification, including periodic boundary conditions via the unit cell.
    """
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
    """
    Generate Gaussian density field from atomic positions.

    This function creates a 3D density field by placing Gaussian functions at atomic
    positions. It is useful for visualization and analysis of ionic distributions in
    molecular dynamics simulations.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory object containing atomic positions and cell information.
    element : str, optional
        Element symbol to include in density calculation. If ``None``, includes all elements unless `indices` is specified.
    outputfile : str, optional
        Output filename for the XSF file. Default is 'out.xsf'. Automatically appends '.xsf' if not present.
    sigma : float, optional
        Gaussian width parameter in Ångstroms. Controls the spread of the Gaussian function. Default is 0.3.
    n_sigma : float, optional
        Cutoff distance as a multiple of `sigma`. Defines the range of the Gaussian function. Default is 3.0.
    density : float, optional
        Grid density in points per Ångstrom. Determines the resolution of the 3D grid. Default is 0.1.
    istart : int, optional
        First frame to include in the density calculation. Default is 1.
    istop : int, optional
        Last frame to include in the density calculation. If ``None``, uses all frames up to the end. Default is ``None``.
    stepsize : int, optional
        Step size for frame sampling between `istart` and `istop`. Default is 1.
    indices : list, optional
        Specific atom indices to include in the density calculation. If ``None``, determined by `element` or includes all atoms.
    indices_exclude_from_plot : list, optional
        Atom indices to exclude from visualization in the XSF file. If ``None``, defaults to the same as `indices`.

    Returns
    -------
    None
        Writes the generated 3D density field to the specified XSF file for visualization.

    Raises
    ------
    Exception
        If the specified `element` is not found in the trajectory or if there are issues with trajectory data.
    ValueError
        If input parameters (e.g., negative `sigma`, invalid frame indices) are invalid.
    IOError
        If there are issues writing to the specified `outputfile`.

    Examples
    --------
    >>> # Create a density field for Lithium atoms from a trajectory
    >>> traj = Trajectory.load_file("trajectory.extxyz")
    >>> get_gaussian_density(traj, element="Li", sigma=0.4, outputfile="li_density.xsf")
    >>> # Visualize the resulting XSF file in a compatible viewer like XCrySDen

    Note
    ----
    The density field is computed by summing Gaussian functions centered at atomic positions
    across selected frames. The resulting XSF file can be visualized using tools like XCrySDen
    or VMD to analyze ionic distributions. Grid dimensions are automatically determined based
    on cell size and specified `density`.
    """
    from samos_modules.gaussian_density import make_gaussian_density

    cell = trajectory.cell
    positions = trajectory.get_positions()

    nstep, nat, _ = positions.shape
    symbols = trajectory.atoms.get_chemical_symbols()
    starting_pos = trajectory.atoms.get_positions()

    if not outputfile.endswith('.xsf'):
        outputfile += '.xsf'

    # indices_i_care are used to calculate the density
    if indices is None:
        if element:
            indices = trajectory.get_indices_of_species(
                element, start=1)
        else:
            indices = np.array(list(range(1, nat+1)))

    print('(get_gaussian_density) indices:', indices)
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
        print('Warning Could not get progressbar ({})'.format(e))
        pbar_frequency = int((istop - istart) / 30)

    pbar_frequency = max([pbar_frequency, 1])
    a, b, c = [np.linalg.norm(cell[i]) for i in range(3)]
    n1, n2, n3 = [int(celldim/density)+1 for celldim in (a, b, c)]

    print('Grid dimensions {} x {} x {}'.format(n1, n2, n3))
    print('Box dimensions  {} x {} x {}'.format(a, b, c))
    print('xsf file: ', format(outputfile))
    if indices_exclude_from_plot is None:
        indices_exclude_from_plot = indices
    print(
        '(get_gaussian_density) We do not show these atoms in the xsf file: '
        f'{indices_exclude_from_plot}')
    write_xsf_header(
        [s for i, s in enumerate(symbols, start=1)
         if i not in indices_exclude_from_plot],
        [p for i, p in enumerate(starting_pos, start=1)
         if i not in indices_exclude_from_plot],
        cell, None, outfilename=outputfile, xdim=n1, ydim=n2, zdim=n3)

    S = np.matrix(np.diag([1, 1, 1, -(sigma*n_sigma/density)**2]))
    cellT = cell.T
    cellTI = np.matrix(cellT).I
    #  I describe the move from atomic to crystal
    # coordinates with an affine transformation M:
    M = np.matrix(np.r_[np.c_[np.matrix(cellTI), np.zeros(3)], [[0, 0, 0, 1]]])
    # Q is a check, but not used. Check is orthogonality
    # Q is the sphere transformed by transformation M
    # Q = M.I.T * S * M.I
    # Now, as defined in the source, I calculate R = Q^(-1)
    R = M * S.I * M.T
    # The boundaries are given by:
    # ~ xmax = (R[0,3] - np.sqrt(R[0,3]**2 - R[0,0]*R[3,3])) / R[3,3]
    # ~ xmin = (R[0,3] + np.sqrt(R[0,3]**2 - R[0,0]*R[3,3])) / R[3,3]
    # ~ ymax = (R[1,3] - np.sqrt(R[1,3]**2 - R[1,1]*R[3,3])) / R[3,3]
    # ~ ymin = (R[1,3] + np.sqrt(R[1,3]**2 - R[1,1]*R[3,3])) / R[3,3]
    # ~ zmax = (R[2,3] - np.sqrt(R[2,3]**2 - R[2,2]*R[3,3])) / R[3,3]
    # ~ zmin = (R[2,3] + np.sqrt(R[2,3]**2 - R[2,2]*R[3,3])) / R[3,3]
    # The size of the bounding box is given by (max - min)
    # for each dimension.
    # I want this to be expressed as integer values in the grid,
    # though, for convenience.
    # In  plain terms, bx,by,bz tell me how many grid point
    # I have to walk up/down in x/y/z
    # maximally to be sure that I contain all the points that lie
    # with n_sigma*sigma from the origin!
    # Of course, of main importance is the density!
    # I add to be sure, since int cuts of floating points!
    b1 = int(np.abs(
        (R[0, 3] - np.sqrt(R[0, 3]**2 - R[0, 0]*R[3, 3]))
        / R[3, 3]) / density) + 1
    # Normally I would have to do 0.5 (xmax - xmin) from above, but I know that
    # I'm at the origin R[0,3] is 0
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


if __name__ == '__main__':
    # Defining the command line arguments:
    from argparse import ArgumentParser
    from ase.io import read
    from samos_modules.samos_trajectory import Trajectory
    from samos_modules.samos_io import read_positions_with_ase
    ap = ArgumentParser()

    ap.add_argument('cif', help='Cif file with structure')
    ap.add_argument('positions', help='a trajectory file to read')

    ap.add_argument('-n', '--n-sigma', type=int, default=3)
    ap.add_argument('-d', '--density', type=float, default=0.1,
                    help='nr of grid points per angstrom')
    # ap.add_argument('-r', '--recenter', action='store_true')  N
    ap.add_argument('--istart', help='starting point', type=int, default=1)
    ap.add_argument('--istop', help='ending point', type=int, default=None)
    ap.add_argument('-i', '--stepsize',
                    help='stepsize in trajectory', default=1)
    ap.add_argument('-s', '--sigma',
                    help='Value of sigma in ANGSTROM', type=float, default=0.3)

    ap.add_argument('-e', '--element',
                    help='Density of this atom-type', type=str, default='Li')
    ap.add_argument('-o', '--outputfile', help='outputfile', default='out.xsf')

    # Parsing the arguments:
    parsed_args = vars(ap.parse_args(sys.argv[1:]))

    t = Trajectory()
    t.set_atoms(read(parsed_args.pop('cif')))
    t.set_array(t._POSITIONS_KEY, read_positions_with_ase(
        parsed_args.pop('positions')), check_nat=False)
    get_gaussian_density(t, **parsed_args)

class BaseAnalyzer(object, metaclass=ABCMeta):
    """
    Abstract base class for molecular dynamics analysis tools.

    This class provides a common interface for all analysis tools in SAMOS,
    ensuring consistent API design and functionality across different analysis types.

    Attributes
    ----------
    trajectory : Trajectory
        Trajectory object representing the molecular dynamics trajectory.
    verbose : bool
        Whether to print progress information during analysis.

    Methods
    -------
    set_trajectory
        Set the trajectory for analysis.
    run
        Perform the analysis (abstract method to be implemented by subclasses).

    Note
    ----
    This is an abstract base class and should be subclassed with a concrete
    implementation of the `run` method.
    """
    def __init__(self, **kwargs):
        """
        Initialize the base analyzer.

        Parameters
        ----------
        trajectory : Trajectory, optional
            Trajectory object for analysis.
        verbose : bool, optional
            Enable or disable verbose output. Default is ``False``.

        Examples
        --------
        >>> # Initialize a base analyzer with a trajectory
        >>> traj = Trajectory.load_file("trajectory.extxyz")
        >>> analyzer = BaseAnalyzer(trajectory=traj, verbose=True)
        """
        for key, val in list(kwargs.items()):
            getattr(self, 'set_{}'.format(key))(val)

    def set_trajectory(self, trajectory):
        """
        Set the trajectory for analysis.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory object representing the molecular dynamics trajectory.

        Raises
        ------
        TypeError
            If the provided trajectory is not an instance of `Trajectory`.

        Examples
        --------
        >>> # Set a new trajectory for an existing analyzer
        >>> traj = Trajectory.load_file("new_trajectory.extxyz")
        >>> analyzer.set_trajectory(traj)
        """
        if not isinstance(trajectory, Trajectory):
            raise TypeError(
                'You need to pass a {} as trajectory'.format(Trajectory.__name__))
        self._trajectory = trajectory

    @abstractmethod
    def run(*args, **kwargs):
        """
        Perform the analysis. Must be implemented by subclasses.

        Returns
        -------
        object
            Analysis results, with the format depending on the specific analyzer implementation.

        Raises
        ------
        NotImplementedError
            If called directly on `BaseAnalyzer` or if not implemented in a subclass.

        Note
        ----
        This is an abstract method and must be overridden in subclasses to define
        the specific analysis logic.
        """
        pass

class RDF(BaseAnalyzer):
    """
    Radial Distribution Function calculator.

    This class computes pair correlation functions (RDF) for specified atom pairs
    in molecular dynamics trajectories. It supports both partial and total RDFs
    with proper normalization. Inherits from `BaseAnalyzer` to provide a consistent
    analysis interface.

    Methods
    -------
    run
        Calculate RDF for specified species pairs.
    run_fort
        Use Fortran backend for RDF calculation acceleration (if available).

    Note
    ----
    RDFs describe the probability of finding a particle at a certain distance from
    a reference particle, useful for structural analysis in molecular simulations.
    """
    def run_fort(self, radius=None, species_pairs=None, istart=0, istop=None, stepsize=1, nbins=100):
        """
        Calculate RDF using a Fortran backend for acceleration.

        Parameters
        ----------
        radius : float, optional
            The maximum radius for RDF calculation. If ``None``, a default value is used.
        species_pairs : list, optional
            List of tuples specifying species pairs to analyze. If ``None``, all unique pairs are considered.
        istart : int, optional
            First frame index to include. Default is 0.
        istop : int, optional
            Last frame index to include. If ``None``, uses all frames.
        stepsize : int, optional
            Step size for frame sampling. Default is 1.
        nbins : int, optional
            Number of radial bins for the histogram. Default is 100.

        Returns
        -------
        AttributedArray
            RDF results containing RDF values, integrals, and radial distances for each species pair.

        Raises
        ------
        NotImplementedError
            If the Fortran backend is not fully implemented or available.

        Examples
        --------
        >>> rdf_calc = RDF(trajectory=traj)
        >>> rdf_data = rdf_calc.run_fort(radius=5.0, species_pairs=[('Li', 'O')])
        """
        if 1:
            raise NotImplementedError('This is not fully implemented')
        from samos_modules.rdf import calculate_rdf
        atoms = self._trajectory.atoms
        volume = atoms.get_volume()
        positions = self._trajectory.get_positions()
        if istop is None:
            istop = len(positions)
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(
                set(atoms.get_chemical_symbols()), 2))
        cell = np.array(atoms.cell.T)
        cellI = np.linalg.inv(cell)
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs)
        for spec1, spec2 in species_pairs:
            ind1 = np.where(chem_sym == spec1)[0] + 1  # +1 for fortran indexing
            ind2 = np.where(chem_sym == spec2)[0] + 1
            density = float(len(ind2)) / volume
            rdf, integral, radii = calculate_rdf(
                positions, istart, istop, stepsize,
                radius, density, cell,
                cellI, ind1, ind2, nbins)
            rdf_res.set_array('rdf_{}_{}'.format(spec1, spec2), rdf)
            rdf_res.set_array('int_{}_{}'.format(spec1, spec2), integral)
            rdf_res.set_array('radii_{}_{}'.format(spec1, spec2), radii)
        return rdf_res

    def run(self, radius=None, species_pairs=None, istart=0, istop=None, stepsize=1, nbins=100):
        """
        Calculate radial distribution function (RDF).

        This method computes pair correlation functions between specified atom types,
        properly accounting for periodic boundary conditions.

        Parameters
        ----------
        radius : float, optional
            Maximum radial distance to consider for RDF calculation.
        species_pairs : list, optional
            List of tuples with species pairs to analyze. If ``None``, all unique pairs are analyzed.
        istart : int, optional
            First frame index to include. Default is 0.
        istop : int, optional
            Last frame index to include. If ``None``, uses all frames.
        stepsize : int, optional
            Step size for frame sampling. Default is 1.
        nbins : int, optional
            Number of radial bins for the histogram. Default is 100.

        Returns
        -------
        AttributedArray
            RDF results containing:
            - rdf_X_Y: RDF values for species X and Y.
            - int_X_Y: Integrated RDF (coordination number).
            - radii_X_Y: Radial distance values.
            - n_pairs_X_Y: Number of atomic pairs analyzed.
            - n_data_X_Y: Total number of data points used.

        Raises
        ------
        ValueError
            If `istop` is greater than or equal to the number of positions in the trajectory.

        Examples
        --------
        >>> rdf_calc = RDF(trajectory=traj)
        >>> rdf_data = rdf_calc.run(radius=5.0, species_pairs=[('Li', 'O')])
        >>> print(f"Li-O RDF at first bin: {rdf_data.get_array('rdf_Li_O')[0]}")

        Note
        ----
        For non-orthogonal cells, the algorithm may need many periodic images to
        properly account for all pairs within the cutoff distance.
        """
        def get_indices(spec, chem_sym):
            """
            Get the indices for specification spec.
            """
            if isinstance(spec, str):
                return np.where(chem_sym == spec)[0].tolist()
            elif isinstance(spec, int):
                return [spec]
            elif isinstance(spec, (tuple, list)):
                list_ = []
                for item in spec:
                    list_ += get_indices(item, chem_sym)
                return list_
            else:
                raise TypeError(
                    '{} can not be transformed to index'.format(spec))

        def get_label(spec, ispec):
            """
            Get a good label for specification spec. If none can be found,
            give one based on iteration counter ispec.
            """
            if isinstance(spec, str):
                return spec
            elif isinstance(spec, (tuple, list)):
                return 'spec_{}'.format(ispec)
            else:
                print(type(spec))

        positions = self._trajectory.get_positions()
        types = self._trajectory.get_types()
        cells = self._trajectory.get_cells()
        range_ = list(range(0, 2))
        if cells is None:
            fixed_cell = True
            atoms = self._trajectory.atoms
            volume = atoms.get_volume()
            try:
                cell = atoms.cell.array
            except AttributeError:
                cell = atoms.cell.copy()
            cellI = np.linalg.inv(cell)
            a, b, c = cell
            corners = [i*a+j*b + k*c for i in range_ for j in range_ for k in range_]
        else:
            fixed_cell = False
        if istop is None:
            istop = len(positions)
        elif istop >= len(positions):
            raise ValueError("Istop ({}) is higher than (or equal to) "
                             "number of positions ({})".format(
                                 istop, len(positions)))
        if species_pairs is None:
            species_pairs = sorted(list(
                itertools.combinations_with_replacement(
                    sorted(set(types)), 2)))
        indices_pairs = []
        labels = []
        species_pairs_pruned = []
        for ispec, (spec1, spec2) in enumerate(species_pairs):
            ind_spec1, ind_spec2 = (get_indices(spec1, types),
                                    get_indices(spec2, types))
            # special situation if there's only one atom of a species
            # and we're making the RDF of that species with itself.
            # there will be empty pairs_of_atoms and the
            # code below would crash!
            if ind_spec1 == ind_spec2 and len(ind_spec1) == 1:
                continue
            indices_pairs.append((ind_spec1, ind_spec2))
            labels.append('{}_{}'.format(
                get_label(spec1, ispec), get_label(spec2, ispec)))
            species_pairs_pruned.append((spec1, spec2))
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs_pruned)
        binsize = float(radius)/nbins

        # wrapping the positions:
        for label, (ind1, ind2) in zip(labels, indices_pairs):
            if ind1 == ind2:
                # lists are equal, I will therefore not double calculate
                pairs_of_atoms = [(i, j) for i in ind1
                                  for j in ind2 if i < j]
                pair_factor = 2.0
            else:
                pairs_of_atoms = [(i, j) for i in ind1
                                  for j in ind2 if i != j]
                pair_factor = 1.0
            # It can happen that pairs_of_atoms
            ind_pair1, ind_pair2 = list(zip(*pairs_of_atoms))

            # doing a loop in time to avoid memory explosion
            # this also makes it easier to deal with cell changes
            hist, bin_edges = np.histogram([], bins=nbins, range=(0, radius))
            hist = hist.astype(float)
            # normalize the histogram, by the number of steps taken,
            # and the number of species1
            prefactor = (
                pair_factor
                / float(len(np.arange(istart, istop, stepsize)))
                / float(len(ind1)))
            for index in np.arange(istart, istop, stepsize):
                if not fixed_cell:
                    cell = cells[index]
                    volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
                    cellI = np.linalg.inv(cell)
                    a, b, c = cell
                    corners = np.array([i*a+j*b + k*c
                                        for i in range_
                                        for j in range_
                                        for k in range_])
                diff_real_unwrapped = (
                    positions[index, ind_pair2, :]
                    - positions[index, ind_pair1, :])
                diff_crystal_wrapped = (diff_real_unwrapped@cellI) % 1.0
                diff_real_wrapped = np.dot(diff_crystal_wrapped, cell)
                # in diff_real_wrapped I have all positions wrapped
                # into periodic cell
                shortest_distances = cdist(
                    diff_real_wrapped, corners).min(axis=1)
                hist += prefactor * \
                    (np.histogram(shortest_distances, bins=nbins,
                     range=(0, radius))[0]).astype(float)

            radii = 0.5*(bin_edges[:-1]+bin_edges[1:])

            rdf = hist / (4.0 * np.pi * radii**2 * binsize) / \
                (len(ind2)/volume)
            integral = np.empty(len(rdf))
            sum_ = 0.0
            for i in range(len(integral)):
                sum_ += hist[i]
                integral[i] = sum_
            rdf_res.set_array('rdf_{}'.format(label), rdf)
            rdf_res.set_array('int_{}'.format(label), integral)
            rdf_res.set_array('radii_{}'.format(label), radii)
            rdf_res.set_attr('n_pairs_{}'.format(label), len(pairs_of_atoms))
            rdf_res.set_attr('n_data_{}'.format(label),
                             len(pairs_of_atoms) * ((istop-istart)//stepsize))

        return rdf_res

class AngularSpectrum(BaseAnalyzer):
    """
    Angular momentum spectrum analyzer.

    This class analyzes rotational dynamics and angular momentum distributions
    in molecular systems. Inherits from `BaseAnalyzer` to provide a consistent
    analysis interface.

    Methods
    -------
    run
        Calculate angular momentum spectrum for specified species triplets.

    Note
    ----
    Angular momentum spectra are useful for understanding rotational behavior
    and bonding geometries in molecular simulations.
    """
    def run(self, radius=None, species_pairs=None, istart=1, istop=None, stepsize=1, nbins=100):
        """
        Calculate angular momentum spectrum.

        Parameters
        ----------
        radius : float, optional
            Maximum distance for considering atom interactions.
        species_pairs : list, optional
            List of tuples with species triplets to analyze. If ``None``, all unique triplets are considered.
        istart : int, optional
            First frame index to include. Default is 1.
        istop : int, optional
            Last frame index to include. If ``None``, uses all frames.
        stepsize : int, optional
            Step size for frame sampling. Default is 1.
        nbins : int, optional
            Number of bins for the angular histogram. Default is 100.

        Returns
        -------
        AttributedArray
            Angular spectrum data containing:
            - aspec_X_Y_Z: Angular spectrum values for species X, Y, Z.
            - angles_X_Y_Z: Corresponding angle values for the spectrum.

        Raises
        ------
        ValueError
            If frame indices are invalid or trajectory data is inaccessible.

        Examples
        --------
        >>> ang_spec = AngularSpectrum(trajectory=traj)
        >>> spec_data = ang_spec.run(radius=3.0, species_pairs=[('Li', 'O', 'Li')])
        >>> print(f"First angle bin: {spec_data.get_array('angles_Li_O_Li')[0]}")
        """
        from samos_modules.rdf import calculate_angular_spec
        atoms = self._trajectory.atoms
        positions = self._trajectory.get_positions()
        if istop is None:
            istop = len(positions)
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(
                set(atoms.get_chemical_symbols()), 3))
        cell = np.array(atoms.cell)
        cellI = np.linalg.inv(cell)
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs)
        for spec1, spec2, spec3 in species_pairs:
            ind1 = np.where(chem_sym == spec1)[0] + 1  # +1 for fortran indexing
            ind2 = np.where(chem_sym == spec2)[0] + 1
            ind3 = np.where(chem_sym == spec3)[0] + 1
            angular_spec, angles = calculate_angular_spec(
                positions, istart, istop, stepsize,
                radius, cell, cellI, ind1, ind2, ind3, nbins)
            rdf_res.set_array('aspec_{}_{}_{}'.format(
                spec1, spec2, spec3), angular_spec)
            rdf_res.set_array('angles_{}_{}_{}'.format(
                spec1, spec2, spec3), angles)
        return rdf_res

def util_rdf_and_plot(trajectory_path, radius=5.0, stepsize=1, bins=100,
                      species_pairs=None, savefig=None, plot=False,
                      printrdf=False, no_int=False):
    """
    Calculate RDF with optional plotting functionality.

    This function provides a high-level interface for RDF calculation from trajectory
    files with automatic plotting and data output capabilities.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file. Supports formats like '.extxyz' or others via `Trajectory.load_file`.
    radius : float, optional
        Maximum distance for RDF calculation in Ångstroms. Default is 5.0.
    stepsize : int, optional
        Frame sampling interval. Default is 1.
    bins : int, optional
        Number of radial bins for the histogram. Default is 100.
    species_pairs : list, optional
        Atom pairs to analyze as "A-B" strings. If ``None``, calculates for all unique pairs.
    savefig : str, optional
        Filename to save RDF plots. If provided, saves the plot instead of displaying it.
    plot : bool, optional
        Whether to generate and display RDF plots. Default is ``False``. Overridden by `savefig` if provided.
    printrdf : bool or str, optional
        If ``True`` or a string prefix, outputs RDF data to files with names like 'prefix-A-B.dat'.
        Default is ``False``.
    no_int : bool, optional
        If ``True``, excludes the RDF integral from plots. Default is ``False``.

    Returns
    -------
    AttributedArray
        RDF data for all requested pairs, containing:
        - rdf_X_Y: RDF values for species X and Y.
        - int_X_Y: Integral of RDF (coordination number).
        - radii_X_Y: Radial distance values in Ångstroms.

    Raises
    ------
    FileNotFoundError
        If the specified `trajectory_path` does not exist.
    ValueError
        If `species_pairs` format is invalid or other input parameters are out of range.

    Examples
    --------
    >>> # Calculate and plot Li-O RDF, saving the figure
    >>> rdf_data = util_rdf_and_plot('traj.extxyz', radius=10.0, species_pairs=['Li-O'], plot=True, savefig='li_o_rdf.png')
    >>> # Access RDF values for Li-O pair
    >>> print(f"First RDF value: {rdf_data.get_array('rdf_Li_O')[0]}")
    """
    if trajectory_path.endswith('.extxyz'):
        from ase.io import read
        aselist = read(trajectory_path, format='extxyz', index=':')
        traj = Trajectory.from_atoms(aselist)
    else:
        traj = Trajectory.load_file(trajectory_path)
    print("Read trajectory of shape {}".format(traj.get_positions().shape))
    if species_pairs:
        species_pairs_ = []
        for spec in species_pairs:
            species_pairs_.append(spec.split('-'))
    else:
        species_pairs_ = None
    rdf = RDF(trajectory=traj)
    res = rdf.run(radius=radius, stepsize=stepsize,
                  nbins=bins, species_pairs=species_pairs_)
    if plot or savefig:
        from samos_modules.samos_plotting import plot_rdf
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(4, 3))
        gs = GridSpec(1, 1, top=0.99, right=0.83, left=0.14, bottom=0.16)
        ax = fig.add_subplot(gs[0])
        plot_rdf(res, ax=ax, no_int=no_int)
        ax.set_xlim(-0.2, radius)
        if savefig:
            plt.savefig(savefig, dpi=250)
        if plot:
            plt.show()

    if printrdf:
        species_pairs = res.get_attr('species_pairs')
        for spec1, spec2 in species_pairs:
            try:
                rdf = res.get_array('rdf_{}_{}'.format(spec1, spec2))
            except KeyError:
                print(
                    'Warning: RDF for {}-{} was not calculated, skipping'
                    ''.format(spec1, spec2))
                continue
            integral = res.get_array('int_{}_{}'.format(spec1, spec2))
            radii = res.get_array('radii_{}_{}'.format(spec1, spec2))
            name = '{}-{}-{}.dat'.format(printrdf, spec1, spec2)
            np.savetxt(name, np.array([radii, rdf, integral]).T,
                       header='radius    rdf     integral')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("analysis/plot of a RDF, given a trajectory")
    parser.add_argument('trajectory_path')
    parser.add_argument('-r', '--radius', required=False, type=float,
                        default=5.0,
                        help='The radius (max) of the RDF, defaults to 5.0')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins, defaults to 100', default=100)
    parser.add_argument('-s', '--stepsize', type=int,
                        help='Stepsize over the trajectory, defaults to 1',
                        default=1)
    parser.add_argument('--species-pairs', nargs='+',
                        help=('species pairs separated by a dash, e.g., '
                              '--species-pairs C-O O-O'))
    parser.add_argument('--printrdf',
                        help='Print the RDF to a file as a csv',)
    parser.add_argument('--plot', action='store_true',
                        help='Plot the RDF to screen')
    parser.add_argument('--no-int', action='store_true',
                        help='dont plot integral')
    parser.add_argument(
        '--savefig',
        help='Where to save figure (will otherwise show on screen)')
    args = parser.parse_args()
    kwargs = vars(args)
    util_rdf_and_plot(**kwargs)

