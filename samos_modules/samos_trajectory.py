# -*- coding: utf-8 -*-

import numpy as np
from samos_modules.samos_utils import AttributedArray


class IncompatibleTrajectoriesException(Exception):
    """
    Exception raised when attempting to combine incompatible trajectories.
    
    This exception indicates that trajectories cannot be combined due to
    differences in atom types, array structure, or timestep values.
    """
    pass


def check_trajectory_compatibility(trajectories):
    """
    Verify that multiple trajectories can be safely combined.
    
    This function checks whether trajectories are compatible by confirming
    they have the same atomic composition, array structure, and timestep.
    
    Args:
        trajectories (list): List of Trajectory objects to check
        
    Returns:
        tuple: (types, timestep) containing the common atom types and timestep
        
    Raises:
        IncompatibleTrajectoriesException: If trajectories differ in atom types,
            array structure, or timestep
        TypeError: If any element is not a Trajectory instance
        AssertionError: If empty list is provided
            
    Example:
        >>> try:
        ...     types, dt = check_trajectory_compatibility([traj1, traj2])
        ...     print(f"Compatible trajectories with {len(types)} atom types")
        ... except IncompatibleTrajectoriesException as e:
        ...     print(f"Incompatible trajectories: {e}")
    """

    assert len(trajectories) >= 1, 'No trajectories passed'
    for t in trajectories:
        if not isinstance(t, Trajectory):
            raise TypeError('{} is not an instance of Trajectory'.format(t))
    array_names_set = set()
    types_set = set()
    timestep_set = set()
    for t in trajectories:
        array_names_set.add(frozenset(t.get_arraynames()))
        types_set.add(tuple(t.types))
        timestep_set.add(t.get_timestep())

    if len(array_names_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different arrays are set')
    if len(types_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different chemical symbols in trajectories')
    if len(timestep_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different timesteps in trajectories')
    return np.array(types_set.pop()), timestep_set.pop()


class Trajectory(AttributedArray):
    """
    Molecular dynamics trajectory representation with analytical capabilities.
    
    The Trajectory class provides a comprehensive container for time-ordered
    phase space data from molecular dynamics simulations. It manages positions,
    velocities, forces, and structural information with consistent unit handling.
    
    Internal units:
        - Time: Femtoseconds
        - Distance: Angstroms
        - Energy: eV
        - Cell/Mass: As defined in the ASE Atoms object
    
    The class provides methods for trajectory manipulation, analysis, and
    conversion to/from ASE Atoms objects. It handles periodic boundary
    conditions and species-specific operations.
    
    Attributes:
        atoms: ASE Atoms object representing the system structure
        types: Chemical symbols of all atoms
        nat: Number of atoms in the system
        cell: Unit cell vectors
        nstep: Number of time steps in the trajectory
        
    Example:
        >>> # Create a trajectory from a list of ASE Atoms objects
        >>> traj = Trajectory.from_atoms(atoms_list, timestep_fs=1.0)
        >>> # Access positions for all atoms at all timesteps
        >>> positions = traj.get_positions()
        >>> # Get indices of specific elements
        >>> li_indices = traj.get_indices_of_species('Li')
        >>> # Export a specific timestep as an ASE Atoms object
        >>> atoms_t100 = traj.get_step_atoms(100)
    """
    _TIMESTEP_KEY = 'timestep_fs'
    _POSITIONS_KEY = 'positions'
    _VELOCITIES_KEY = 'velocities'
    _STRESS_KEY = 'stress'
    _CELL_KEY = 'cells'
    _FORCES_KEY = 'forces'
    _POT_ENER_KEY = 'potential_energy'
    _ATOMS_FILENAME = 'atoms.traj'
    _TYPES_KEY = 'types'

    def __init__(self, **kwargs):
        """
        Instantiating a trajectory class.
        Optional keyword-arguments are everything with a set-method.
        """
        self._atoms = None
        super(Trajectory, self).__init__(**kwargs)

    @classmethod
    def from_atoms(cls, atoms_list, timestep_fs=None):
        """
        Create a Trajectory from a list of ASE Atoms objects.
        
        This is the primary constructor for creating trajectory objects from
        existing atomic structure snapshots.
        
        Args:
            atoms_list (list): List of ASE Atoms objects representing trajectory frames
            timestep_fs (float, optional): Time step between frames in femtoseconds
        
        Returns:
            Trajectory: New trajectory instance containing data from atoms_list
            
        Raises:
            TypeError: If atoms_list items are not ASE Atoms objects
            ValueError: If atoms_list is empty or has inconsistent chemical symbols
                
        Example:
            >>> # Create trajectory from a series of Atoms objects
            >>> from ase.io import read
            >>> atoms_list = read('simulation.xyz', index=':')
            >>> traj = Trajectory.from_atoms(atoms_list, timestep_fs=1.0)
            >>> print(f"Created trajectory with {traj.nstep} frames and {traj.nat} atoms")
        """
        from ase import Atoms
        chem_sym_set = set()
        for atoms in atoms_list:
            if not isinstance(atoms, Atoms):
                raise TypeError("I have to receive a list/iterable over "
                                "{}".format(Atoms))
            chem_sym_set.add(tuple(atoms.get_chemical_symbols()))
        if len(chem_sym_set) < 1:
            raise ValueError("Empty list provided")
        elif len(chem_sym_set) > 1:
            raise ValueError("The chemical_symbols list of provided atoms "
                             "are not the same for all, cannot proceed")

        positions = np.array([atoms.get_positions() for atoms in atoms_list])
        velocities = np.array([atoms.get_velocities() for atoms in atoms_list])
        try:
            forces = np.array([atoms.get_forces() for atoms in atoms_list])
        except Exception:
            forces = None
        cells = np.array([atoms.cell for atoms in atoms_list])
        new = cls(atoms=atoms_list[0])
        new.set_positions(positions)
        try:
            if (velocities**2).sum() > 1e-12:
                new.set_velocities(velocities)
        except TypeError:
            pass  # velocities are returned as none if not existen
        if forces is not None and (forces**2).sum() > 1e-12:
            new.set_forces(forces)
        if (cells.std(axis=0).sum()) > 1e-12:
            new.set_cells(cells)
        if timestep_fs is not None:
            new.set_timestep(timestep_fs)
        return new
    
    def get_atom_types(self):
        """Return the LAMMPS atom types if available."""
        if hasattr(self, 'atom_types') and self.atom_types is not None:
            return self.atom_types
        else:
            raise AttributeError("Atom types not available in trajectory")
    
    def set_atom_types(self, atom_types):
        """Set the LAMMPS atom types."""
        self.atom_types = np.array(atom_types)

    def _save_atoms(self, folder_name):
        from os.path import join
        if self._atoms:
            from ase.io import write
            write(join(folder_name, self._ATOMS_FILENAME), self._atoms)

    def get_timestep(self):
        return self.get_attr(self._TIMESTEP_KEY)

    def set_timestep(self, timestep_fs):
        """
        :param timestep: expects value of the timestep in femtoseconds
        """
        self.set_attr(self._TIMESTEP_KEY, float(timestep_fs))

    def get_atoms(self):
        if self._atoms is not None:
            return self._atoms.copy()
        else:
            return None
            # raise ValueError('Atoms have not been set')

    def set_types(self, types):
        types = np.array(types, dtype=str)
        self.set_array(self._TYPES_KEY, types,
                       check_existing=False,
                       check_nat=False, check_nstep=False,)

    def get_types(self):
        if self._TYPES_KEY in self.get_arraynames():
            return self.get_array(self._TYPES_KEY)
        elif self._atoms is not None:
            return np.array(self._atoms.get_chemical_symbols())
        else:
            return None

    @property
    def types(self):
        return self.get_types()

    @property
    def atoms(self):
        return self.get_atoms()

    def set_atoms(self, atoms):
        from ase import Atoms
        if not isinstance(atoms, Atoms):
            raise ValueError('You have to pass an instance of ase.Atoms')
        self._atoms = atoms


    @property
    def nat(self):
        types = self.get_types()
        if types is None:
            raise ValueError('Types have not been set')
        else:
            return len(types)

    @property
    def cell(self):
        return self.atoms.cell

    def set_cells(self, array, check_existing=False):
        self.set_array(self._CELL_KEY, array,
                       check_existing=check_existing,
                       check_nat=False, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_1=3,
                       wanted_shape_2=3)

    def get_cells(self):
        if self._CELL_KEY in self.get_arraynames():
            return self.get_array(self._CELL_KEY)
        return None

    def get_volumes(self):
        cells = self.get_cells()
        if cells is None:
            volume = self.atoms.get_volume()
            return np.array([volume]*self.nstep)
        volumes = [np.linalg.det(cell) for cell in cells]
        return np.array(volumes)

    def get_indices_of_species(self, species, start=0):
        """
            Get array indices for atoms of a specific element type.
            
            This convenience function identifies all atoms of a specified element
            and returns their indices, with optional index offset.
            
            Args:
                species (str or int): Element symbol (e.g., 'Li') or specific atom index
                start (int): Starting index offset, defaults to 0 (Python indexing)
                    Use start=1 for FORTRAN-style indexing
            
            Returns:
                numpy.ndarray: Array of atom indices matching the specified species
                
            Raises:
                TypeError: If species is neither a string nor an integer
                
            Example:
                >>> # Find all lithium atoms
                >>> li_indices = traj.get_indices_of_species('Li')
                >>> print(f"Found {len(li_indices)} Li atoms at indices {li_indices}")
                >>> 
                >>> # Get FORTRAN-style indices (1-based) for oxygen atoms
                >>> o_indices = traj.get_indices_of_species('O', start=1)
            """
        assert isinstance(start, int), 'Start is not an integer'
        types = self.get_types()
        if isinstance(species, str):
            msk = types == species
        elif isinstance(species, int):
            msk = np.zeros(len(types), dtype=bool)
            msk[species] = True
        else:
            raise TypeError('species  has  to be an integer or a string, '
                            'I got {}'.format(type(species)))
        return np.where(msk)[0] + start
    

    def set_positions(self, array, check_existing=False):
        """
        Set atomic positions for the entire trajectory.
        
        Args:
            array (numpy.ndarray): Positions array with shape (nsteps, natoms, 3)
                Values should be in Angstroms
            check_existing (bool): If True, raise error if positions already exist
                Defaults to False (overwrite existing positions)
                
        Raises:
            ValueError: If array shape doesn't match trajectory dimensions
            ValueError: If check_existing=True and positions already exist
            
        Example:
            >>> # Set positions from a numpy array
            >>> positions = np.zeros((100, 50, 3))  # 100 steps, 50 atoms
            >>> traj.set_positions(positions)
        """
        self.set_array(self._POSITIONS_KEY, array,
                       check_existing=check_existing,
                       check_nat=self.nat, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_2=3)

    def get_positions(self):
        return self.get_array(self._POSITIONS_KEY)

    def set_velocities(self, array, check_existing=False):
        """
        Set the velocites of the trajectory.
        :param array:
            A numpy array with the velocites in absolute
            values in units of angstrom/femtoseconds
        :param bool check_exising:
            Check if the velocities have been set, and
            raise in such case. Defaults to False.
        """
        self.set_array(self._VELOCITIES_KEY, array,
                       check_existing=check_existing, check_nat=self.nat,
                       check_nstep=True, wanted_shape_len=3, wanted_shape_2=3)

    def calculate_velocities_from_positions(self, overwrite=False):
        """
        Calculate atomic velocities from position differences between frames.
        
        This method uses finite difference approximation to derive velocities
        from position data, using the velocity Verlet update formula.
        
        Args:
            overwrite (bool): Whether to replace existing velocity data
                Defaults to False (raise exception if velocities exist)
                
        Returns:
            numpy.ndarray: Calculated velocities array
            
        Raises:
            Exception: If velocities already exist and overwrite=False
            
        Note:
            The first and last frames use forward and backward differences,
            while intermediate frames use central differences for better accuracy.
            
        Example:
            >>> # Calculate velocities from positions
            >>> velocities = traj.calculate_velocities_from_positions()
            >>> print(f"Average velocity magnitude: {np.mean(np.linalg.norm(velocities, axis=2)):.4f} Å/fs")
        """
        if self._VELOCITIES_KEY in self.get_arraynames():
            if not overwrite:
                raise Exception("I am overwriting an existing velocity array"
                                "Pass overwrite=True to allow")
        pos = self.get_positions()
        timestep_fs = self.get_timestep()
        vel_first = (pos[1] - pos[0]) / timestep_fs
        vel_last = (pos[-1] - pos[-2]) / timestep_fs
        vel_intermediate = (pos[2:] - pos[:-2]) / (2*timestep_fs)
        vel = np.vstack(([vel_first], vel_intermediate, [vel_last]))
        self.set_velocities(vel)
        return vel.copy()

    def get_velocities(self):
        return self.get_array(self._VELOCITIES_KEY)

    def set_forces(self, array, check_existing=False):
        """
        Set the forces of the trajectory.
        :param array:
            A numpy array with the forces in absolute
            values in units of eV/angstrom
        :param bool check_exising:
            Check if the forces have been set, and raise in
            such case. Defaults to False.
        """
        self.set_array(self._FORCES_KEY, array, check_existing=check_existing,
                       check_nat=self.nat, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_2=3)

    def set_stress(self, array, order='voigt', check_existing=False):
        """
        order voigt expects keys ('xx', 'yy', 'zz', 'yz', 'xz', 'xy')
        """
        if order == 'voigt':
            self.set_array(self._STRESS_KEY, array,
                           check_existing=check_existing, check_nstep=True,
                           wanted_shape_1=6, wanted_shape_len=2)
        else:
            raise ValueError("Not implemented order {}".format(order))

    def get_stress(self):
        return self.get_array(self._STRESS_KEY)

    def get_forces(self):
        return self.get_array(self._FORCES_KEY)

    def set_pot_energies(self, array, check_existing=False):

        self.set_array(self._POT_ENER_KEY, array,
                       check_existing=check_existing,
                       check_nat=False, check_nstep=True,
                       wanted_shape_len=1)

    def get_step_atoms(self, index, ignore_calculated=False,
                       warnings=True):
        """
        Extract a single timestep as an ASE Atoms object.
        
        This method creates an ASE Atoms object for a specific trajectory frame,
        optionally including calculated properties like forces and energies.
        
        Args:
            index (int): Trajectory frame index to extract
            ignore_calculated (bool): If True, don't include calculated properties
                like forces, energies, and stress
            warnings (bool): Whether to show warnings for unavailable data
                
        Returns:
            ase.Atoms: Atoms object representing the system at the specified timestep
            
        Raises:
            AssertionError: If index is not an integer
            
        Example:
            >>> # Extract frame 50 with all calculated properties
            >>> atoms_50 = traj.get_step_atoms(50)
            >>> 
            >>> # Extract frame 100 without calculated properties
            >>> atoms_100 = traj.get_step_atoms(100, ignore_calculated=True)
            >>> 
            >>> # Access properties of the extracted frame
            >>> positions = atoms_50.get_positions()
            >>> forces = atoms_50.get_forces()  # If available
        """
        assert isinstance(index, (int, np.int64)
                          ), "step index has to be an integer"

        need_calculator = False
        if not ignore_calculated:
            for key in (self._FORCES_KEY, self._POT_ENER_KEY):
                if key in self.get_arraynames():
                    need_calculator = True
                    break
        atoms = self.atoms.copy()

        if need_calculator:
            from ase.calculators.singlepoint import SinglePointCalculator
            calc_kwargs = {}

        for k, v in list(self._arrays.items()):
            if k == self._CELL_KEY:
                atoms.set_cell(v[index])
            elif k == self._FORCES_KEY:
                if need_calculator:
                    calc_kwargs['forces'] = v[index]
            elif k == self._POT_ENER_KEY:
                if need_calculator:
                    calc_kwargs['energy'] = v[index]
            elif k == self._STRESS_KEY:
                if need_calculator:
                    calc_kwargs['stress'] = v[index]
            else:
                try:
                    getattr(atoms, 'set_{}'.format(k))(v[index])
                except AttributeError as e:
                    if warnings:
                        print(e)
        if need_calculator:
            calc = SinglePointCalculator(atoms, **calc_kwargs)
            atoms.set_calculator(calc)  # this seems to be deprecated,
            # replace with atoms.calc = calc at somepoint

        return atoms

    def get_ase_trajectory(self, start=0, end=None, stepsize=1):
        """
        Convert a portion of the trajectory to a list of ASE Atoms objects.
        
        This method extracts multiple frames as ASE Atoms objects, with options
        for selecting a range and stride.
        
        Args:
            start (int): First frame index to extract (default: 0)
            end (int): Last frame index to extract (exclusive)
                If None, uses the full trajectory length
            stepsize (int): Frame stride for extraction (default: 1)
                
        Returns:
            list: List of ASE Atoms objects for the requested frames
            
        Raises:
            ValueError: If end exceeds trajectory length or no frames match criteria
            AssertionError: If parameters are not valid integers
            
        Example:
            >>> # Get the first 10 frames
            >>> atoms_list = traj.get_ase_trajectory(end=10)
            >>> 
            >>> # Get every 10th frame of the entire trajectory
            >>> atoms_list = traj.get_ase_trajectory(stepsize=10)
            >>> 
            >>> # Get frames 100-200 with stride 5
            >>> atoms_list = traj.get_ase_trajectory(start=100, end=200, stepsize=5)
            >>> print(f"Extracted {len(atoms_list)} frames")
        """
        if end is None:
            end = self.nstep
        assert isinstance(
            start, int) and start >= 0, "start has to be a positive integer"
        assert isinstance(
            end, int) and end >= 0, "end has to be a positive integer"
        assert isinstance(
            stepsize, int
        ) and stepsize >= 0, "stepsize has to be a positive integer"
        if end > self.nstep:
            raise ValueError(
                "End > nsteps, leave None and it will be set to nstep")
        indices = np.arange(start, end, stepsize)

        if len(indices) < 1:
            raise ValueError("No indices for trajectory")
        assert isinstance(
            start, int) and start >= 0, "start has to be a positive integer"
        atomslist = [self.get_step_atoms(idx) for idx in indices]

        return atomslist

    def recenter(self, sublattice=None, mode=None):
        """
        Recenter the trajectory to remove system drift or center on specific atoms.
        
        This method modifies positions and velocities in-place to recenter the system.
        Useful for removing center-of-mass drift or for centering on a specific
        sublattice (e.g., a rigid framework or specific element).
        
        Args:
            sublattice (list, optional): Element names or indices defining a sublattice
                to center on. If None, centers on the entire system.
            mode (str, optional): Centering mode:
                - None: Center using atomic masses (default)
                - 'geometric': Use geometric center (equal weights)
        
        Raises:
            TypeError: If sublattice is not a list, tuple or set
            IndexError: If sublattice contains invalid atom indices
            TypeError: If sublattice contains items of invalid type
            
        Example:
            >>> # Center on the entire system (remove drift)
            >>> traj.recenter()
            >>> 
            >>> # Center on the Al-O framework
            >>> traj.recenter(sublattice=['Al', 'O'])
            >>> 
            >>> # Center geometrically on Li atoms
            >>> traj.recenter(sublattice=['Li'], mode='geometric')
        """
        from samos_modules.mdutils import recenter_positions, recenter_velocities

        # masses are either set to 1.0 (all) or to the actual masses
        # of the atoms
        # Setting to 1 means that the mass is not accounted for and the
        # the center is purely geometric.
        if mode == 'geometric':
            masses = [1.0] * len(self.atoms)
        else:
            masses = self.atoms.get_masses()

        if sublattice is not None:
            if not isinstance(sublattice, (tuple, list, set)):
                raise TypeError(
                    'You have to pass a tuple/list/set as sublattice')
            factors = [0]*len(masses)
            for item in sublattice:
                if isinstance(item, int):
                    try:
                        factors[item] = 1
                    except IndexError:
                        raise IndexError(
                            'You passed an integer for the sublattice, '
                            'but it is out of range')
                elif isinstance(item, str):
                    for index in self.get_indices_of_species(item):
                        factors[index] = 1
                else:
                    raise TypeError(
                        'You passed {} {} as a sublattice specifier, '
                        'this is not recognized'.format(type(item), item))
        else:
            factors = [1] * len(self.atoms)
        self.set_positions(recenter_positions(self.get_positions(), masses, factors))
        if 'velocities' in self.get_arraynames():
            self.set_velocities(recenter_velocities(
                self.get_velocities(), masses, factors))
