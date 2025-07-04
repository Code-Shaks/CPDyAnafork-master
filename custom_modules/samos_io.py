"""
I/O Operations Module for SAMOS

This module provides comprehensive input/output functions for the SAMOS (Statistical
Analysis of MOlecular Simulations) package. It handles reading and writing of various
trajectory formats commonly used in molecular dynamics simulations.

Supported Formats:
    - LAMMPS dump files (.lammpstrj)
    - Extended XYZ files (.extxyz)
    - XSF files for visualization (.xsf)
    - ASE-compatible trajectory formats
    - Grid data files for density analysis

Key Features:
    - Format detection and conversion
    - Trajectory preprocessing and filtering
    - Periodic boundary condition handling
    - Unit conversions and coordinate transformations

:author: SAMOS Development Team
:date: 01-02-2024
"""


from ase.io import read
import numpy as np
import sys
import re
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from custom_modules.samos_trajectory import Trajectory

bohr_to_ang = 0.52917720859

def read_positions_with_ase(filename):
    """
    Read atomic positions using ASE with automatic format detection.

    This function provides a unified interface for reading various trajectory
    formats using the ASE library. It automatically detects file formats when
    possible and returns a list of Atoms objects.

    Parameters
    ----------
    filename : str
        Path to the trajectory file.
    format : str, optional
        Explicit format specification. If ``None``, ASE will attempt to auto-detect the format.

    Returns
    -------
    list
        List of ASE Atoms objects for each frame in the trajectory.

    Raises
    ------
    FileNotFoundError
        If the trajectory file cannot be found.
    ValueError
        If the file format cannot be determined or is unsupported by ASE.

    Examples
    --------
    >>> # Read XYZ trajectory with auto-detection
    >>> frames = read_positions_with_ase('trajectory.xyz')
    >>> # Read with explicit format for VASP output
    >>> frames = read_positions_with_ase('OUTCAR', format='vasp-out')
    >>> print(f"Loaded {len(frames)} frames")
    """
    # Read all frames
    images = read(filename, index=':')
    if not isinstance(images, list):
        images = [images]
    positions = np.array([img.get_positions() for img in images])
    return positions

# Fix the regex patterns by using raw strings
integer_regex = re.compile(r'(?P<int>\d+)')  # Fixed: use raw string
float_regex = re.compile(r'(?P<float>[\-]?\d+\.\d+(e[+\-]\d+)?)')  # Fixed: use raw string


def get_indices(header_list, prefix="", postfix=""):
    """
    Get indices of specific fields in a header list.

    This function searches for field indices in a header list based on a prefix
    and postfix pattern, commonly used to identify coordinate or velocity fields
    in trajectory file headers.

    Parameters
    ----------
    header_list : list
        List of header strings to search through.
    prefix : str, optional
        Prefix to match before the field name (e.g., 'v' for velocities). Default is "".
    postfix : str, optional
        Postfix to match after the field name (e.g., 'u' for unwrapped). Default is "".

    Returns
    -------
    tuple
        A tuple of (bool, np.ndarray or None):
        - Boolean indicating if the fields were found.
        - Array of indices for x, y, z components if found, otherwise ``None``.

    Examples
    --------
    >>> # Find position indices in a header list
    >>> headers = ['id', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    >>> found, pos_indices = get_indices(headers, postfix="")
    >>> print(f"Position indices found: {pos_indices}")
    """
    try:
        idc = np.array([header_list.index(f'{prefix}{dim}{postfix}')
                        for dim in 'xyz'])
    except Exception:
        return False, None
    return True, idc


def get_position_indices(header_list):
    """
    Extract position indices from a header list.

    This function identifies the indices of position fields (x, y, z) in a header list
    from trajectory file data, supporting different coordinate types (unwrapped, scaled,
    or wrapped).

    Parameters
    ----------
    header_list : list
        List of header strings from a trajectory file dump (e.g., ['id', 'x', 'y', 'z']).

    Returns
    -------
    tuple
        A tuple of (str, np.ndarray):
        - postfix: String indicating the type of coordinates found ('u' for unwrapped,
          's' for scaled, or '' for wrapped).
        - idc: Array of indices corresponding to x, y, z fields in the header list.

    Raises
    ------
    TypeError
        If no position indices are found in the header list.
    NotImplementedError
        If scaled unwrapped coordinates ('xsu') are present in the header list, as they are not supported.

    Examples
    --------
    >>> # Extract position indices from a header list
    >>> headers = ['id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    >>> postfix, indices = get_position_indices(headers)
    >>> print(f"Position indices: {indices}, Coordinate type: {postfix}")

    Note
    ----
    The function checks for different postfixes ('u', 's', '') to identify coordinate types.
    For scaled ('s') or wrapped ('') coordinates, a warning is printed as unwrapping is not yet implemented.
    """
    # Unwrapped positions u, # scaled positions s
    # wrapped positions given as x y z
    for postfix in ("u", "s", ""):
        found, idc = get_indices(header_list, prefix='',
                                 postfix=postfix)
        if found:
            if postfix in ("s", ""):
                print("Warning: I am not unwrapping positions,"
                      " this is not yet implemented")
            return postfix, idc
    if 'xsu' in header_list:
        raise NotImplementedError("Do not support scaled"
                                  " unwrapped coordinates")

    raise TypeError("No position indices found")


def read_step_info(lines, lidx=0, start=False, additional_kw=[], quiet=False):
    """
    Read time step information from trajectory file lines.

    This function extracts time step numbers and cell dimensions from a block of
    lines corresponding to a single time step in various trajectory file formats,
    primarily LAMMPS dump files.

    Parameters
    ----------
    lines : list of str
        List of lines representing a single time step block from a trajectory file.
    lidx : int, optional
        Line index offset for error messages. Default is 0.
    start : bool, optional
        If ``True``, parse additional header information such as atom IDs and positions.
        Default is ``False``.
    additional_kw : list of str, optional
        Additional keywords to look for in the header line when parsing atom data.
        Default is an empty list.
    quiet : bool, optional
        If ``True``, suppresses print statements for debugging or status updates.
        Default is ``False``.

    Returns
    -------
    tuple
        If `start` is ``False``, returns:
        - nat (int): Number of atoms.
        - timestep (int): Current time step number.
        - cell (np.ndarray): 3x3 array of cell dimensions.

        If `start` is ``True``, returns:
        - nat (int): Number of atoms.
        - atomid_idx (int): Index of atom ID in header.
        - element_idx (int or None): Index of element symbol in header, or ``None`` if not present.
        - type_idx (int or None): Index of atom type in header, or ``None`` if not present.
        - postype (str): Position type postfix ('u' for unwrapped, 's' for scaled, or '' for wrapped).
        - posids (np.ndarray): Indices of position components in header.
        - has_vel (bool): Whether velocities are present.
        - velids (np.ndarray or None): Indices of velocity components if present, otherwise ``None``.
        - has_frc (bool): Whether forces are present.
        - frcids (np.ndarray or None): Indices of force components if present, otherwise ``None``.
        - additional_ids (list): Indices of additional keywords if present.

    Raises
    ------
    Exception
        If the lines do not start with expected headers (e.g., 'ITEM: TIMESTEP') or required data is missing.
    ValueError
        If cell dimensions, atom counts, or other critical data cannot be parsed correctly.

    Examples
    --------
    >>> # Read time step information from the first block of a LAMMPS dump file
    >>> with open('dump.lammpstrj', 'r') as f:
    ...     lines = [next(f) for _ in range(9)]
    >>> nat, timestep, cell = read_step_info(lines)
    >>> print(f"Timestep: {timestep}, Number of atoms: {nat}")

    Note
    ----
    This function is designed to parse LAMMPS dump file formats and expects a specific
    structure with headers like 'ITEM: TIMESTEP', 'ITEM: NUMBER OF ATOMS', and 'ITEM: BOX BOUNDS'.
    It supports both orthogonal and triclinic cells.
    """
    
    assert len(lines) == 9
    if not lines[0].startswith("ITEM: TIMESTEP"):
        raise Exception("Did not start with 'ITEM: TIMESTEP'\n"
                        "Is this not a valid lammps dump file?")
    try:
        timestep = int(integer_regex.search(lines[1]).group('int'))
    except Exception as e:
        print("Timestep is not an integer or was not found in "
              f"line {lidx+1} ({lines[1]})")
        raise e
    if not lines[2].startswith("ITEM: NUMBER OF ATOMS"):
        raise Exception("Not a valid lammps dump file, "
                        "expected NUMBER OF ATOMS")
    try:
        nat = int(integer_regex.search(lines[3]).group('int'))
    except Exception as e:
        print("Could not read number of atoms")
        raise e
    cell = np.zeros((3, 3))
    if lines[4].startswith("ITEM: BOX BOUNDS pp pp pp"):
        try:
            for idim in range(3):
                d1, d2 = [float(m.group('float'))
                          for m in float_regex.finditer(lines[5+idim])]
                cell[idim, idim] = d2 - d1
        except Exception as e:
            print(f"Could not read cell dimension {idim}")
            raise e
    elif lines[4].startswith("ITEM: BOX BOUNDS xy xz yz pp pp pp"):
        try:
            # see https://docs.lammps.org/dump.html
            # and https://docs.lammps.org/Howto_triclinic.html
            xlo, xhi, xy = [float(m.group('float'))
                            for m in float_regex.finditer(lines[5])]
            ylo, yhi, xz = [float(m.group('float'))
                            for m in float_regex.finditer(lines[6])]
            zlo, zhi, yz = [float(m.group('float'))
                            for m in float_regex.finditer(lines[7])]
            cell[0, 0] = xhi - xlo
            cell[1, 1] = yhi - ylo
            cell[2, 2] = zhi - zlo
            cell[1, 0] = xy
            cell[2, 0] = xz
            cell[2, 1] = yz
        except Exception as e:
            print(f"Could not read cell dimension {idim}")
            raise e
    else:
        raise ValueError("unsupported lammps dump file, "
                         "expected  BOX BOUNDS pp pp pp or "
                         "BOX BOUNDS xy xz yz pp pp pp")
    if start:
        if not quiet:
            print(f"Read starting cell as:\n{cell}")
        if not lines[8].startswith("ITEM: ATOMS"):
            raise Exception("Not a supported format, expected ITEM: ATOMS")
        header_list = lines[8].lstrip("ITEM: ATOMS").split()

        atomid_idx = header_list.index('id')
        if not quiet:
            print(f'Atom ID index: {atomid_idx}')
        try:
            element_idx = header_list.index('element')
            if not quiet:
                print("Element found at index {element_idx}")
        except ValueError:
            element_idx = None
        try:
            type_idx = header_list.index('type')
            if not quiet: print("type found at index {type_idx}")
        except ValueError:
            type_idx = None
        try:
            postype, posids = get_position_indices(header_list)
        except Exception:
            print("Abandoning because positions are not given")
            sys.exit(1)
        if not quiet:
            print("Positions are given as: {}".format(
                {'u': "unwrapped", 's': "Scaled (wrapped)",
                "": "Wrapped"}[postype]))
        if not quiet: print("Position indices are: {}".format(posids))
        has_vel, velids = get_indices(header_list, 'v')
        if has_vel:
            if not quiet: print("Velocities found at indices: {}".format(velids))
        else:
            if not quiet: print("Velocities were not found")
        has_frc, frcids = get_indices(header_list, 'f')
        if has_frc:
            if not quiet: print("Forces found at indices: {}".format(frcids))
        else:
            if not quiet: print("Forces were not found")
        additional_ids = []
        if additional_kw:
            for kw in additional_kw:
                additional_ids.append(header_list.index(kw))

        return (nat, atomid_idx, element_idx, type_idx, postype,
                posids, has_vel, velids, has_frc, frcids, additional_ids)
    else:
        return nat, timestep, cell


def pos_2_absolute(cell, pos, postype):
    """
    Convert fractional coordinates to absolute (Cartesian) coordinates.

    This function transforms positions from fractional (scaled) coordinates to
    absolute Cartesian coordinates using the provided unit cell vectors. It handles
    different position types based on the `postype` parameter.

    Parameters
    ----------
    cell : np.ndarray
        Unit cell vectors with shape (3, 3). Each row represents one lattice vector [a, b, c].
    pos : np.ndarray
        Position array with shape (..., 3). Can be in fractional (scaled) or absolute coordinates
        depending on `postype`.
    postype : str
        Type of position coordinates:
        - 'u' or 'w' or '': Already absolute (unwrapped or wrapped), returned as is.
        - 's': Scaled (fractional), converted to absolute using the cell vectors.

    Returns
    -------
    np.ndarray
        Positions in Cartesian coordinates with the same shape as the input `pos`.

    Raises
    ------
    RuntimeError
        If `postype` is an unknown value (not 'u', 'w', '', or 's').

    Examples
    --------
    >>> # Convert from fractional to Cartesian coordinates
    >>> frac_pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    >>> cell = np.array([[10.0, 0.0, 0.0],
    ...                  [0.0, 10.0, 0.0],
    ...                  [0.0, 0.0, 10.0]])
    >>> cart_pos = pos_2_absolute(cell, frac_pos, postype='s')
    >>> print(cart_pos)  # Output: array([[0., 0., 0.], [5., 5., 5.]])

    Note
    ----
    This function is typically used when reading trajectory data from formats like LAMMPS
    dump files, where positions may be provided in scaled (fractional) coordinates that
    need conversion to absolute coordinates for analysis.
    """
    if postype in ("u", "w", ""):
        return pos
    elif postype == 's':
        return pos.dot(cell)
    else:
        raise RuntimeError(f"Unknown postype '{postype}'")


def get_thermo_props(fname):
    """
    Extract thermodynamic properties from simulation output files.

    This function parses molecular dynamics (MD) output files to extract thermodynamic
    quantities such as temperature, pressure, energy, and volume from a log file format
    with a header line and data array.

    Parameters
    ----------
    fname : str
        Path to the simulation output file (e.g., a LAMMPS log file).

    Returns
    -------
    tuple
        A tuple containing:
        - header (list of str): List of column names from the file header.
        - arr (np.ndarray): 2D array of data values from the file, with shape (n_steps, n_columns).
        - ts_index (int): Index of the 'TimeStep' column in the header and data array.

    Raises
    ------
    FileNotFoundError
        If the specified output file cannot be found.
    ValueError
        If the file is empty, the header cannot be parsed, or the data array cannot be loaded.

    Examples
    --------
    >>> # Extract thermodynamic properties from a log file
    >>> header, data, ts_idx = get_thermo_props('md.log')
    >>> timesteps = data[:, ts_idx]
    >>> print(f"Read {len(timesteps)} time steps from the log file")

    Note
    ----
    This function assumes a specific log file format with a header line starting with '#'
    followed by space-separated column names, and data lines starting from the third line.
    It uses `np.loadtxt` to parse the data, which expects numerical values.
    """
    with open(fname) as f:
        f.readline()  # first line
        header = f.readline().lstrip('#').strip().split()
    # header = [h.lstrip('v_').lstrip('c_') for h in header]
    arr = np.loadtxt(fname, skiprows=2)
    if len(arr.shape) == 1:
        # special case if only one line is present
        arr = np.array([arr])
    ts_index = header.index('TimeStep')
    return header, arr, ts_index


def read_lammps_dump(filename, elements=None, elements_file=None, types=None, timestep=None,
                     mass_types=None, thermo_file=None, thermo_pe=None, thermo_stress=None,
                     save_extxyz=False, outfile=None, ignore_forces=False, ignore_velocities=False,
                     skip=0, f_conv=1.0, e_conv=1.0, s_conv=1.0,
                     additional_keywords_dump=[], quiet=False, istep=1,
                     return_sorted_atom_ids=False):
    """
    Read LAMMPS dump file and convert to ASE trajectory format.

    This function parses LAMMPS dump files containing atomic positions and converts
    them to ASE Atoms objects. It handles periodic boundary conditions, atom ordering,
    and provides options for format conversion.

    Parameters
    ----------
    filename : str
        Path to LAMMPS dump file (.lammpstrj format).
    elements : list, optional
        List of element symbols in order of atom types. Example: ['Li', 'Al', 'P', 'S', 'O'] for types 1-5.
    elements_file : str, optional
        Path to a file containing space-separated element symbols.
    types : list, optional
        List of type mappings to element symbols if type field is present in dump.
    timestep : float, optional
        Time step between frames in picoseconds. Default is ``None`` (not set).
    mass_types : str, optional
        Path to LAMMPS input file with mass definitions to infer elements.
    thermo_file : str, optional
        Path to thermodynamic output file for additional properties.
    thermo_pe : str, optional
        Thermo keyword for potential energy in `thermo_file`.
    thermo_stress : str, optional
        Thermo keyword prefix for stress components (without xx/yy/zz suffixes).
    save_extxyz : bool, optional
        Whether to save trajectory in extended XYZ format. Default is ``False``.
    outfile : str, optional
        Output filename for saved trajectory. If ``None`` and `save_extxyz=True`, uses input filename with .extxyz extension.
    ignore_forces : bool, optional
        If ``True``, ignores force data in dump file. Default is ``False``.
    ignore_velocities : bool, optional
        If ``True``, ignores velocity data in dump file. Default is ``False``.
    skip : int, optional
        Number of initial steps to skip. Default is 0.
    f_conv : float, optional
        Conversion factor for forces. Default is 1.0.
    e_conv : float, optional
        Conversion factor for energies. Default is 1.0.
    s_conv : float, optional
        Conversion factor for stresses. Default is 1.0.
    additional_keywords_dump : list, optional
        Additional keywords to read from dump file. Default is empty list.
    quiet : bool, optional
        If ``True``, suppresses output messages. Default is ``False``.
    istep : int, optional
        Frequency of steps to take from trajectory. Default is 1.

    Returns
    -------
    Trajectory
        A SAMOS Trajectory object representing the parsed trajectory frames with positions, cells, and optional velocities/forces.

    Raises
    ------
    FileNotFoundError
        If the LAMMPS dump file or related files (e.g., `elements_file`, `thermo_file`) cannot be found.
    ValueError
        If the file format is invalid, element list is inconsistent, or thermodynamic data mismatches timesteps.
    NotImplementedError
        If certain features (e.g., scaled unwrapped coordinates) are not supported.

    Examples
    --------
    >>> # Read LAMMPS trajectory with 5 element types
    >>> elements = ['Li', 'Al', 'P', 'S', 'O']
    >>> traj = read_lammps_dump('dump.lammpstrj', elements, timestep=0.001)
    >>> print(f"Read {len(traj.get_positions())} frames with {len(traj.atoms)} atoms each")
    >>> # Save as extended XYZ for further analysis
    >>> traj = read_lammps_dump('dump.lammpstrj', elements, save_extxyz=True, outfile='trajectory.extxyz')

    Note
    ----
    - LAMMPS atom types are mapped to element symbols based on the order in the `elements` list or inferred from masses if `mass_types` is provided.
    - Periodic boundary conditions are automatically handled.
    - The function assumes a standard LAMMPS dump format with id, type, x, y, z columns.
    """
    # opening a first time to check file and get
    # indices of positions, velocities etc...
    with open(filename) as f:
        # first doing a check
        lines = [next(f) for _ in range(9)]
        (nat_must, atomid_idx, element_idx, type_idx,
         postype, posids, has_vel, velids,
         has_frc, frcids, additional_ids_dump) = read_step_info(
            lines, lidx=0, start=True,
            additional_kw=additional_keywords_dump, quiet=quiet)

        if ignore_forces:
            has_frc = False
        if ignore_velocities:
            has_vel = False
        # these are read as strings
        atom_types_for_trajectory = None
        body = np.array([f.readline().split() for _ in range(nat_must)])
        atomids = np.array(body[:, atomid_idx], dtype=int)
        sorting_key = atomids.argsort()
        # figuring out elements of structure
        if types is not None:
            if type_idx is None:
                raise ValueError("types specified but not found in file")
            types_in_body = np.array(body[:, type_idx][sorting_key], dtype=int)
            print("types in body: {}".format(', '.join(sorted(map(str, set(types_in_body))))))
            atom_types_for_trajectory = types_in_body.copy()  # Store for trajectory
            types_in_body -= 1  # 1-based to 0-based indexing
            symbols = np.array(types, dtype=str)[types_in_body]
        elif element_idx is not None:
            # readingin elements frmo body
            symbols = np.array(body[:, element_idx])[sorting_key]
            # print(elements)
            # print(len(elements))
        elif elements is not None:
            assert len(elements) == nat_must
            symbols = elements[:]
        elif elements_file is not None:
            with open(elements_file) as f:
                for line in f:
                    if line:
                        break
                elements = line.strip().split()
                if len(elements) != nat_must:
                    raise ValueError(
                        f"length of list of elements ({len(elements)}) "
                        f"is not equal number of atoms ({nat_must})")
                symbols = elements[:]
        elif mass_types is not None:
            # reading in masses from lammps input file stored in mass_types
            # and using these to infer elements
            with open(mass_types) as fmass:
                masses = []
                reading_masses = False
                for line in fmass:
                    line = line.strip()
                    if not line: continue
                    if reading_masses:
                        try:
                            typ, mass = line.split()[:2]
                            masses.append((int(typ), float(mass)))
                        except ValueError:
                            break
                    elif line.startswith('Masses'):
                        reading_masses = True
                    else:
                        pass
            type_indices, masses = zip(*masses)
            masses = np.array(masses, dtype=float)
            type_indices = np.array(type_indices, dtype=int)
            # small check whether all types are present
            if not np.all(np.arange(1, type_indices.max()+1) == type_indices):
                raise ValueError("Types are not consecutive")
            # trying to figure out elements
            if type_idx is None:
                raise ValueError("types specified but not found in file")
            types = []
            for mass in masses:
                # finding the closest element based on its mass
                idx = np.argmin(np.abs(atomic_masses - mass))
                types.append(chemical_symbols[idx])
            types_in_body = np.array(body[:, type_idx][sorting_key], dtype=int)
            print("types in body: {}".format(', '.join(sorted(map(str, set(types_in_body))))))
            print("symbols: {}".format(', '.join(types)))
            types_in_body -= 1  # 1-based to 0-based indexing
            symbols = np.array(types, dtype=str)[types_in_body]


        else:
            # last resort, setting everything to Hydrogen
            symbols = ['H']*nat_must

    positions = []
    timesteps = []
    cells = []
    if has_vel:
        velocities = []
    if has_frc:
        forces = []

    lidx = 0
    iframe = 0
    # dealing with additional kwywods
    additional_arrays = {kw: [] for kw in additional_keywords_dump}

    with open(filename) as f:
        while True:
            step_info = [f.readline() for _ in range(9)]
            if ''.join(step_info) == '':
                if not quiet:
                    print(f"End reached at line {lidx}, stopping")
                break
            nat, timestep_, cell = read_step_info(
                step_info, lidx=lidx, start=False, quiet=quiet)
            lidx += 9
            if nat != nat_must:
                print("Changing number of atoms is not supported, breaking")
                break

            # these are read as strings
            body = np.array([f.readline().split() for _ in range(nat_must)])
            lidx += nat_must
            atomids = np.array(body[:, atomid_idx], dtype=int)
            sorting_key = atomids.argsort()
            pos = np.array(body[:, posids], dtype=float)[sorting_key]
            if iframe >= skip and iframe % istep == 0:
                positions.append(pos_2_absolute(cell, pos, postype))
                timesteps.append(timestep_)
                cells.append(cell)
                if has_vel:
                    velocities.append(np.array(body[:, velids],
                                               dtype=float)[sorting_key])
                if has_frc:
                    forces.append(f_conv*np.array(body[:, frcids],
                                                  dtype=float)[sorting_key])
                for kw, idx_add in zip(additional_keywords_dump,
                                       additional_ids_dump):
                    additional_arrays[kw].append(
                        np.array(body[:, idx_add],
                                 dtype=float)[sorting_key])
            iframe += 1
    if not quiet:
        print(f"Read trajectory of length {iframe}\n"
            f"Creating Trajectory of length {len(timesteps)}")
    try:
        atoms = Atoms(symbols, positions[0], cell=cells[0], pbc=True)
        traj = Trajectory(atoms=atoms,
                          positions=positions, cells=cells)
    except KeyError:
        traj = Trajectory(types=symbols, cells=cells)
        traj.set_positions(positions)
    if atom_types_for_trajectory is not None:
        traj.atom_types = atom_types_for_trajectory
    if has_vel:
        traj.set_velocities(velocities)
    if has_frc:
        traj.set_forces(forces)
    if timestep:
        traj.set_timestep(timestep)
    for key, arr in additional_arrays.items():
        traj.set_array(key, np.array(arr))
    if thermo_file:
        header, arr, ts_index = get_thermo_props(thermo_file)
        timesteps_thermo = np.array(arr[:, ts_index], dtype=int).tolist()
        indices = []
        for ts in timesteps:
            try:
                indices.append(timesteps_thermo.index(ts))
            except ValueError:
                raise ValueError(f"Index {ts} is not in thermo file")
        indices = np.array(indices, dtype=int)
        # if thermo_te:
        #     colidx = header.index(thermo_te)
        #     traj.set_total_energies(arr[indices, colidx])
        if thermo_pe:
            colidx = header.index(thermo_pe)
            traj.set_pot_energies(e_conv*arr[indices, colidx])
        if thermo_stress:
            stressall = []
            # voigt notation for stress:
            keys = ('xx', 'yy', 'zz', 'yz', 'xz', 'xy')
            # first diagonal terms:
            for key in keys:
                fullkey = thermo_stress + key
                colidx = header.index(fullkey)
                stressall.append(s_conv*arr[indices, colidx])
            traj.set_stress(np.array(stressall).T)
        # if thermo_ke:
        #     colidx = header.index(thermo_ke)
        #     traj.set_kinetic_energies(arr[indices, colidx])
    if save_extxyz or (outfile is not None and outfile.endswith('extxyz')):
        from ase.io import write
        path_to_save = outfile or filename + '.extxyz'
        asetraj = traj.get_ase_trajectory()
        write(path_to_save, asetraj, format='extxyz')
    elif outfile:
        path_to_save = outfile or filename + '.traj'
        traj.save(path_to_save)
    return traj

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(
        """Reads lammps input and returns/saves a Trajectory instance""")
    parser.add_argument('filename',
                        help='The filename/path of the lammps trajectory')
    parser.add_argument('-o', '--outfile',
                        help='The filename/path to save trajectory at')
    parser.add_argument('-t', '--types', nargs='+',
                        help=('list of types, will be matched with'
                              ' types given in lammps'))
    parser.add_argument('-e', '--elements', nargs='+',
                        help='list of elements')
    parser.add_argument('--elements-file',
                        help=('A file containing the elements '
                              'as space-separated strings'))
    parser.add_argument('--mass-types',
                        help=('The input file of lammps containing '
                                'the masses of the types'))
    parser.add_argument('--timestep', type=float,
                        help='The timestep of the trajectory printed')
    parser.add_argument('--f-conv', type=float,
                        help='The conversion factor for forces',
                        default=1.0)
    parser.add_argument('--e-conv', type=float,
                        help='The conversion factor for energies',
                        default=1.0)
    parser.add_argument('--s-conv', type=float,
                        help='The conversion factor for stresses',
                        default=1.0)
    parser.add_argument('-i', '--istep', type=int, default=1,
                        help='Just take this frequency of steps from the trajectory')
    parser.add_argument(
        '--thermo-file', help='File path to equivalent thermo-file')
    parser.add_argument(
        '--thermo-pe',
        help='Thermo keyword for potential energy',)
    parser.add_argument('--thermo-stress',
                        help=('Thermo keyword for stress '
                              'without the xx/yy/xz..'))
    parser.add_argument('--save-extxyz',
                        action='store_true',
                        help='save extxyz instead of traj')
    parser.add_argument('--ignore-velocities',
                        action='store_true',
                        help='Ignore velocities in dump file')
    parser.add_argument('--ignore-forces', action='store_true',
                        help='Ignore forces in dump file')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip this many first steps')
    parser.add_argument('-a', '--additional-keywords-dump', nargs='+',
                        help=('Additional keywords to be read from dump file'),
                        default=[])
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not print anything')
    args = parser.parse_args()
    read_lammps_dump(**vars(args))

def read_xsf(filename, fold_positions=False):
    """
    Read XSF file and return ASE Atoms objects.

    This function parses XSF files and converts them to ASE format for further
    analysis. It handles both single structures and animated (multi-frame) XSF files.

    Parameters
    ----------
    filename : str
        Path to the XSF file.
    fold_positions : bool, optional
        If ``True``, folds atomic positions into the unit cell using periodic boundary conditions. Default is ``False``.

    Returns
    -------
    dict
        Dictionary containing:
        - data: 3D grid data as a numpy array.
        - volume_ang: Volume of the unit cell in Ångstroms³.
        - volume_au: Volume of the unit cell in atomic units (Bohr³).
        - atoms: List of atomic symbols.
        - positions: List of atomic positions.
        - cell: Unit cell vectors as a 3x3 array.

    Raises
    ------
    FileNotFoundError
        If the XSF file cannot be found.
    Exception
        If no cell is read from the XSF file or the file format is invalid.

    Examples
    --------
    >>> # Read an XSF file for a crystal structure
    >>> xsf_data = read_xsf('crystal.xsf')
    >>> print(f"Read structure with {len(xsf_data['atoms'])} atoms")
    >>> print(f"Cell volume: {xsf_data['volume_ang']:.2f} Å³")
    """
    finished = False
    skip_lines = 0
    reading_grid = False
    reading_dims = False
    reading_structure = False
    reading_nat = False
    reading_cell = False
    x, y, z, xdim, ydim, zdim = 0, 0, 0, 0, 0, 0
    rho_of_r, atoms, positions, cell = [], [], [], []
    with open(filename) as f:
        finished = False
        for line in f.readlines():
            if reading_grid:
                try:
                    for value in line.split():
                        if x != xdim-1 and y != ydim-1 and z != zdim-1:
                            rho_of_r[x, y, z] = float(value)
                        # Move on to next gridpoint
                        x += 1
                        if x == xdim:
                            x = 0
                            y += 1
                        if y == ydim:
                            z += 1
                            y = 0
                        if z == zdim-1:
                            finished = True
                            break

                except ValueError:
                    break
            elif skip_lines:
                skip_lines -= 1
            elif reading_structure:
                pos = list(map(float, line.split()[1:]))
                if len(pos) != 3:
                    reading_structure = False
                else:
                    atoms.append(line.split()[0])
                    positions.append(pos)
            elif reading_nat:
                nat, _ = list(map(int, line.split()))
                reading_nat = False
                reading_structure = True
            elif reading_cell:
                cell.append(list(map(float, line.split())))
                if len(cell) == 3:
                    reading_cell = False
                    reading_grid = True
            elif reading_dims:
                xdim, ydim, zdim = list(map(int, line.split()))
                rho_of_r = np.zeros([xdim-1, ydim-1, zdim-1])
                # ~ data2 = np.empty(xdim*ydim*zdim)
                # ~ iiii=0
                reading_dims = False
                reading_cell = True
                skip_lines = 1
            elif 'DATAGRID_3D_UNKNOWN' in line:
                x = 0
                y = 0
                z = 0
                cell = []
                reading_dims = True
            elif 'PRIMCOORD' in line:
                atoms = []
                positions = []
                reading_nat = True
            if finished:
                break

    try:
        volume_ang = np.dot(np.cross(cell[0], cell[1]), cell[2])
    except UnboundLocalError:
        raise Exception('No cell was read in XSF file, stopping')
    volume_au = volume_ang / bohr_to_ang**3

    if fold_positions:
        invcell = np.matrix(cell).T.I
        cell = np.array(cell)
        for idx, pos in enumerate(positions):
            # point in crystal coordinates
            points_in_crystal = np.dot(invcell, pos).tolist()[0]
            # point collapsed into unit cell
            points_in_unit_cell = [i % 1 for i in points_in_crystal]
            positions[idx] = np.dot(cell.T, points_in_unit_cell)

    return dict(
        data=rho_of_r, volume_ang=volume_ang, volume_au=volume_au,
        atoms=atoms, positions=positions, cell=cell
    )


def write_xsf(
        atoms, positions, cell, data,
        vals_per_line=6, outfilename=None,
        is_flattened=False, shape=None,):
    """
    Write trajectory to XSF format for visualization.

    XSF (XCrySDen Structure File) format is commonly used for crystal structure
    visualization. This function writes a trajectory as an XSF file that can be
    viewed in programs like XCrySDen or VESTA, including structural data and
    optional 3D grid data for scalar fields.

    Parameters
    ----------
    atoms : list
        List of atomic symbols corresponding to the atoms in the structure.
    positions : np.ndarray
        Array of atomic positions with shape (N, 3) where N is the number of atoms.
    cell : np.ndarray
        Unit cell vectors as a 3x3 array representing the simulation box.
    data : np.ndarray
        3D grid data for the scalar field (e.g., electron density) to be visualized.
    vals_per_line : int, optional
        Number of values per line in the output file for grid data. Default is 6.
    outfilename : str or None, optional
        Output XSF filename. If ``None``, writes to stdout. Default is ``None``.
    is_flattened : bool, optional
        Whether the `data` array is flattened. Default is ``False``.
    shape : tuple, optional
        Original shape of the `data` if flattened (required if `is_flattened` is ``True``).
        Default is ``None``.

    Returns
    -------
    None
        Writes the XSF formatted data to the specified file or stdout.

    Raises
    ------
    Exception
        If the output file cannot be opened or if `shape` is not provided for flattened data.
    IOError
        If there are issues writing to the specified `outfilename`.

    Examples
    --------
    >>> # Write XSF file for visualization of a crystal structure with density data
    >>> atoms = ['Li', 'O', 'O']
    >>> positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
    >>> cell = np.eye(3) * 10.0
    >>> data = np.random.rand(10, 10, 10)
    >>> write_xsf(atoms, positions, cell, data, outfilename='output.xsf')

    Note
    ----
    XSF files support both structural data (atoms and positions) and associated scalar
    fields (like electron density) on a 3D grid. This function handles both, formatting
    the data according to the XSF specification for visualization in compatible software.
    """
    if isinstance(outfilename, str):
        f = open(outfilename, 'w')
    elif outfilename is None:
        f = sys.stdout
    else:
        raise Exception('No file')

    if is_flattened:
        try:
            xdim, ydim, zdim = shape
        except (TypeError, ValueError):
            raise Exception(
                'if you pass a flattend array you '
                'need to give the original shape')
    else:
        xdim, ydim, zdim = data.shape
        shape = data.shape
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
""".format(*[i+1 for i in shape]))
    for row in cell:
        f.write('    {}\n'.format('    '.join(
            ['{:.9f}'.format(r) for r in row])))
    col = 1
    if is_flattened:
        for val in data:
            f.write('  {:0.4E}'.format(val))
            if col < vals_per_line:
                col += 1
            else:
                f.write('\n')
                col = 1
    else:
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


def write_grid(data, outfilename=None, vals_per_line=5,):
    """
    Write 3D grid data to file for visualization or further analysis.

    This function outputs 3D scalar field data (such as electron density or ionic
    density) in a simple text format suitable for visualization programs or further
    processing.

    Parameters
    ----------
    data : np.ndarray
        3D grid data with shape (nx, ny, nz) containing the scalar field values.
    outfilename : str or None, optional
        Output filename for the grid data. If ``None``, writes to stdout. Default is ``None``.
    vals_per_line : int, optional
        Number of values per line in the output file. Default is 5.

    Returns
    -------
    None
        Writes the grid data to the specified file or stdout.

    Raises
    ------
    Exception
        If the output file cannot be opened or if `outfilename` is invalid.
    IOError
        If there are issues writing to the specified `outfilename`.

    Examples
    --------
    >>> # Write a random 3D density grid to a file for visualization
    >>> import numpy as np
    >>> density = np.random.random((50, 50, 50))
    >>> write_grid(density, outfilename='density.grid', vals_per_line=5)

    Note
    ----
    The output format is a simple text file with the first line indicating the grid
    dimensions (nx+1, ny+1, nz+1) followed by the grid data values in scientific notation,
    ordered by z, y, x indices. This format is compatible with certain visualization tools
    or can be post-processed for other formats like XSF.
    """
    xdim, ydim, zdim = data.shape
    if isinstance(outfilename, str):
        f = open(outfilename, 'w')
    elif outfilename is None:
        f = sys.stdout
    else:
        raise Exception('No file')

    xdim, ydim, zdim = data.shape
    f.write('3         {}         {}         {}\n'.format(
        *[i+1 for i in data.shape]))
    col = 0
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                f.write('  {:0.4E}'.format(data[x, y, z]))
                if col < vals_per_line:
                    col += 1
                else:
                    f.write('\n')
                    col = 0
    if col:
        f.write('\n')
    f.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(
        "Reads and writes an XSF file or a data file.\n"
        "python temp.xsf -o grid.xyz")
    p.add_argument('file', type=str)
    p.add_argument('--format', choices=['xsf', 'grid', 'none'],
                   default='grid',
                   help='whether to print the output in xsf or grid format')
    p.add_argument('-o', '--output',
                   help='The name of the output file, default to sys.out')
    p.add_argument(
        '--min', help='print minimum grid value and exit', action='store_true')
    p.add_argument(
        '--max', help='print maximum grid value and exit', action='store_true')

    pa = p.parse_args(sys.argv[1:])
    r = read_xsf(filename=pa.file)
    if pa.min:
        print(r['data'].min())
    elif pa.max:
        print(r['data'].max())
    elif pa.format == 'grid':
        write_grid(outfilename=pa.output, **r)
    elif pa.format == 'xsf':
        write_xsf(outfilename=pa.output, **r)
    elif pa.format == 'none':
        pass
    else:
        raise Exception('unknown format {}'.format(pa.format))


