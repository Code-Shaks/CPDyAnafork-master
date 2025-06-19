from ase.io import read
import numpy as np
import sys
import re
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from samos_modules.samos_trajectory import Trajectory

bohr_to_ang = 0.52917720859

def read_positions_with_ase(filename):
    """
    Reads all frames from a trajectory file using ASE and returns
    a numpy array of shape (nsteps, natoms, 3).
    """
    # Read all frames
    images = read(filename, index=':')
    if not isinstance(images, list):
        images = [images]
    positions = np.array([img.get_positions() for img in images])
    return positions

integer_regex = re.compile('(?P<int>\d+)')  # noqa: W605
float_regex = re.compile('(?P<float>[\-]?\d+\.\d+(e[+\-]\d+)?)')  # noqa: W605


def get_indices(header_list, prefix="", postfix=""):
    try:
        idc = np.array([header_list.index(f'{prefix}{dim}{postfix}')
                        for dim in 'xyz'])
    except Exception:
        return False, None
    return True, idc


def get_position_indices(header_list):
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
    Transforming positions to absolute positions
    """
    if postype in ("u", "w", ""):
        return pos
    elif postype == 's':
        return pos.dot(cell)
    else:
        raise RuntimeError(f"Unknown postype '{postype}'")


def get_thermo_props(fname):
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


def read_lammps_dump(filename, elements=None,
                     elements_file=None, types=None, timestep=None,
                     mass_types=None,
                     thermo_file=None, thermo_pe=None, thermo_stress=None,
                     save_extxyz=False, outfile=None,
                     ignore_forces=False, ignore_velocities=False,
                     skip=0, f_conv=1.0, e_conv=1.0, s_conv=1.0,
                     additional_keywords_dump=[], quiet=False,
                     istep=1):
    """
    Read a filedump from lammps.
    It expects atomid to be printed, and positions
    to be given in scaled or unwrapped coordinates
    :param filename: lammps dump file to read
    :param elements: list of elements to use
    :param elements_file:
        file containing elements (separated by space),
        instead of elements
    :param types:
        list of types if elements are not specified
        and type is a column
    :param timestep: timestep of dump (in fs)
    :param thermo_file: file containing thermo output in case required
    :param thermo_pe:
        potential energy column in thermo file (as given in header)
    :param thermo_stress:
        stress column in thermo file (as given in header,
        will do the _xx/_yy etc)
    :param save_extxyz:
        save to extxyz file
        (or if outfile is given with .extxyz)
    :param outfile: output file name, will write trajectory by default
    :param ignore_forces: ignore forces even if written in dump
    :param ignore_velocities: ignore velocities even if written in dump
    :param skip: skip first n steps
    :param f_conv: force conversion factor
    :param e_conv: energy conversion factor
    :param s_conv: stress conversion factor
    :param additional_keywords_dump:
        additional keywords to be added read form dump and
        to be added as array. The column name is used both
        as key but also as arrayname
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
        body = np.array([f.readline().split() for _ in range(nat_must)])
        atomids = np.array(body[:, atomid_idx], dtype=int)
        sorting_key = atomids.argsort()
        # figuring out elements of structure
        if types is not None:
            if type_idx is None:
                raise ValueError("types specified but not found in file")
            types_in_body = np.array(body[:, type_idx][sorting_key], dtype=int)
            print("types in body: {}".format(', '.join(sorted(map(str, set(types_in_body))))))
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


