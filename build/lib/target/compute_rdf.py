from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from samos.trajectory import Trajectory
from samos.analysis.rdf import RDF
from samos.plotting.plot_rdf import plot_rdf
import os

def read_custom_file(filename, nat=52):
    """
    Custom parser for .pos, .in, .cel, or .evp files.
    Currently handles .pos for trajectory and .in for cell parameters and species.
    Returns a Trajectory object compatible with samos.
    """
    import os
    import numpy as np
    from samos.trajectory import Trajectory

    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in ['.pos', '.in', '.cel', '.evp']:
        raise ValueError(f"Unsupported file extension: {file_ext}. Supported: .pos, .in, .cel, .evp")

    # Default values
    cell_parameters = None
    species = []
    frames = []

    if file_ext == '.pos':
        # Parse .pos file for trajectory data
        with open(filename) as f:
            lines = f.readlines()
        non_comment_lines = [l for l in lines if not l.startswith('#') and l.strip()]
        i = 0
        while i < len(non_comment_lines):
            line = non_comment_lines[i].strip()
            parts = line.split()
            if len(parts) >= 2:
                try:
                    timestamp = float(parts[1])
                    positions = []
                    for j in range(1, nat + 1):
                        if i + j < len(non_comment_lines):
                            pos_line = non_comment_lines[i + j].strip().split()
                            if len(pos_line) >= 3:
                                try:
                                    pos = [float(pos_line[k]) for k in range(3)]
                                    positions.append(pos)
                                except ValueError:
                                    break
                            else:
                                break
                        else:
                            break
                    if len(positions) == nat:
                        frames.append((timestamp, positions))
                        i += nat + 1
                    else:
                        i += 1
                except ValueError:
                    i += 1
            else:
                i += 1
        if not frames:
            raise ValueError("No valid frames found in .pos file")
        # Cell parameters (placeholder from earlier query)
        cell_parameters = np.array([
            [-0.107997170, -8.458155452, 0.044216056],
            [8.777469622, 0.026326306, 0.068956511],
            [-0.086091025, 0.096523928, 13.810239380]
        ])
        # Explicitly set species based on earlier query data (Li:22, Al:2, P:4, S:24)
        species = ['Li']*22 + ['Al']*2 + ['P']*4 + ['S']*24
        print("Species set for .pos file based on default composition (Li:22, Al:2, P:4, S:24)")

    elif file_ext == '.in':
        # Parse .in file for cell parameters and species (single frame)
        with open(filename) as f:
            lines = f.readlines()
        cell_section = False
        atomic_pos_section = False
        positions = []
        species = []
        cell_parameters = []
        for line in lines:
            line = line.strip()
            if line.startswith('CELL_PARAMETERS'):
                cell_section = True
                continue
            elif line.startswith('ATOMIC_POSITIONS'):
                atomic_pos_section = True
                continue
            elif line.startswith('/') or line.startswith('&'):
                cell_section = False
                atomic_pos_section = False
            elif cell_section and line:
                parts = line.split()
                if len(parts) == 3:
                    cell_parameters.append([float(p) for p in parts])
            elif atomic_pos_section and line:
                parts = line.split()
                if len(parts) >= 4:
                    species.append(parts[0])
                    positions.append([float(p) for p in parts[1:4]])
        if not cell_parameters or len(cell_parameters) != 3:
            raise ValueError("Could not parse CELL_PARAMETERS from .in file")
        if not positions or len(positions) != nat:
            raise ValueError("Could not parse ATOMIC_POSITIONS from .in file")
        cell_parameters = np.array(cell_parameters)
        frames = [(0.0, positions)]  # Single frame with dummy timestamp
        print(f"Species parsed from .in file: {species}")

    else:  # .cel or .evp
        raise NotImplementedError(f"Parsing for {file_ext} files is not implemented yet. Please provide format details.")

    # Construct trajectory with strict order
    nt = len(frames)
    positions_arr = np.zeros((nt, nat, 3), dtype=float)
    for frame_idx, (timestamp, pos) in enumerate(frames):
        for atom_idx, p in enumerate(pos):
            positions_arr[frame_idx, atom_idx, :] = p

    traj = Trajectory()
    # Set types FIRST to avoid 'Types have not been set' error
    traj.set_types(species)
    # Then set positions
    traj.set_positions(positions_arr)
    # Adjust cell_parameters to shape (nt, 3, 3) for set_cells
    if nt > 1:
        # Repeat the cell parameters for each frame if multiple frames
        cell_parameters_expanded = np.tile(cell_parameters[np.newaxis, :, :], (nt, 1, 1))
    else:
        # For single frame, add a time dimension
        cell_parameters_expanded = cell_parameters[np.newaxis, :, :]
    print(f"Setting cell parameters with shape: {cell_parameters_expanded.shape}")
    # Use set_cells with the correct shape
    traj.set_cells(cell_parameters_expanded)
    return traj

def main():
    # Update filename to point to a .pos or .in file
    filename = 'LiAlPS.pos'  # Replace with your file path
    print(f"Set filename to {filename}")
    traj = read_custom_file(filename)

    print("Running Dynamics Analyzer")
    rdf_analyzer = RDF(trajectory=traj)
    res = rdf_analyzer.run(radius=6)
    print("Making figure 1")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plot_rdf(res, ax=ax)
    plt.savefig('rdf-plot1.png', dpi=150)
    print("Arraynames are: " + ', '.join(res.get_arraynames()))

    print("Making figure 2")
    # Creating my own plot
    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(1, 2, left=0.08, bottom=0.15, right=0.98)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    for ax in (ax0, ax1):
        ax.set_xlabel(r'r ($\AA$)')
    ax0.set_ylabel(r'g(r)')
    ax1.set_ylabel(r'integrated number density')
    for stepsize in (1, 10, 100):
        # Increasing the sampling stepsize of the RDF
        res = rdf_analyzer.run(radius=6, stepsize=stepsize, istart=1)
        # Adjust array names based on species in your system if needed
        l, = ax0.plot(res.get_array('radii_Li_Li') if 'radii_Li_Li' in res.get_arraynames() else list(res.get_arraynames())[0],
                      res.get_array('rdf_Li_Li') if 'rdf_Li_Li' in res.get_arraynames() else list(res.get_arraynames())[1],
                      label=f'stepsize-{stepsize}')
        ax1.plot(res.get_array('radii_Li_Li') if 'radii_Li_Li' in res.get_arraynames() else list(res.get_arraynames())[0],
                 res.get_array('int_Li_Li') if 'int_Li_Li' in res.get_arraynames() else list(res.get_arraynames())[2],
                 color=l.get_color(),
                 label=f'stepsize-{stepsize}')
    ax0.legend()
    ax1.legend()
    plt.savefig('rdf-plot2.png')
    return

if __name__ == '__main__':
    main()
