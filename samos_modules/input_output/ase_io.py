from ase.io import read
import numpy as np

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