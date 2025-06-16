import numpy as np
from collections import Counter
from scipy.stats import norm

def Van_Hove_self(avg_step, dt, rmax, step_skip, sigma, ngrid, moving_ion_pos):
    """
    Calculate the self part of the Van Hove function for ion diffusion.

    Parameters:
    - avg_step (int): Number of steps to average over.
    - dt (numpy.ndarray): Time array.
    - rmax (float): Maximum distance for radial distribution.
    - step_skip (int): Number of steps to skip.
    - sigma (float): Standard deviation for Gaussian function.
    - ngrid (int): Number of grid points.
    - moving_ion_pos (numpy.ndarray): Position array for moving ions.

    Returns:
    - numpy.ndarray, int, numpy.ndarray: Distance interval, reduced number of time steps, self part of Van Hove function.
    """
    nstep = len(dt) - avg_step  # Number of time steps for averaging
    dr = rmax / (ngrid - 1)  # Distance interval
    dist_interval = np.linspace(0.0, rmax, ngrid)  # Create distance intervals
    reduced_nt = int(nstep/float(step_skip)) + 1  # Number of reduced time steps
    gsrt = np.zeros((reduced_nt, ngrid), dtype=np.double)  # Initialize Van Hove self part array
    gaussians = norm.pdf(dist_interval[:, None], dist_interval[None, :], sigma) / (float(avg_step) * float(len(moving_ion_pos[:,0,0])))  # Precompute Gaussian functions
    for it in range(reduced_nt):  # Loop over each reduced time step
        it0 = min(it * step_skip, nstep)  # Calculate the starting time step
        for it1 in range(avg_step):  # Loop over each averaging step
            dists = np.sqrt(np.square(moving_ion_pos[:,it0+it1,0] - moving_ion_pos[:,it1,0]) + np.square(moving_ion_pos[:,it0+it1,1] - moving_ion_pos[:,it1,1]) + np.square(moving_ion_pos[:,it0+it1,2] - moving_ion_pos[:,it1,2])).tolist()  # Calculate distances
            r_indices = [int(dist / dr) for dist in filter(lambda e: e < rmax, dists)]  # Find distance indices
            if it1 == 0:
                common_matrix = Counter(r_indices)  # Initialize counter for distance indices
            else:
                common_matrix_old = common_matrix
                common_matrix = Counter(r_indices)  # Update counter for distance indices
                common_matrix_old.update(common_matrix)
                common_matrix = common_matrix_old
        for a, b in common_matrix.most_common(ngrid):  # Loop over most common distance indices
            gsrt[it, :] += gaussians[a, :] * b  # Update Van Hove self part
    return dist_interval, reduced_nt, gsrt  # Return results

def Van_Hove_distinct(avg_step, dt, rmax, step_skip, sigma, ngrid, moving_ion_pos, volume, x1, x2, x3, y1, y2, y3, z1, z2, z3):
    """
    Calculate the distinct part of the Van Hove function for ion diffusion.

    Parameters:
    - avg_step (int): Number of steps to average over.
    - dt (numpy.ndarray): Time array.
    - rmax (float): Maximum distance for radial distribution.
    - step_skip (int): Number of steps to skip.
    - sigma (float): Standard deviation for Gaussian function.
    - ngrid (int): Number of grid points.
    - moving_ion_pos (numpy.ndarray): Position array for moving ions.
    - volume (float): System volume.
    - x1, x2, x3, y1, y2, y3, z1, z2, z3 (float): Box vectors.

    Returns:
    - numpy.ndarray, int, numpy.ndarray: Distance interval, reduced number of time steps, distinct part of Van Hove function.
    """
    nstep = len(dt) - avg_step  # Number of time steps for averaging
    dr = rmax / (ngrid - 1)  # Distance interval
    dist_interval = np.linspace(0.0, rmax, ngrid)  # Create distance intervals
    reduced_nt = int(nstep/float(step_skip)) + 1  # Number of reduced time steps
    gdrt = np.zeros((reduced_nt, ngrid), dtype=np.double)  # Initialize Van Hove distinct part array
    gaussians = norm.pdf(dist_interval[:, None], dist_interval[None, :], sigma) / (float(avg_step) * float(len(moving_ion_pos[:,0,0])))  # Precompute Gaussian functions
    images = np.array([[(i, j, k) for i in range(-1, 2)] for j in range(-1, 2) for k in range(-1, 2)]).reshape((-1, 3))  # Generate image vectors for periodic boundary conditions
    zd = np.sum(images**2, axis=1)  # Calculate squared distances for image vectors
    indx0 = np.argmin(zd)  # Find the index of the smallest distance
    aux_factor = 4.0 * np.pi * dist_interval**2  # Auxiliary factor for normalization
    aux_factor[0] = np.pi * dr**2  # Special case for zero distance
    rho = float(len(moving_ion_pos[:,0,0]) / volume)  # Number density of ions
    U = []  # Initialize lists for ion pairs and image vectors
    V = []
    J = []
    for u in range(len(moving_ion_pos[:,0,0])):  # Loop over all ions
        for v in range(len(moving_ion_pos[:,0,0])):  # Loop over all ions
            for j in range(3 ** 3):  # Loop over all image vectors
                if u != v or j != indx0:
                    U.append(u)  # Add ion pair and image vector if not self-pair at origin
                    V.append(v)
                    J.append(j)
    for it in range(reduced_nt):  # Loop over each reduced time step
        it0 = min(it * step_skip, nstep)  # Calculate the starting time step
        for it1 in range(avg_step):  # Loop over each averaging step
            print(it, it0, it1)
            dists = [np.sqrt(np.square(moving_ion_pos[U[Y],it0+it1,0] + (x1 + x2 + x3) * images[J[Y]][0] - moving_ion_pos[V[Y],it1,0]) + np.square(moving_ion_pos[U[Y],it0+it1,1] + (y1 + y2 + y3) * images[J[Y]][1] - moving_ion_pos[V[Y],it1,1]) + np.square(moving_ion_pos[U[Y],it0+it1,2] + (z1 + z2 + z3) * images[J[Y]][2] - moving_ion_pos[V[Y],it1,2])) for Y in range(len(U))]  # Calculate distances considering periodic boundary conditions
            r_indices = [int(dist / dr) for dist in filter(lambda e: e < rmax, dists)]  # Find distance indices
            if it1 == 0:
                common_matrix = Counter(r_indices)  # Initialize counter for distance indices
            else:
                common_matrix_old = common_matrix
                common_matrix = Counter(r_indices)  # Update counter for distance indices
                common_matrix_old.update(common_matrix)
                common_matrix = common_matrix_old
        for a, b in common_matrix.most_common(ngrid):  # Loop over most common distance indices
            gdrt[it, :] += (gaussians[a, :] * b) / (aux_factor[a] * rho)  # Update Van Hove distinct part
    return dist_interval, reduced_nt, gdrt  # Return results

