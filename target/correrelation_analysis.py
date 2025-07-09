"""
Correlation Analysis Module for CPDyAna
=======================================

This module provides functions for calculating Van Hove correlation functions
(self and distinct parts) for molecular dynamics trajectories. It is compatible
with Quantum ESPRESSO, LAMMPS, and BOMD (.trj) trajectory formats as parsed by
the CPDyAna input_reader module.

Functions:
    Van_Hove_self:   Compute the self part of the Van Hove function for ion diffusion.
    Van_Hove_distinct: Compute the distinct part of the Van Hove function for ion diffusion.

Author: CPDyAna Development Team
Version: 2025-07-09
"""

import numpy as np
from collections import Counter
from scipy.stats import norm

def Van_Hove_self(avg_step, dt, rmax, step_skip, sigma, ngrid, moving_ion_pos):
    """
    Calculate the self part of the Van Hove function for ion diffusion.

    Compatible with BOMD (.trj), QE, and LAMMPS trajectories as parsed by input_reader.

    Args:
        avg_step (int): Number of steps to average over.
        dt (np.ndarray): Time array (ps).
        rmax (float): Maximum distance for radial distribution (Å).
        step_skip (int): Number of steps to skip between time origins.
        sigma (float): Standard deviation for Gaussian broadening (Å).
        ngrid (int): Number of grid points for distance histogram.
        moving_ion_pos (np.ndarray): Ion positions, shape (n_ions, n_steps, 3).

    Returns:
        tuple: (dist_interval, reduced_nt, gsrt)
            dist_interval (np.ndarray): Distance grid (Å).
            reduced_nt (int): Number of reduced time steps.
            gsrt (np.ndarray): Van Hove self part, shape (reduced_nt, ngrid).
    """
    nstep = len(dt) - avg_step  # Number of time steps for averaging
    dr = rmax / (ngrid - 1)  # Distance interval
    dist_interval = np.linspace(0.0, rmax, ngrid)  # Create distance intervals
    reduced_nt = int(nstep / float(step_skip)) + 1  # Number of reduced time steps
    gsrt = np.zeros((reduced_nt, ngrid), dtype=np.double)  # Initialize Van Hove self part array

    # Avoid division by zero with a safe denominator
    ion_count = len(moving_ion_pos[:, 0, 0]) if moving_ion_pos.shape[0] > 0 else 1
    denominator = float(max(1, avg_step)) * float(ion_count)
    gaussians = norm.pdf(dist_interval[:, None], dist_interval[None, :], sigma) / denominator

    for it in range(reduced_nt):  # Loop over each reduced time step
        it0 = min(it * step_skip, nstep)  # Calculate the starting time step

        # Initialize common_matrix for each time step to avoid UnboundLocalError
        common_matrix = Counter()

        for it1 in range(avg_step):  # Loop over each averaging step
            if it0 + it1 >= moving_ion_pos.shape[1] or it1 >= moving_ion_pos.shape[1]:
                continue  # Skip if indices would be out of bounds

            # Compute displacement for all ions at this time origin
            dists = np.sqrt(
                np.square(moving_ion_pos[:, it0 + it1, 0] - moving_ion_pos[:, it1, 0]) +
                np.square(moving_ion_pos[:, it0 + it1, 1] - moving_ion_pos[:, it1, 1]) +
                np.square(moving_ion_pos[:, it0 + it1, 2] - moving_ion_pos[:, it1, 2])
            ).tolist()

            # Find distance indices and filter valid ones
            r_indices = [int(dist / dr) for dist in filter(lambda e: e < rmax, dists)]

            # Update counter with new indices
            current_counts = Counter(r_indices)
            common_matrix.update(current_counts)

        # Only process if we have data
        if common_matrix:
            for a, b in common_matrix.most_common(ngrid):
                if 0 <= a < ngrid:  # Ensure index is within bounds
                    gsrt[it, :] += gaussians[a, :] * b  # Update Van Hove self part

    return dist_interval, reduced_nt, gsrt  # Return results

def Van_Hove_distinct(avg_step, dt, rmax, step_skip, sigma, ngrid, moving_ion_pos, volume,
                      x1, x2, x3, y1, y2, y3, z1, z2, z3):
    """
    Calculate the distinct part of the Van Hove function for ion diffusion.

    Compatible with BOMD (.trj), QE, and LAMMPS trajectories as parsed by input_reader.

    Args:
        avg_step (int): Number of steps to average over.
        dt (np.ndarray): Time array (ps).
        rmax (float): Maximum distance for radial distribution (Å).
        step_skip (int): Number of steps to skip between time origins.
        sigma (float): Standard deviation for Gaussian broadening (Å).
        ngrid (int): Number of grid points for distance histogram.
        moving_ion_pos (np.ndarray): Ion positions, shape (n_ions, n_steps, 3).
        volume (float): Simulation cell volume (Å³).
        x1, x2, x3, y1, y2, y3, z1, z2, z3 (float): Cell vectors (Å).

    Returns:
        tuple: (dist_interval, reduced_nt, gdrt)
            dist_interval (np.ndarray): Distance grid (Å).
            reduced_nt (int): Number of reduced time steps.
            gdrt (np.ndarray): Van Hove distinct part, shape (reduced_nt, ngrid).
    """
    nstep = len(dt) - avg_step  # Number of time steps for averaging
    dr = rmax / (ngrid - 1)  # Distance interval
    dist_interval = np.linspace(0.0, rmax, ngrid)  # Create distance intervals
    reduced_nt = int(nstep / float(step_skip)) + 1  # Number of reduced time steps
    gdrt = np.zeros((reduced_nt, ngrid), dtype=np.double)  # Initialize Van Hove distinct part array

    # Precompute Gaussian functions for histogram smoothing
    gaussians = norm.pdf(dist_interval[:, None], dist_interval[None, :], sigma) / (
        float(avg_step) * float(len(moving_ion_pos[:, 0, 0]))
    )

    # Generate image vectors for periodic boundary conditions
    images = np.array([[(i, j, k) for i in range(-1, 2)] for j in range(-1, 2) for k in range(-1, 2)]).reshape((-1, 3))
    zd = np.sum(images ** 2, axis=1)  # Squared distances for image vectors
    indx0 = np.argmin(zd)  # Index of the origin image

    # Auxiliary factor for normalization (shell volume)
    aux_factor = 4.0 * np.pi * dist_interval ** 2
    aux_factor[0] = np.pi * dr ** 2  # Special case for zero distance

    # Number density of ions
    rho = float(len(moving_ion_pos[:, 0, 0]) / volume)

    # Build lists of all ion pairs and image vectors (excluding self at origin)
    U, V, J = [], [], []
    n_ions = len(moving_ion_pos[:, 0, 0])
    for u in range(n_ions):
        for v in range(n_ions):
            for j in range(3 ** 3):
                if u != v or j != indx0:
                    U.append(u)
                    V.append(v)
                    J.append(j)

    for it in range(reduced_nt):  # Loop over each reduced time step
        it0 = min(it * step_skip, nstep)  # Calculate the starting time step
        for it1 in range(avg_step):  # Loop over each averaging step
            # Compute distances considering periodic boundary conditions
            dists = [
                np.sqrt(
                    np.square(moving_ion_pos[U[Y], it0 + it1, 0] + (x1 + x2 + x3) * images[J[Y]][0] - moving_ion_pos[V[Y], it1, 0]) +
                    np.square(moving_ion_pos[U[Y], it0 + it1, 1] + (y1 + y2 + y3) * images[J[Y]][1] - moving_ion_pos[V[Y], it1, 1]) +
                    np.square(moving_ion_pos[U[Y], it0 + it1, 2] + (z1 + z2 + z3) * images[J[Y]][2] - moving_ion_pos[V[Y], it1, 2])
                )
                for Y in range(len(U))
            ]
            r_indices = [int(dist / dr) for dist in filter(lambda e: e < rmax, dists)]
            if it1 == 0:
                common_matrix = Counter(r_indices)
            else:
                common_matrix_old = common_matrix
                common_matrix = Counter(r_indices)
                common_matrix_old.update(common_matrix)
                common_matrix = common_matrix_old
        for a, b in common_matrix.most_common(ngrid):
            gdrt[it, :] += (gaussians[a, :] * b) / (aux_factor[a] * rho)
    return dist_interval, reduced_nt, gdrt  # Return