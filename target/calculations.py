import numpy as np
from scipy.stats import linregress

# FFT-based MSD calculation
def calc_fft(x):
    """
    Calculate FFT-based autocorrelation function for MSD computation.
    
    This function uses Fast Fourier Transform to efficiently compute the 
    autocorrelation function needed for mean square displacement calculations.
    
    Args:
        x (numpy.ndarray): Input array of position data for one dimension
        
    Returns:
        numpy.ndarray: Normalized autocorrelation function values
    """
    N = x.shape[0]
    F = np.fft.fft(x, n=2 * N)  # Zero-pad to 2N for proper autocorrelation
    PSD = F * F.conjugate()  # Power spectral density
    res = np.fft.ifft(PSD)  # Inverse FFT to get autocorrelation
    res = (res[:N]).real  # Take only the first N real values
    n = N * np.ones(N) - np.arange(N)  # Normalization factor
    return res / n

# Tracer MSD calculation helper
def calc_msd_tracer(r, d_idx):
    """
    Calculate tracer mean square displacement using FFT method.
    
    This function computes MSD for a single particle trajectory using
    the efficient FFT-based algorithm to avoid O(N²) complexity.
    
    Args:
        r (numpy.ndarray): Position array of shape (time_steps, 3) for one particle
        d_idx (numpy.ndarray): Array of time lag indices
        
    Returns:
        tuple: Contains:
            - MSD values for specified time lags
            - MSD components for each spatial dimension
    """
    step, dim = r.shape
    r_sq = np.square(r)  # Square of positions
    r_sq = np.append(r_sq, np.zeros((1, 3)), axis=0)  # Zero padding
    part1_c = np.zeros((3, step))
    r_sq_sum = 2 * np.sum(r_sq, axis=0)  # Initial sum for all dimensions
    
    # Calculate first part of MSD formula (direct term)
    for i in range(step):
        r_sq_sum -= r_sq[i - 1, :] + r_sq[step - i, :]
        part1_c[:, i] = r_sq_sum / (step - i)
    
    part1 = np.sum(part1_c, axis=0)  # Sum over all dimensions
    
    # Calculate second part using FFT (cross-correlation term)
    part2_c = np.array([calc_fft(r[:, i]) for i in range(r.shape[1])])
    part2 = np.sum(part2_c, axis=0)
    
    return (part1 - 2 * part2)[d_idx], (part1_c - 2 * part2_c)[:, d_idx]


def msd_tracer(structure, step):
    """
    Calculate average tracer MSD for multiple particles.
    
    This function computes the tracer MSD by averaging over all mobile ions
    in the system for various time lags.
    
    Args:
        structure (numpy.ndarray): 3D array of shape (n_ions, n_timesteps, 3)
        step (int): Total number of time steps
        
    Returns:
        numpy.ndarray: Average MSD values for time lags from 1 to step-1
    """
    num_mobile_ions = structure.shape[0]
    msd_ions = np.empty([0, len(np.arange(1, step, 1))])  # Initialize empty array
    
    # Calculate MSD for each ion and collect results
    for i in range(num_mobile_ions):
        msd_i, _ = calc_msd_tracer(structure[i, :, :], np.arange(1, step, 1))
        msd_ions = np.append(msd_ions, msd_i.reshape(1, len(np.arange(1, step, 1))), axis=0)
    
    return np.average(msd_ions, axis=0)  # Return ensemble average

# Charged MSD calculation
def msd_charged(structure, final_msd):
    """
    Calculate the charged MSD for a given 3D position array and tracer MSD.
    
    The charged MSD accounts for collective motion and correlations between
    different ions in the system.
    
    Args:
        structure (numpy.ndarray): 3D array of shape (num_mobile_ions, steps, 3) 
                                 containing position data
        final_msd (numpy.ndarray): 1D array containing the tracer MSD values
    
    Returns:
        numpy.ndarray: 1D array of charged MSD values
    """
    num_mobile_ions = structure.shape[0]  # Number of mobile ions
    steps = structure.shape[1]           # Number of time steps
    MSD_charged = np.zeros(steps - 1)    # Initialize output array
    
    # Calculate charged MSD for each time lag
    for delt in range(1, steps):
        MSD_charged[delt-1] = (disp_sum(delt, structure) - 
                               final_msd[delt-1] * num_mobile_ions * (steps - delt)) / \
                              (0.5 * num_mobile_ions * (num_mobile_ions - 1) * (steps - delt))
    
    return MSD_charged

# Displacement sum for charged MSD
def disp_sum(delt, mobile_ion_pos):
    """
    Calculate the sum of squared displacements for a given time delta.
    
    This function computes the collective displacement term needed for
    charged MSD calculations.
    
    Args:
        delt (int): Time step difference (time lag)
        mobile_ion_pos (numpy.ndarray): 3D array of shape (num_mobile_ions, steps, 3)
    
    Returns:
        float: Sum of squared collective displacements
    """
    t = np.arange(mobile_ion_pos.shape[1] - delt)  # Valid time origins
    # Calculate displacements for all ions at time lag delt
    displacements = mobile_ion_pos[:, t + delt, :] - mobile_ion_pos[:, t, :]
    # Sum displacements over all ions, then square and sum over time and space
    return np.sum(np.square(np.sum(np.abs(displacements), axis=0)))

# Main MSD and diffusivity calculation function
def calculate_msd(diffusing_elements, diffusivity_direction_choices, diffusivity_type_choices, 
                  pos_full, conduct_rectified_structure_array, conduct_ions_array, dt, Last_term, 
                  initial_slope_time, final_slope_time, block):
    """
    Main function to calculate MSD and diffusivity for multiple elements and conditions.
    
    This function performs comprehensive MSD and diffusivity analysis including:
    - Multiple diffusion types (Tracer, Charged)
    - Multiple spatial directions (X, Y, Z, XY, YZ, ZX, XYZ)
    - Linear regression analysis for diffusivity extraction
    - Block analysis for error estimation
    
    Args:
        diffusing_elements (list): List of element names to analyze
        diffusivity_direction_choices (list): List of spatial directions
        diffusivity_type_choices (list): List of diffusion types
        pos_full (numpy.ndarray): Full position array
        conduct_rectified_structure_array (numpy.ndarray): Rectified position data
        conduct_ions_array (numpy.ndarray): Ion type information
        dt (numpy.ndarray): Time step array
        Last_term (float): Final time value
        initial_slope_time (float): Start time for linear fitting
        final_slope_time (float): End time for linear fitting
        block (int): Block size for error analysis
    
    Returns:
        dict: Nested dictionary containing MSD arrays, diffusivities, and error estimates
              Structure: {element: {property_name: [values]}}
    """
    result_dict = {}
    
    # Loop over each diffusing element
    for ele in range(len(diffusing_elements)):
        element = diffusing_elements[ele]
        result_dict[element] = {}
        
        # Loop over each spatial direction
        for direction_idx, direction in enumerate(diffusivity_direction_choices):
            # Determine dimensionality for diffusivity calculation
            d = {'XYZ': 3, 'XY': 2, 'YZ': 2, 'ZX': 2, 'X': 1, 'Y': 1, 'Z': 1}[direction]
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            
            posit = conduct_rectified_structure_array[direction_idx, :, :, :]
            step_counts = len(posit[0, :, 0])  
            
            print(f"[DEBUG] posit shape: {posit.shape}")
            print(f"[DEBUG] step_counts: {step_counts}")
            
            # Loop over each diffusion type (Tracer, Charged)
            for diff_type_idx, diff_type in enumerate(diffusivity_type_choices):
                # Define result dictionary keys
                key_msd = f"{diff_type}_msd_array{suffix}"
                key_diff = f"{diff_type}_diffusivity{suffix}"
                key_sem = f"{diff_type}_slope_sem{suffix}"
                key_diff_err = f"{diff_type}_diffusivity_error{suffix}"
                
                if diff_type == "Tracer":
                    msd_array = msd_tracer(posit, step_counts)
                elif diff_type == "Charged":
                    tracer_msd = msd_tracer(posit, step_counts)
                    msd_array = msd_charged(posit, tracer_msd)
                else:
                    raise ValueError(f"Unknown diffusivity type: {diff_type}")
                
                print(f"[DEBUG] MSD array length: {len(msd_array)}")
                print(f"[DEBUG] dt length: {len(dt)}")
                
                # Find indices for linear regression time window
                first_idx = np.searchsorted(dt, initial_slope_time, side='left')
                last_idx = np.searchsorted(dt, final_slope_time, side='right') - 1
                
                # Perform linear regression to extract diffusivity
                if first_idx <= last_idx and last_idx < len(msd_array):
                    slope, _, _, _, std_err = linregress(dt[first_idx:last_idx + 1], 
                                                         msd_array[first_idx:last_idx + 1])
                    diffusivity = (slope * 1e-4) / (2 * d)  # Convert Å²/ps to cm²/s
                else:
                    diffusivity = np.nan
                
                B = step_counts // block  # Number of complete blocks
                slope_blocks = []
                
                # Calculate diffusivity for each block
                for b in range(B):
                    t_start = b * block
                    t_end = (b + 1) * block
                    if t_end > step_counts:
                        break
                    
                    pos_block = posit[:, t_start:t_end, :]  
                    step_block = t_end - t_start
                    
                    # Calculate MSD for current block
                    if diff_type == "Tracer":
                        msd_block = msd_tracer(pos_block, step_block)
                    elif diff_type == "Charged":
                        tracer_msd_block = msd_tracer(pos_block, step_block)
                        msd_block = msd_charged(pos_block, tracer_msd_block)
                    
                    # Create time array for block and perform regression
                    dt_block = np.arange(1, step_block) * (dt[1] - dt[0])
                    if len(dt_block) >= 2:
                        slope_block, _, _, _, _ = linregress(dt_block, msd_block)
                        slope_blocks.append(slope_block)
                
                # Calculate block statistics for error estimation
                if len(slope_blocks) > 1:
                    slope_mean = np.mean(slope_blocks)
                    slope_std = np.std(slope_blocks)
                    slope_sem = slope_std / np.sqrt(len(slope_blocks) - 1)
                    diffusivity_block_mean = (slope_mean * 1e-4) / (2 * d)
                    diffusivity_sem = (slope_sem * 1e-4) / (2 * d)
                else:
                    diffusivity_block_mean = np.nan
                    diffusivity_sem = np.nan
                    slope_sem = np.nan
                
                # Store results in nested dictionary structure
                result_dict[element].setdefault(key_msd, []).append(msd_array)
                result_dict[element].setdefault(key_diff, []).append(diffusivity)
                result_dict[element].setdefault(key_sem, []).append(slope_sem if 'slope_sem' in locals() else np.nan)
                result_dict[element].setdefault(key_diff_err, []).append(diffusivity_sem)
    
    return result_dict