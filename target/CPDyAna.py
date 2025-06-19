#!/usr/bin/env python3
"""
CPDyAna – Combined Molecular Dynamics Analysis Tool
====================================================

CPDyAna is a comprehensive command-line interface for analyzing molecular dynamics
simulations. It provides unified access to multiple analysis types including:

- **Mean Square Displacement (MSD)**: Calculate diffusion coefficients and tracer diffusivity
- **Van Hove Correlation Functions**: Analyze spatial correlations (self and distinct)
- **Radial Distribution Functions (RDF)**: Compute pair correlation functions
- **Velocity Autocorrelation Functions (VAF)**: Analyze velocity correlations
- **Vibrational Density of States (VDOS)**: Calculate phonon spectra
- **Ionic Density Maps**: Generate 3D density distributions

This single-file variant merges the original CPDyAna.py driver with helper
functions for a complete standalone workflow.

Supported File Formats:
    - Quantum ESPRESSO trajectory files (.pos, .cel, .evp, .in)
    - LAMMPS trajectory files
    - ASE-compatible formats

Usage Examples:
    # Mean Square Displacement analysis
    python CPDyAna.py msd -T 800 --data-dir /path/to/data --diffusing-elements Li Na
    
    # Van Hove correlation function
    python CPDyAna.py vh -T 800 --data-dir /path/to/data --rmax 10 --sigma 0.1
    
    # Radial distribution function
    python CPDyAna.py rdf --data-dir /path/to/data --central-atom Li Al
    
    # Velocity autocorrelation function
    python CPDyAna.py vaf --data-dir /path/to/data --element Li --nframes 1000
    
    # Ionic density mapping
    python CPDyAna.py ionic-density --data-dir /path/to/data --element Li --sigma 0.3

Author: CPDyAna Development Team
Version: Combined version (01-02-2024)
License: Open Source
"""

import argparse
import json
import glob
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm

# Internal module imports for analysis functionality
from . import correrelation_analysis as corr
from . import input_reader as inp
from . import calculations as cal
from . import json_serializable as js
from . import plotting as p
from . import data_processing as dp
from . import probability_density as prob
from . import compute_rdf as rdf
from . import compute_vaf

def Job(temperature, diffusing_elements, diffusivity_direction_choices, diffusivity_choices, correlation, 
        pos_file, cel_file, evp_file, ion_file, Conv_factor, initial_time, final_time, 
        initial_slope_time, final_slope_time, block, rmax, step_skip, sigma, ngrid, mode=None):
    """
    Main analysis job function for MSD, Van Hove, and related molecular dynamics analyses.

    This function orchestrates the complete analysis workflow:
    1. Reads input files (trajectory, cell parameters, energies, ion definitions)
    2. Processes and segments data based on time windows
    3. Performs element-specific analysis for each temperature
    4. Calculates MSD or Van Hove correlation functions based on mode
    5. Returns structured dictionaries with input and output data

    Args:
        temperature (list of float): List of temperatures in Kelvin for analysis.
        diffusing_elements (list of str): Element symbols to analyze (e.g., ['Li', 'Na']).
        diffusivity_direction_choices (list of str): Spatial directions for diffusivity 
            analysis (e.g., ['X', 'Y', 'Z', 'XY', 'XYZ']).
        diffusivity_choices (list of str): Types of diffusivity to calculate 
            (e.g., ['Tracer', 'Collective']).
        correlation (list of str): Correlation function types 
            (e.g., ['Self', 'Distinct']).
        pos_file (list of str): Paths to position trajectory files (.pos).
        cel_file (list of str): Paths to cell parameter files (.cel).
        evp_file (list of str): Paths to energy/volume/pressure files (.evp).
        ion_file (list of str): Paths to ion definition files (.in).
        Conv_factor (float): Unit conversion factor (typically 0.529177249 
            for Bohr to Angstrom conversion).
        initial_time (float): Start time for analysis window in picoseconds.
        final_time (float): End time for analysis window in picoseconds.
        initial_slope_time (float): Start time for slope calculation in MSD analysis.
        final_slope_time (float): End time for slope calculation in MSD analysis.
        block (int): Block size for statistical averaging.
        rmax (float): Maximum radial distance for correlation function analysis in Angstroms.
        step_skip (int): Number of time steps to skip between analysis frames.
        sigma (float): Gaussian broadening parameter for correlation functions.
        ngrid (int): Number of grid points for spatial discretization.
        mode (str, optional): Analysis mode ('msd' for Mean Square Displacement, 
            'vh' for Van Hove correlation functions).

    Returns:
        tuple: A tuple containing two dictionaries:
            - temp_input_dict (dict): Input data organized by (temperature, element) keys,
              containing processed trajectory data and structural information.
            - temp_output_dict (dict): Output data organized by (temperature, element) keys,
              containing analysis results (MSD data, correlation functions, time steps).

    Raises:
        FileNotFoundError: If any of the input files cannot be found.
        ValueError: If time windows or parameters are invalid.
        
    Note:
        The function handles multiple temperatures and elements simultaneously,
        with results organized in nested dictionaries for easy access and plotting.
    """
    # Initialize dictionaries to store input and output data
    temp_input_dict, temp_output_dict = {}, {}
    
    # Process each temperature condition
    for temp_count, temp in enumerate(temperature):
        # Read input files for current temperature
        inp_array = inp.read_ion_file(ion_file[temp_count])
        cell_param_full = inp.read_cel_file(cel_file[temp_count], Conv_factor)
        
        # Extract thermodynamic properties from EVP file
        (ke_elec_full, cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, 
         tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full, 
         n_frames, time_diff) = inp.read_evp_file(evp_file[temp_count], Conv_factor)
        
        # Read trajectory data
        pos_full, steps_full, dt_full, t_full = inp.read_pos_file(pos_file[temp_count], inp_array, 
                                                              Conv_factor, n_frames, time_diff)
        
        # Determine analysis time window boundaries
        First_term, Last_term = dp.find_terms(t_full.tolist(), initial_time, final_time)
        
        # Segment all data arrays to analysis time window
        (pos, steps, dt, t, cell_param, ke_elec, cell_temp, ion_temp, tot_energy, 
         enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure) = dp.segmenter_func(
            First_term, Last_term, pos_full, dt_full, t_full, cell_param_full, ke_elec_full, 
            cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, 
            tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full)
        
        # Process each diffusing element
        for ele in diffusing_elements:
            # Extract element-specific trajectory data
            (pos_array, rectified_structure_array, conduct_ions_array, frame_ions_array, 
             frame_pos_array, conduct_pos_array, conduct_rectified_structure_array, 
             frame_rectified_structure_array) = dp.data_evaluator(diffusivity_direction_choices, 
                                                               [ele], pos, inp_array, steps)
            
            # Organize element data by direction for easy access
            ele_dict = {direction: {
                'pos_array': pos_array[i, :, :, :],
                'rectified_structure_array': rectified_structure_array[i, :, :, :],
                'conduct_ions_array': conduct_ions_array[i],
                'frame_ions_array': frame_ions_array[i],
                'frame_pos_array': frame_pos_array[i, :, :, :],
                'conduct_pos_array': conduct_pos_array[i, :, :, :],
                'conduct_rectified_structure_array': conduct_rectified_structure_array[i, :, :, :],
                'frame_rectified_structure_array': frame_rectified_structure_array[i, :, :, :]
            } for i, direction in enumerate(diffusivity_direction_choices)}
            
            # Initialize MSD data dictionary
            msd_data_dict = {}
            
            # Calculate Mean Square Displacement if requested
            if mode == "msd":
                msd_data_dict = cal.calculate_msd([ele], diffusivity_direction_choices, diffusivity_choices, 
                                          pos_full, conduct_rectified_structure_array, 
                                          conduct_ions_array, dt, Last_term, initial_slope_time, 
                                          final_slope_time, block)
            
            # Initialize correlation function dictionary
            evaluated_corr_dict = {}
            
            # Calculate Van Hove correlation functions if requested
            if mode == "vh":
                for correlation_type in ['Self', 'Distinct']:
                    if correlation_type == 'Self':
                        # Calculate self-correlation (single particle dynamics)
                        dist_interval, reduced_nt, grt = corr.Van_Hove_self(
                            avg_step=min(100, len(dt)//4),  # Adaptive averaging
                            dt=dt,
                            rmax=rmax,
                            step_skip=step_skip,
                            sigma=sigma,
                            ngrid=ngrid,
                            moving_ion_pos=conduct_pos_array[0]
                        )
                        evaluated_corr_dict[correlation_type] = {
                            'grt': grt.tolist(),
                            'dist_interval': dist_interval.tolist(),
                            'reduced_nt': reduced_nt
                        }
                    elif correlation_type == 'Distinct':
                        # Calculate distinct correlation (pair dynamics)
                        # Compute average cell parameters for volume calculation
                        avg_cell = np.mean(cell_param, axis=0)
                        volume = np.linalg.det(avg_cell.reshape(3, 3))
                        
                        # Extract cell vectors for periodic boundary conditions
                        x1, x2, x3 = avg_cell[0], avg_cell[1], avg_cell[2]
                        y1, y2, y3 = avg_cell[3], avg_cell[4], avg_cell[5]
                        z1, z2, z3 = avg_cell[6], avg_cell[7], avg_cell[8]
                        
                        try:
                            dist_interval, reduced_nt, grt = corr.Van_Hove_distinct(
                                avg_step=min(50, len(dt)//8),  # More conservative averaging for distinct
                                dt=dt,
                                rmax=rmax,
                                step_skip=step_skip,
                                sigma=sigma,
                                ngrid=ngrid,
                                moving_ion_pos=conduct_pos_array[0],
                                volume=volume,
                                x1=x1, x2=x2, x3=x3,
                                y1=y1, y2=y2, y3=y3,
                                z1=z1, z2=z2, z3=z3
                            )
                            evaluated_corr_dict[correlation_type] = {
                                'grt': grt.tolist(),
                                'dist_interval': dist_interval.tolist(),
                                'reduced_nt': reduced_nt
                            }
                        except Exception:
                            # Handle cases where distinct correlation calculation fails
                            evaluated_corr_dict[correlation_type] = {
                                'grt': [],
                                'dist_interval': [],
                                'reduced_nt': 0
                            }

            # Store processed data in dictionaries with (temperature, element) keys
            temp_input_dict[(temp, ele)] = {'evaluated_data': ele_dict, 'evaluated_corr': evaluated_corr_dict}
            temp_output_dict[(temp, ele)] = {'dt_dict': dt, 'msd_data': msd_data_dict, 'evaluated_corr': evaluated_corr_dict}
            
    return temp_input_dict, temp_output_dict

def parser():
    """
    Create and configure the command-line argument parser for CPDyAna.

    This function sets up the main argument parser with subcommands for different
    analysis modes. Each subcommand has its own specific arguments while sharing
    common parameters where appropriate.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing all user-specified
            options for the requested analysis mode.

    Subcommands:
        msd: Mean Square Displacement analysis for diffusion coefficients
        vh: Van Hove correlation function analysis  
        ionic-density: 3D ionic density mapping
        rdf: Radial Distribution Function calculation
        vaf: Velocity Autocorrelation Function analysis
        vdos: Vibrational Density of States calculation

    Common Arguments:
        - temperature: Temperature(s) for thermodynamic analysis
        - diffusing-elements: Element types to analyze
        - data-dir: Directory containing input files
        - time windows: Analysis and fitting time ranges
        - statistical parameters: Block sizes, grid points, etc.

    Example:
        >>> args = parser()
        >>> # args.mode will contain the selected subcommand
        >>> # args.temperature will contain the temperature list
    """
    # Main parser setup
    p = argparse.ArgumentParser(
        description="CPDyAna CLI – Combined Molecular Dynamics Analysis Tool",
        epilog="For detailed help on subcommands, use: python CPDyAna.py <subcommand> --help"
    )
    
    # Create subcommand parsers
    sub = p.add_subparsers(dest="mode", required=True, help="Analysis mode to run")

    # Define subcommand parsers
    msd = sub.add_parser("msd", help="Mean Square Displacement analysis")
    vh = sub.add_parser("vh", help="Van Hove correlation function analysis")
    ionic_density = sub.add_parser("ionic-density", help="3D ionic density mapping")
    rdf = sub.add_parser("rdf", help="Radial Distribution Function calculation")
    vaf = sub.add_parser("vaf", help="Velocity Autocorrelation Function analysis")
    vdos = sub.add_parser("vdos", help="Vibrational Density of States calculation")

    # Add common arguments for MSD and Van Hove analyses
    for sp in (msd, vh):
        sp.add_argument(
            "-T", "--temperature",
            nargs="+", type=float,
            default=[800.0],
            help="Temperature(s) in Kelvin for analysis (default: 800)"
        )
        sp.add_argument(
            "-e", "--diffusing-elements",
            nargs="+", default=["Li"],
            help="Diffusing element symbols to analyze (default: ['Li'])"
        )
        sp.add_argument(
            "--data-dir", required=True,
            help="Directory containing trajectory files (.pos, .cel, .evp, .in) - REQUIRED"
        )
        sp.add_argument(
            "--diffusivity-direction-choices",
            nargs="+", default=["XYZ"],
            help="Spatial directions for diffusivity analysis: X, Y, Z, XY, XZ, YZ, XYZ (default: ['XYZ'])"
        )
        sp.add_argument(
            "--diffusivity-choices",
            nargs="+", default=["Tracer"],
            help="Diffusivity calculation types: Tracer, Collective (default: ['Tracer'])"
        )
        sp.add_argument(
            "--correlation",
            nargs="*", default=["Self"],
            help="Correlation function types: Self, Distinct (default: ['Self'])"
        )
        sp.add_argument(
            "--initial-time",
            type=float, default=2.0,
            help="Start time for analysis window in picoseconds (default: 2.0)"
        )
        sp.add_argument(
            "--final-time",
            type=float, default=200.0,
            help="End time for analysis window in picoseconds (default: 200.0)"
        )
        sp.add_argument(
            "--initial-slope-time",
            type=float, default=5.0,
            help="Start time for MSD slope fitting in picoseconds (default: 5.0)"
        )
        sp.add_argument(
            "--final-slope-time",
            type=float, default=100.0,
            help="End time for MSD slope fitting in picoseconds (default: 100.0)"
        )
        sp.add_argument(
            "--Conv-factor",
            type=float, default=0.529177249,
            help="Unit conversion factor, typically Bohr to Angstrom (default: 0.529177249)"
        )
        sp.add_argument(
            "--block",
            type=int, default=500,
            help="Block size for statistical averaging (default: 500)"
        )
        sp.add_argument(
            "--rmax",
            type=float, default=10,
            help="Maximum radial distance for correlation analysis in Angstroms (default: 10)"
        )
        sp.add_argument(
            "--step-skip",
            type=int, default=500,
            help="Number of trajectory steps to skip between analysis frames (default: 500)"
        )
        sp.add_argument(
            "--sigma",
            type=float, default=0.1,
            help="Gaussian broadening parameter for correlation functions (default: 0.1)"
        )
        sp.add_argument(
            "--ngrid",
            type=int, default=101,
            help="Number of radial grid points for discretization (default: 101)"
        )
        sp.add_argument(
            "--json-output",
            default="OUTPUT.json",
            help="Output file for JSON-formatted results (default: 'OUTPUT.json')"
        )

    # MSD-specific plotting arguments
    msd.add_argument(
        "--plot-data",
        action="append", nargs="+",
        default=None,
        help="Specific data to plot as tuples: T element direction type. If omitted, plot all."
    )
    msd.add_argument(
        "--first-time",
        type=float, default=2.0,
        help="Start time for MSD plot display in picoseconds (default: 2.0)"
    )
    msd.add_argument(
        "--last-time",
        type=float, default=200.0,
        help="End time for MSD plot display in picoseconds (default: 200.0)"
    )
    msd.add_argument(
        "--save-path",
        default="MSD.jpg",
        help="File path to save MSD plot (default: 'MSD.jpg')"
    )

    # Van Hove-specific plotting arguments
    vh.add_argument(
        "--plot-data",
        action="append", nargs="+",
        default=None,
        help="Specific data to plot as tuples: T element correlation_type. If omitted, plot all."
    )
    vh.add_argument(
        "--figsize",
        nargs=2, type=float,
        default=[10, 8],
        help="Figure size for Van Hove plots as [width, height] (default: [10, 8])"
    )
    vh.add_argument(
        "--save-path",
        default="van_hove_plot.png",
        help="File path to save Van Hove plot (default: 'van_hove_plot.png')"
    )

    # Ionic density analysis arguments
    ionic_density.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel) - REQUIRED"
    )
    ionic_density.add_argument(
        "--time-after-start", type=float, default=0.0,
        help="Time in picoseconds after simulation start to begin analysis (default: 0.0)"
    )
    ionic_density.add_argument(
        "--num-frames", type=int, default=0,
        help="Number of trajectory frames to analyze (default: 0 = all available frames)"
    )
    ionic_density.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Time interval between consecutive frames in picoseconds (default: 0.00193511)"
    )
    ionic_density.add_argument(
        "--element", default="Li",
        help="Element symbol for density calculation (default: Li)"
    )
    ionic_density.add_argument(
        "--sigma", type=float, default=0.3,
        help="Gaussian sigma for density smoothing in Angstroms (default: 0.3)"
    )
    ionic_density.add_argument(
        "--n-sigma", type=float, default=4.0,
        help="Number of sigma for Gaussian cutoff distance (default: 4.0)"
    )
    ionic_density.add_argument(
        "--density", type=float, default=0.2,
        help="Density scaling factor for visualization (default: 0.2)"
    )
    ionic_density.add_argument(
        "--output", default="density.xsf",
        help="Output file name for density data in XSF format (default: density.xsf)"
    )
    ionic_density.add_argument(
        "--step-skip", type=int, default=1,
        help="Skip every N frames for efficiency (default: 1 = use all frames)"
    )
    ionic_density.add_argument(
        "--mask", default=None,
        help="Atom index mask for selective analysis (e.g., '0,1,2,5-10')"
    )
    ionic_density.add_argument(
        "--recenter", action="store_true",
        help="Recenter trajectory to remove drift before density calculation"
    )
    ionic_density.add_argument(
        "--bbox", default=None,
        help="Bounding box for density grid: 'xmin,xmax,ymin,ymax,zmin,zmax' (comma-separated)"
    )

    # RDF analysis arguments  
    rdf.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel) - REQUIRED"
    )
    rdf.add_argument(
        "--time-after-start", type=float, default=60,
        help="Equilibration time in picoseconds before RDF analysis starts (default: 60)"
    )
    rdf.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Time step between frames in picoseconds (default: 0.00193511)"
    )
    rdf.add_argument(
        "--num-frames", type=int, default=100,
        help="Number of frames for RDF statistical averaging (default: 100)"
    )
    rdf.add_argument(
        "--central-atom", nargs="+", default=['Li','Al','P','S'],
        help="Central atom types for RDF calculation (default: ['Li', 'Al', 'P', 'S'])"
    )
    rdf.add_argument('--pair-atoms', nargs='+', default=None,
                        help='Pair atom types for RDF (default: same as central-atoms)')
    rdf.add_argument('--ngrid', type=int, default=1001,
                        help='Number of radial grid points for RDF histogram (default: 1001)')
    rdf.add_argument('--rmax', type=float, default=10.0,
                        help='Maximum radial distance for RDF in Angstroms (default: 10.0)')
    rdf.add_argument('--sigma', type=float, default=0.2,
                        help='Gaussian broadening for RDF smoothing (default: 0.2)')
    rdf.add_argument('--xlim', nargs=2, type=float, default=[1.5, 8.0],
                        help='X-axis limits for RDF plots in Angstroms (default: 1.5 8.0)')

    # VAF analysis arguments
    vaf.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel, optionally .evp) - REQUIRED"
    )
    vaf.add_argument(
        "--element", nargs="+", required=True,
        help="Element symbol(s) for VAF calculation (e.g., Li Na) - REQUIRED"
    )
    vaf.add_argument(
        "--start", type=float, default=0.0,
        help="Start time for VAF analysis in picoseconds (default: 0.0)"
    )
    vaf.add_argument(
        "--nframes", type=int, default=0,
        help="Number of frames for VAF (default: 0 = all available frames)"
    )
    vaf.add_argument(
        "--stride", type=int, default=1,
        help="Frame stride: 1=all frames, 2=every other frame, etc. (default: 1)"
    )
    vaf.add_argument(
        "--blocks", type=int, default=4,
        help="Number of blocks for error estimation in VAF (default: 4)"
    )
    vaf.add_argument(
        "--out-prefix", default="vaf",
        help="Prefix for VAF output file names (default: vaf)"
    )
    vaf.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Default time step in picoseconds if no .evp file present (default: 0.00193511)"
    )
    vaf.add_argument(
        "--t-start-fit-ps", type=int, default=0,
        help="Start frame index for VAF calculation (default: 0)"
    )
    vaf.add_argument(
        "--stepsize-t", type=int, default=1,
        help="Time step size for VAF time axis (default: 1)"
    )
    vaf.add_argument(
        "--stepsize-tau", type=int, default=1,
        help="Time step size for VAF correlation time (default: 1)"
    )
    vaf.add_argument(
        "--t-end-fit-ps", type=float, default=10,
        help="End time for VAF fitting in picoseconds (default: 10)"
    )
     
    # VDOS analysis arguments
    vdos.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel, optionally .evp) - REQUIRED"
    )
    vdos.add_argument(
        "--elements", nargs="+", default=["Li", "Al", "P", "S"],
        help="Element symbols for VDOS calculation (default: ['Li', 'Al', 'P', 'S'])"
    )
    vdos.add_argument(
        "--out-prefix", default="vdos",
        help="Prefix for VDOS output file names (default: vdos)"
    )
    vdos.add_argument(
        "--start", type=float, default=0.0,
        help="Start time for VDOS analysis in picoseconds (default: 0.0)"
    )
    vdos.add_argument(
        "--nframes", type=int, default=0,
        help="Number of frames for VDOS (default: 0 = all available frames)"
    )
    vdos.add_argument(
        "--stride", type=int, default=1,
        help="Frame stride for VDOS: 1=all frames, 2=every other, etc. (default: 1)"
    )
    vdos.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Default time step in picoseconds if no .evp file present (default: 0.00193511)"
    )
    
    return p.parse_args()

def main():
    """
    Main entry point for the CPDyAna command-line interface.

    This function serves as the central dispatcher that:
    1. Parses command-line arguments to determine analysis mode
    2. Validates input files and directory structure  
    3. Dispatches to appropriate analysis functions based on mode
    4. Handles file I/O operations and subprocess calls
    5. Manages error handling and user feedback

    The function supports multiple analysis modes:
    - **msd**: Mean Square Displacement analysis with plotting
    - **vh**: Van Hove correlation function analysis with plotting  
    - **ionic-density**: 3D ionic density map generation
    - **rdf**: Radial Distribution Function calculation
    - **vaf**: Velocity Autocorrelation Function analysis
    - **vdos**: Vibrational Density of States calculation

    For MSD and Van Hove modes, the function:
    - Calls the main Job() function to perform analysis
    - Saves results to JSON format for data persistence
    - Generates publication-quality plots

    For other modes, the function:
    - Validates required input files (.pos, .in, .cel, optionally .evp)
    - Calls specialized analysis modules via subprocess
    - Handles multiple file sets in batch mode

    Raises:
        SystemExit: If required files are missing or file counts don't match
        subprocess.CalledProcessError: If subprocess analysis fails
        FileNotFoundError: If data directory or files cannot be found

    Note:
        The function automatically discovers input files based on extensions
        in the specified data directory and processes them in sorted order.
    """
    # Parse command-line arguments
    a = parser()

    # Handle MSD and Van Hove correlation analyses
    if a.mode in ("msd", "vh"):
        # Discover input files in the data directory
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))
        evp_files = sorted(glob.glob(os.path.join(a.data_dir, "*.evp")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))

        # Validate that all required file types are present
        if not (pos_files and cel_files and evp_files and ion_files):
            sys.exit("ERROR: Missing required data files in data directory. "
                    "Need: .pos (positions), .cel (cell), .evp (energy), .in (ions)")
        
        # Ensure equal number of each file type for consistent analysis
        if not (len(pos_files) == len(cel_files) == len(evp_files) == len(ion_files)):
            sys.exit("ERROR: Mismatch in number of input files. "
                    f"Found: {len(pos_files)} .pos, {len(cel_files)} .cel, "
                    f"{len(evp_files)} .evp, {len(ion_files)} .in files")

        # Execute main analysis job
        print(f"Starting {a.mode.upper()} analysis...")
        print(f"Processing {len(pos_files)} file sets for {len(a.temperature)} temperature(s)")
        print(f"Analyzing elements: {', '.join(a.diffusing_elements)}")
        
        Temp_inp_data, Temp_out_data = Job(
            a.temperature, a.diffusing_elements, a.diffusivity_direction_choices,
            a.diffusivity_choices, a.correlation, pos_files, cel_files, evp_files, ion_files,
            a.Conv_factor, a.initial_time, a.final_time, a.initial_slope_time,
            a.final_slope_time, a.block, a.rmax, a.step_skip, a.sigma, a.ngrid, a.mode
        )

        # Save results to JSON file for persistence and data sharing
        if a.json_output:
            print(f"Saving results to {a.json_output}...")
            Temp_out_data_serializable = js.convert_to_serializable(Temp_out_data)
            with open(a.json_output, 'w') as output_file:
                json.dump(Temp_out_data_serializable, output_file, indent=2)
            data_source = a.json_output
        else:
            data_source = Temp_out_data

    # Generate MSD plots
    if a.mode == "msd":
        # Determine which data to plot
        if a.plot_data is None:
            # Plot all combinations if not specified
            Plot_data_tracer = []
            for (temp, ele) in [(t, e) for t in a.temperature for e in a.diffusing_elements]:
                for direction in a.diffusivity_direction_choices:
                    for diff_type in a.diffusivity_choices:
                        Plot_data_tracer.append([temp, ele, diff_type, direction])
            pdata = Plot_data_tracer
        else:
            # Use user-specified plot data
            pdata = [[float(x[0]), x[1], x[2], x[3]] for x in a.plot_data]
        
        print(f"Generating MSD plot with {len(pdata)} data series...")
        p.msd_plot(data_source, pdata, a.first_time, a.last_time, save_path=a.save_path)
        print(f"MSD plot saved to: {a.save_path}")

    # Generate Van Hove correlation plots
    elif a.mode == "vh":
        # Determine which correlation data to plot
        if a.plot_data is None:
            pdata = []
            for (T, ele), blob in Temp_out_data.items():
                for corr_type in blob["evaluated_corr"]:
                    pdata.append([T, ele, corr_type])
        else:
            pdata = [[float(x[0]), x[1], x[2]] for x in a.plot_data]

        print(f"Generating Van Hove plot with {len(pdata)} correlation functions...")
        p.van_hove_plot(
            data_source,
            pdata,
            save_path=a.save_path,
            figsize=tuple(a.figsize)
        )
        print(f"Van Hove plot saved to: {a.save_path}")

    # Handle ionic density mapping
    elif a.mode == "ionic-density":
        # Discover and validate input files
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))

        if not pos_files or not ion_files:
            sys.exit("ERROR: Missing .pos or .in files for ionic density analysis")
        if not cel_files:
            sys.exit("ERROR: Ionic density analysis requires .cel files for unit cell information")
        if len(pos_files) != len(ion_files) or len(pos_files) != len(cel_files):
            sys.exit("ERROR: Number of .pos, .in, and .cel files must match")

        print(f"Starting ionic density analysis for element: {a.element}")
        print(f"Processing {len(pos_files)} file sets...")

        results = {}
        # Process each file set
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            # Convert to absolute paths for subprocess calls
            pos_file = os.path.abspath(pos_file)
            ion_file = os.path.abspath(ion_file)
            cel_file = os.path.abspath(cel_file)

            # Check file existence
            missing_files = []
            for f in [pos_file, ion_file, cel_file]:
                if not os.path.exists(f):
                    missing_files.append(f)
            if missing_files:
                print(f"Skipping file set {i+1}: Missing files {missing_files}")
                continue

            # Generate unique output filename
            base_name = os.path.splitext(os.path.basename(pos_file))[0]
            output_file = f"{base_name}_density.xsf"

            try:
                # Build subprocess command for density calculation
                cmd = [
                    sys.executable, "-m", "target.probability_density",
                    "--in-file", ion_file,
                    "--pos-file", pos_file,
                    "--cel-file", cel_file,
                    "--output", output_file,
                    "--time-after-start", str(a.time_after_start),
                    "--num-frames", str(a.num_frames),
                    "--time-interval", str(a.time_interval),
                    "--element", a.element,
                    "--sigma", str(a.sigma),
                    "--n-sigma", str(a.n_sigma),
                    "--density", str(a.density),
                    "--step-skip", str(getattr(a, "step_skip", 1))
                ]
                
                # Add optional arguments if provided
                if hasattr(a, "mask") and a.mask:
                    cmd += ["--mask", a.mask]
                if hasattr(a, "recenter") and a.recenter:
                    cmd += ["--recenter"]
                if hasattr(a, "bbox") and a.bbox:
                    cmd += ["--bbox", a.bbox]

                print(f"Processing file set {i+1}/{len(pos_files)}: {base_name}")
                subprocess.run(cmd, check=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
                results[base_name] = {"density_file": output_file}
                print(f"  → Density file created: {output_file}")
                
            except subprocess.CalledProcessError as e:
                print(f"  → Failed to process {base_name}: {e}")
                continue
            except Exception as e:
                print(f"  → Unexpected error for {base_name}: {e}")
                continue

        print(f"Ionic density analysis completed. Generated {len(results)} density files.")

    # Handle Radial Distribution Function analysis
    elif a.mode == "rdf":
        # Discover and validate input files
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))

        if not pos_files or not ion_files:
            sys.exit("ERROR: Missing .pos or .in files for RDF analysis")
        if len(pos_files) != len(ion_files):
            sys.exit("ERROR: Number of .pos and .in files must match")
        if not cel_files:
            sys.exit("ERROR: RDF analysis requires .cel files for unit cell information")
        if len(cel_files) != len(pos_files):
            sys.exit("ERROR: Number of .cel files must match .pos files")    

        print(f"Starting RDF analysis...")
        print(f"Central atoms: {', '.join(a.central_atom)}")
        print(f"Processing {len(pos_files)} file sets...")

        results = {}
        # Process each file set for RDF calculation
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            try:
                base_name = os.path.splitext(os.path.basename(pos_file))[0]
                output_prefix = f"rdf_plot_{base_name}"

                print(f"Processing file set {i+1}/{len(pos_files)}: {base_name}")

                # Build ASE trajectory from input files
                extracted_frames = rdf.build_ase_trajectory(
                    ion_file, pos_file, cel_file,
                    time_after_start=getattr(a, "time_after_start", 60),
                    num_frames=getattr(a, "num_frames", 100),
                    time_interval=getattr(a, "time_interval", 0.00193511),
                )

                # Compute RDF with user-specified parameters
                rdf.compute_rdf(
                    extracted_frames,
                    output_prefix=output_prefix,
                    time_after_start=getattr(a, "time_after_start", 60),
                    central_atoms=getattr(a, "central_atoms", ['Li', 'Al', 'P', 'S']),
                    pair_atoms=getattr(a, "pair_atoms", None),
                    ngrid=getattr(a, "ngrid", 1001),
                    rmax=getattr(a, "rmax", 10.0),
                    sigma=getattr(a, "sigma", 0.2),
                    xlim=tuple(getattr(a, "xlim", [1.5, 8.0]))
                )

                results[base_name] = {"rdf_plots": f"{output_prefix}_*.png"}
                print(f"  → RDF plots created with prefix: {output_prefix}")
                
            except Exception as e:
                print(f"  → Failed to process {base_name}: {e}")
                continue

        print(f"RDF analysis completed for {len(results)} file sets.")

    # Handle Velocity Autocorrelation Function analysis
    if a.mode == "vaf":
        # Discover and validate input files
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))
        evp_files = sorted(glob.glob(os.path.join(a.data_dir, "*.evp")))

        if not pos_files or not ion_files or not cel_files:
            sys.exit("ERROR: Missing required files (.pos, .in, .cel) for VAF analysis")
        if not (len(pos_files) == len(ion_files) == len(cel_files)):
            sys.exit("ERROR: Mismatch in number of .pos, .in, .cel files")

        print(f"Starting VAF analysis for elements: {', '.join(a.element)}")
        print(f"Processing {len(pos_files)} file sets...")

        results = {}
        # Process each file set for VAF calculation
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            base_name = os.path.splitext(os.path.basename(pos_file))[0]
            evp_file = evp_files[i] if i < len(evp_files) else None

            # Build subprocess command for VAF analysis
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "compute_vaf.py"),
                "--in-file", ion_file,
                "--pos-file", pos_file,
                "--cel-file", cel_file,
                "--element"
            ] + a.element

            # Add EVP file if available for accurate time steps
            if evp_file:
                cmd += ["--evp-file", evp_file]
                
            # Add analysis parameters
            cmd += [
                "--start", str(a.start),
                "--nframes", str(a.nframes),
                "--stride", str(a.stride),
                "--blocks", str(a.blocks),
                "--out-prefix", f"{a.out_prefix}_{base_name}",
                "--t-end-fit-ps", str(a.t_end_fit_ps),
                "--time-interval", str(a.time_interval),
                "--t-start-fit-ps", str(a.t_start_fit_ps),
                "--stepsize-t", str(a.stepsize_t),
                "--stepsize-tau", str(a.stepsize_tau)
            ]

            print(f"Processing file set {i+1}/{len(pos_files)}: {base_name}")
            try:
                subprocess.run(cmd, check=True)
                results[base_name] = {"vaf_output_prefix": f"{a.out_prefix}_{base_name}"}
                print(f"  → VAF analysis completed with prefix: {a.out_prefix}_{base_name}")
            except subprocess.CalledProcessError as e:
                print(f"  → VAF analysis failed for {base_name}: {e}")
                continue
            except Exception as e:
                print(f"  → Unexpected error in VAF analysis for {base_name}: {e}")
                continue

        print(f"VAF analysis completed for {len(results)} file sets.")

    # Handle Vibrational Density of States analysis
    if a.mode == "vdos":
        # Discover and validate input files
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))
        evp_files = sorted(glob.glob(os.path.join(a.data_dir, "*.evp")))

        if not pos_files or not ion_files or not cel_files:
            print("ERROR: Missing required files (.pos, .in, .cel) for VDOS analysis")
            return
        if not (len(pos_files) == len(ion_files) == len(cel_files)):
            print("ERROR: Mismatch in number of .pos, .in, .cel files")
            return

        print(f"Starting VDOS analysis for elements: {', '.join(a.elements)}")
        print(f"Processing {len(pos_files)} file sets...")

        # Process each file set for VDOS calculation
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            base_name = os.path.splitext(os.path.basename(pos_file))[0]
            evp_file = evp_files[i] if i < len(evp_files) else None
            
            # Build subprocess command for VDOS analysis
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "vdos.py"),
                "--in_file", ion_file,
                "--pos_file", pos_file,
                "--cel_file", cel_file,
                "--out_prefix", f"{a.out_prefix}_{base_name}",
                "--start", str(a.start),
                "--nframes", str(a.nframes),
                "--stride", str(a.stride),
                "--time_interval", str(a.time_interval),
                "--elements"
            ] + a.elements
            
            # Add EVP file if available
            if evp_file:
                cmd += ["--evp_file", evp_file]
                
            print(f"Processing file set {i+1}/{len(pos_files)}: {base_name}")
            try:
                subprocess.run(cmd, check=True)
                print(f"  → VDOS analysis completed with prefix: {a.out_prefix}_{base_name}")
            except subprocess.CalledProcessError as e:
                print(f"  → VDOS analysis failed for {base_name}: {e}")
                continue
                
        print("VDOS analysis completed.")
        return

# Entry point for script execution
if __name__ == "__main__":
    main()