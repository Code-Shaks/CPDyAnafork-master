#!/usr/bin/env python3
"""
CPDyAna – Combined Molecular Dynamics Analysis Tool
====================================================

CPDyAna is a comprehensive command-line interface for analyzing molecular dynamics
simulations. It provides unified access to multiple analysis types including:

- **Mean Square Displacement (MSD)**: Calculate diffusion coefficients and tracer diffusivity
- **Non-Gaussian Parameters (NGP)**: Analyze non-Gaussian behavior in diffusion
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
    - BOMD (.trj) trajectory files
    - ASE-compatible formats

Usage Examples:
    # Mean Square Displacement analysis
    python CPDyAna.py msd -T 800 --data-dir /path/to/data --diffusing-elements Li Na
    
    # Non-Gaussian Parameters analysis
    python CPDyAna.py ngp -T 800 --data-dir /path/to/data --diffusing-elements Li Na --initial-time 0 --final-time 200

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
"""

import argparse
import json
import glob
import os
import sys
import subprocess
from turtle import pos

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm

from ase.io import read as read_positions_with_ase
from ase.io import write
from ase import Atoms

# Internal module imports for analysis functionality
from target.trajectory import Trajectory
from target.io import read_xsf, read_positions_with_ase
from target import correrelation_analysis as corr
from target import io
from target import input_reader as inp
from target import calculations as cal
from target import json_serializable as js
from target import plotting as p
from target import data_processing as dp
from target import data_processing_lammps as dpl
from target import probability_density as prob
from target import compute_rdf as rdf
from target import compute_vaf

def Job(temperature, diffusing_elements, diffusivity_direction_choices,
        diffusivity_choices, correlation, data_dir, Conv_factor,
        initial_time, final_time, initial_slope_time, final_slope_time,
        block, rmax, step_skip, sigma, ngrid, mode=None,
        lammps_elements=None, lammps_timestep=None, element_mapping=None,
        export_verification=False, show_recommendations=True, lammps_units="metal"):
# def Job(temperature, diffusing_elements, diffusivity_direction_choices,
#         diffusivity_choices, correlation, data_dir, Conv_factor,
#         initial_time, final_time, initial_slope_time, final_slope_time,
#         block, rmax, step_skip, sigma, ngrid, mode=None,
#         lammps_elements=None, lammps_timestep=None, element_mapping=None,
#         export_verification=False, show_recommendations=True, lammps_units="metal",
#         use_lazy_loading=True):
    """
    Main analysis job function for MSD and related analyses, using UNIFIED calculation approach.

    Handles trajectory reading, format detection, and dispatches to the appropriate
    analysis routines for MSD, NGP, Van Hove, etc. Compatible with QE, LAMMPS, and BOMD (.trj).

    Args:
        temperature (list): List of temperatures for analysis.
        diffusing_elements (list): Elements to analyze.
        diffusivity_direction_choices (list): Directions for MSD/diffusion.
        diffusivity_choices (list): Types of diffusivity (Tracer, Collective, etc.).
        correlation (list): Correlation function types.
        data_dir (str): Directory containing trajectory files.
        Conv_factor (float): Unit conversion factor.
        initial_time, final_time, initial_slope_time, final_slope_time (float): Time windows.
        block (int): Block size for statistical analysis.
        rmax (float): Max radius for correlation functions.
        step_skip (int): Step skip for correlation functions.
        sigma (float): Gaussian broadening for correlation functions.
        ngrid (int): Number of grid points for correlation functions.
        mode (str): Analysis mode (msd, ngp, vh, etc.).
        lammps_elements, lammps_timestep, element_mapping: LAMMPS/BOMD-specific options.
        export_verification (bool): Export verification trajectory.
        show_recommendations (bool): Show analysis recommendations.
        lammps_units (str): LAMMPS unit system.

    Returns:
        tuple: (temp_input_dict, temp_output_dict) with all processed data.
    """
    # Initialize dictionaries to store input and output data
    temp_input_dict, temp_output_dict = {}, {}
    
    # Detect trajectory format
    format_info = inp.detect_trajectory_format(data_dir)
    if format_info['format'] is None:
        raise ValueError("No recognized trajectory files found in data directory")
    
    original_conv_factor = Conv_factor
    if format_info['format'] == 'lammps' and Conv_factor == 0.529177249:  # Default QE value
        # Set appropriate conversion factors based on LAMMPS units
        if lammps_units in ['metal', 'real']:
            Conv_factor = 1.0
        elif lammps_units == 'si':
            Conv_factor = 1e10
        elif lammps_units == 'lj':
            Conv_factor = 1.0
        if Conv_factor != original_conv_factor:
            print(f"LAMMPS {lammps_units} units detected: Automatically adjusted conversion factor from {original_conv_factor} to {Conv_factor}")
    
    # Process each temperature condition
    for temp_count, temp in enumerate(temperature):
        # Initialize variables to avoid UnboundLocalError
        pos_full, n_frames, dt_full, t_full, cell_param_full = None, 0, None, None, None
        thermo_data, volumes, inp_array = {}, [], []
        ke_elec_full, cell_temp_full, ion_temp_full = None, None, None
        tot_energy_full, enthalpy_full, tot_energy_ke_ion_full = None, None, None
        tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full = None, None, None
        pos, steps, dt, t, cell_param = None, None, None, None, None
        ke_elec, cell_temp, ion_temp, tot_energy = None, None, None, None
        enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec = None, None, None
        vol, pressure = None, None
        dt_value, n_timesteps, First_term, Last_term = 1.0, 0, 0, 0
        
        if format_info['format'] == 'lammps':
            lammps_file = format_info['lammps_files'][temp_count]
            (pos_full, n_frames, dt_full, t_full, cell_param_full,
             thermo_data, volumes, inp_array) = inp.read_lammps_trajectory(
                lammps_file,
                elements=lammps_elements,
                timestep=lammps_timestep,
                Conv_factor=Conv_factor,
                element_mapping=element_mapping,
                export_verification=export_verification,
                show_recommendations=show_recommendations
            )
            
            # Define n_atoms from pos_full shape
            n_atoms = pos_full.shape[0]
            
            # Enhanced element validation with mapping consideration
            unique_elements = set(inp_array)
            print(f"Elements found in trajectory: {unique_elements}")
            
            # Validate that requested diffusing elements are available
            missing_elements = [ele for ele in diffusing_elements if ele not in unique_elements]
            if missing_elements:
                if element_mapping:
                    mapped_elements = set(element_mapping.values())
                    if len(unique_elements) == 1 and 'H' in unique_elements and len(mapped_elements) > 0:
                        print("Warning: Default element 'H' detected, overriding with mapped elements from --element-mapping.")
                        # Distribute atoms among mapped elements based on element mapping
                        inp_array = []
                        for i in range(n_atoms):
                            # Simple distribution - in practice, this should use actual atom types
                            element_idx = i % len(list(mapped_elements))
                            inp_array.append(list(mapped_elements)[element_idx])
                        unique_elements = mapped_elements
                    else:
                        raise ValueError(f"Error: Diffusing elements {missing_elements} not found in the system. "
                                       f"Available elements are {list(unique_elements)}. "
                                       f"Element mapping provided: {element_mapping}. "
                                       f"Please check --diffusing-elements and --element-mapping arguments.")
                else:
                    raise ValueError(f"Error: Diffusing elements {missing_elements} not found in the system. "
                                   f"Available elements are {list(unique_elements)}. "
                                   f"Please check --diffusing-elements and --lammps-elements arguments.")
            
            # Enhanced thermodynamic data creation
            ke_elec_full = np.zeros(n_frames)
            cell_temp_full = np.full(n_frames, temp)
            ion_temp_full = np.full(n_frames, temp)
            tot_energy_full = thermo_data.get('potential_energy', np.zeros(n_frames))
            enthalpy_full = np.zeros(n_frames)
            tot_energy_ke_ion_full = np.zeros(n_frames)
            tot_energy_ke_ion_ke_elec_full = np.zeros(n_frames)
            vol_full = np.array(volumes)
            pressure_full = np.zeros(n_frames)
            
            print(f"\n=== UNIFIED TRAJECTORY ANALYSIS PREPARATION ===")
            print(f"Position array shape: {pos_full.shape} (atoms, frames, xyz)")
            print(f"Cell parameter array shape: {cell_param_full.shape} (frames, params)")
            print(f"Time range: {t_full[0]:.3f} - {t_full[-1]:.3f} ps")
            print(f"Analysis window: {initial_time} - {final_time} ps")
            print(f"Slope calculation: {initial_slope_time} - {final_slope_time} ps")
            
            # Use LAMMPS-specific processing for data segmentation
            dt_value = dt_full[0] if len(dt_full) > 0 else lammps_timestep or 1.0
            n_timesteps = len(t_full)
            First_term, Last_term = dpl.find_terms_lammps(dt_value, n_timesteps, initial_time, final_time)
            (pos, steps, dt, t, cell_param, ke_elec, cell_temp, ion_temp, tot_energy,
             enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure) = dpl.segmenter_func_lammps(
                First_term, Last_term, pos_full, dt_value, n_timesteps, cell_full=cell_param_full)
    # temp_input_dict, temp_output_dict = {}, {}

    # format_info = inp.detect_trajectory_format(data_dir)
    # if format_info['format'] is None:
    #     raise ValueError("No recognized trajectory files found in data directory")

    # original_conv_factor = Conv_factor
    # if format_info['format'] == 'lammps' and Conv_factor == 0.529177249:
    #     if lammps_units in ['metal', 'real']:
    #         Conv_factor = 1.0
    #     elif lammps_units == 'si':
    #         Conv_factor = 1e10
    #     elif lammps_units == 'lj':
    #         Conv_factor = 1.0
    #     if Conv_factor != original_conv_factor:
    #         print(f"LAMMPS {lammps_units} units detected: Automatically adjusted conversion factor from {original_conv_factor} to {Conv_factor}")

    # for temp_count, temp in enumerate(temperature):
    #     # --- LAZY LOADING PATH ---
    #     if use_lazy_loading and format_info['format'] == 'lammps':
    #         lammps_file = format_info['lammps_files'][temp_count]
    #         frame_gen = inp.iter_lammps_trajectory(
    #             lammps_file,
    #             elements=lammps_elements,
    #             timestep=lammps_timestep,
    #             Conv_factor=Conv_factor,
    #             element_mapping=element_mapping
    #         )
    #         first_frame = next(frame_gen)
    #         n_atoms = len(first_frame["positions"])
    #         inp_array = first_frame["symbols"]
    #         # Re-create generator for full pass
    #         frame_gen = inp.iter_lammps_trajectory(
    #             lammps_file,
    #             elements=lammps_elements,
    #             timestep=lammps_timestep,
    #             Conv_factor=Conv_factor,
    #             element_mapping=element_mapping
    #         )
    #         (pos_array, rectified_structure_array, conduct_ions_array, frame_ions_array,
    #          frame_pos_array, conduct_pos_array, conduct_rectified_structure_array,
    #          frame_rectified_structure_array) = dpl.data_evaluator_lammps(
    #             diffusivity_direction_choices, diffusing_elements, frame_gen, inp_array
    #         )
    #         msd = cal.msd_tracer(
    #             inp.iter_lammps_trajectory(lammps_file, elements=lammps_elements, timestep=lammps_timestep, Conv_factor=Conv_factor, element_mapping=element_mapping),
    #             n_atoms
    #         )
    #         msd_charged = cal.msd_charged(
    #             inp.iter_lammps_trajectory(lammps_file, elements=lammps_elements, timestep=lammps_timestep, Conv_factor=Conv_factor, element_mapping=element_mapping),
    #             n_atoms
    #         )
    #         ngp = cal.calc_ngp_tracer(
    #             inp.iter_lammps_trajectory(lammps_file, elements=lammps_elements, timestep=lammps_timestep, Conv_factor=Conv_factor, element_mapping=element_mapping),
    #             n_atoms
    #         )
    #         # Store results
    #         temp_input_dict[(temp, diffusing_elements[0])] = {'evaluated_data': {}, 'evaluated_corr': {}}
    #         temp_output_dict[(temp, diffusing_elements[0])] = {
    #             'msd_data': msd,
    #             'msd_charged': msd_charged,
    #             'ngp_data': ngp
    #         }
    #         continue

    #     if use_lazy_loading and format_info['format'] == 'bomd':
    #         bomd_file = format_info['bomd_files'][temp_count]
    #         frame_gen = inp.iter_bomd_trajectory(bomd_file, elements=lammps_elements)
    #         first_frame = next(frame_gen)
    #         n_atoms = len(first_frame["positions"])
    #         inp_array = first_frame["symbols"]
    #         frame_gen = inp.iter_bomd_trajectory(bomd_file, elements=lammps_elements)
    #         (pos_array, rectified_structure_array, conduct_ions_array, frame_ions_array,
    #          frame_pos_array, conduct_pos_array, conduct_rectified_structure_array,
    #          frame_rectified_structure_array) = dp.data_evaluator(
    #             diffusivity_direction_choices, diffusing_elements, frame_gen, inp_array
    #         )
    #         msd = cal.msd_tracer(
    #             inp.iter_bomd_trajectory(bomd_file, elements=lammps_elements),
    #             n_atoms
    #         )
    #         msd_charged = cal.msd_charged(
    #             inp.iter_bomd_trajectory(bomd_file, elements=lammps_elements),
    #             n_atoms
    #         )
    #         ngp = cal.calc_ngp_tracer(
    #             inp.iter_bomd_trajectory(bomd_file, elements=lammps_elements),
    #             n_atoms
    #         )
    #         temp_input_dict[(temp, diffusing_elements[0])] = {'evaluated_data': {}, 'evaluated_corr': {}}
    #         temp_output_dict[(temp, diffusing_elements[0])] = {
    #             'msd_data': msd,
    #             'msd_charged': msd_charged,
    #             'ngp_data': ngp
    #         }
    #         continue
                
        elif format_info['format'] == 'quantum_espresso':
            # Original QE processing (unchanged)
            inp_array = inp.read_ion_file(format_info['ion_files'][temp_count])
            cell_param_full = inp.read_cel_file(format_info['cel_files'][temp_count], Conv_factor)
            (ke_elec_full, cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full,
             tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full,
             n_frames, time_diff) = inp.read_evp_file(format_info['evp_files'][temp_count], Conv_factor)
            pos_full, steps_full, dt_full, t_full = inp.read_pos_file(
                format_info['pos_files'][temp_count], inp_array,
                Conv_factor, n_frames, time_diff)
            
            # Use QE-specific processing
            First_term, Last_term = dp.find_terms(t_full.tolist(), initial_time, final_time)
            (pos, steps, dt, t, cell_param, ke_elec, cell_temp, ion_temp, tot_energy,
             enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure) = dp.segmenter_func(
                First_term, Last_term, pos_full, dt_full, t_full, cell_param_full, ke_elec_full,
                cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full,
                tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full)
                
            # Set dt_value for unified calculation
            dt_value = dt_full[0] if len(dt_full) > 0 else 1.0
        
        # elif format_info['format'] == 'bomd':
        #     bomd_file = format_info['bomd_files'][temp_count]
        #     (pos_full, n_frames, dt_full, t_full, cell_param_full,
        #     thermo_data, volumes, inp_array) = inp.read_bomd_trajectory(
        #         bomd_file,
        #         elements=bomd_elements,
        #         timestep=bomd_timestep,
        #         export_verification=export_verification
        #     )
            # # Provide dummy arrays for missing thermodynamic data
            # ke_elec_full = np.zeros(n_frames)
            # cell_temp_full = np.full(n_frames, temp)
            # ion_temp_full = np.full(n_frames, temp)
            # tot_energy_full = thermo_data.get('total_energy_ry', np.zeros(n_frames))
            # enthalpy_full = np.zeros(n_frames)
            # tot_energy_ke_ion_full = np.zeros(n_frames)
            # tot_energy_ke_ion_ke_elec_full = np.zeros(n_frames)
            # vol_full = np.array(volumes)
            # pressure_full = np.zeros(n_frames)
            # # Use QE-like segmentation for BOMD
            # First_term, Last_term = dp.find_terms(t_full.tolist(), initial_time, final_time)
            # (pos, steps, dt, t, cell_param, ke_elec, cell_temp, ion_temp, tot_energy,
            # enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure) = dp.segmenter_func(
            #     First_term, Last_term, pos_full, dt_full, t_full, cell_param_full,
            #     ke_elec_full=ke_elec_full, cell_temp_full=cell_temp_full, ion_temp_full=ion_temp_full,
            #     tot_energy_full=tot_energy_full, enthalpy_full=enthalpy_full, tot_energy_ke_ion_full=tot_energy_ke_ion_full,
            #     tot_energy_ke_ion_ke_elec_full=tot_energy_ke_ion_ke_elec_full, vol_full=vol_full, pressure_full=pressure_full
            # )
            
        elif format_info['format'] == 'bomd':
            bomd_file = format_info['bomd_files'][temp_count]
            # --- NEW: Use read_bomd_files instead of inp.read_bomd_trajectory ---
            # Try to find the .in file in the same directory
            in_files = [f for f in os.listdir(data_dir) if f.endswith('.in')]
            if not in_files:
                raise ValueError("No .in file found in BOMD data directory")
            in_file = os.path.join(data_dir, in_files[0])
            # Call your new function
            (pos_full, n_frames, dt_full, t_full, cell_param_full,
             thermo_data, volumes, inp_array) = inp.read_bomd_files(
                in_file, bomd_file, Conv_factor=Conv_factor
            )
            # Provide dummy arrays for missing thermodynamic data if needed
            ke_elec_full = np.zeros(n_frames)
            cell_temp_full = np.full(n_frames, temp)
            ion_temp_full = np.full(n_frames, temp)
            tot_energy_full = thermo_data.get('total_energy_ry', np.zeros(n_frames))
            enthalpy_full = np.zeros(n_frames)
            tot_energy_ke_ion_full = np.zeros(n_frames)
            tot_energy_ke_ion_ke_elec_full = np.zeros(n_frames)
            vol_full = np.array(volumes)
            pressure_full = np.zeros(n_frames)
            # Use QE-like segmentation for BOMD
            First_term, Last_term = dp.find_terms(t_full.tolist(), initial_time, final_time)
            (pos, steps, dt, t, cell_param, ke_elec, cell_temp, ion_temp, tot_energy,
            enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure) = dp.segmenter_func(
                First_term, Last_term, pos_full, dt_full, t_full, cell_param_full,
                ke_elec_full=ke_elec_full, cell_temp_full=cell_temp_full, ion_temp_full=ion_temp_full,
                tot_energy_full=tot_energy_full, enthalpy_full=enthalpy_full, tot_energy_ke_ion_full=tot_energy_ke_ion_full,
                tot_energy_ke_ion_ke_elec_full=tot_energy_ke_ion_ke_elec_full, vol_full=vol_full, pressure_full=pressure_full
            )
        
        elif format_info['format'] == 'xsf':
            xsf_file = format_info['xsf_files'][temp_count]
            atoms, positions, cell, data = read_xsf(xsf_file)
            pos_full = np.transpose(positions, (1, 0, 2))
            n_frames = positions.shape[0]
            cell_param_full = np.tile(cell.flatten(), (n_frames, 1))
            # Generate mock data for missing information
            dt_full = np.ones(n_frames - 1) * (lammps_timestep or 0.001)
            t_full = np.arange(n_frames) * (lammps_timestep or 0.001)
            ke_elec_full = np.zeros(n_frames)
            cell_temp_full = np.full(n_frames, temp)
            ion_temp_full = np.full(n_frames, temp)
            tot_energy_full = np.zeros(n_frames)
            enthalpy_full = np.zeros(n_frames)
            tot_energy_ke_ion_full = np.zeros(n_frames)
            tot_energy_ke_ion_ke_elec_full = np.zeros(n_frames)
            vol_full = np.ones(n_frames)
            pressure_full = np.zeros(n_frames)
            inp_array = lammps_elements or ['H'] * pos_full.shape[0]

        elif format_info['format'] in ('vasp', 'cp2k', 'gromacs'):
            traj_file = (format_info['vasp_files'] + format_info['cp2k_files'] + format_info['gromacs_files'])[temp_count]
            positions = read_positions_with_ase(traj_file)
            pos_full = np.transpose(positions, (1, 0, 2))
            n_frames = positions.shape[0]
            cell_param_full = np.tile(np.eye(3).flatten(), (n_frames, 1))
            dt_full = np.ones(n_frames - 1) * (lammps_timestep or 0.001)
            t_full = np.arange(n_frames) * (lammps_timestep or 0.001)
            ke_elec_full = np.zeros(n_frames)
            cell_temp_full = np.full(n_frames, temp)
            ion_temp_full = np.full(n_frames, temp)
            tot_energy_full = np.zeros(n_frames)
            enthalpy_full = np.zeros(n_frames)
            tot_energy_ke_ion_full = np.zeros(n_frames)
            tot_energy_ke_ion_ke_elec_full = np.zeros(n_frames)
            vol_full = np.ones(n_frames)
            pressure_full = np.zeros(n_frames)
            inp_array = lammps_elements or ['H'] * pos_full.shape[0]
            
        else:
            raise ValueError(f"Unsupported trajectory format: {format_info['format']}")
        
        # Process each diffusing element
        for ele in diffusing_elements:
            # Extract element-specific trajectory data using appropriate method
            if format_info['format'] == 'lammps':
                (pos_array, rectified_structure_array, conduct_ions_array, frame_ions_array,
                 frame_pos_array, conduct_pos_array, conduct_rectified_structure_array,
                 frame_rectified_structure_array) = dpl.data_evaluator_lammps(diffusivity_direction_choices,
                                                                            [ele], pos, inp_array, steps)
            else:
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
            
            # Calculate Mean Square Displacement using UNIFIED method
            if mode == "msd":
                msd_data_dict = cal.calculate_msd([ele], diffusivity_direction_choices, diffusivity_choices,
                                                  pos_full, conduct_rectified_structure_array,
                                                  conduct_ions_array, t, Last_term, initial_slope_time,
                                                  final_slope_time, block, 
                                                  is_lammps=(format_info['format'] == 'lammps'),
                                                  is_bomd=(format_info['format'] == 'bomd'),
                                                  dt_value=dt_value, lammps_units=lammps_units,
                                                  atom_types=inp_array, cell_param_full=cell_param_full)
            
            # Initialize other analysis dictionaries
            ngp_data_dict = {}
            evaluated_corr_dict = {}
            
            if mode == "ngp":
                if format_info['format'] == 'lammps':
                    num_frames = conduct_rectified_structure_array.shape[2]
                    timestep = dt_value  # dt_value is set above from dt_full[0] or lammps_timestep
                    dt_ngp = np.arange(num_frames) * timestep
                elif format_info['format'] == 'bomd':
                    num_frames = conduct_rectified_structure_array.shape[2]
                    # Use the actual time array (t_full) if available and matches frame count
                    if t_full is not None and len(t_full) == num_frames:
                        dt_ngp = t_full
                        print(f"BOMD: Using actual time array from trajectory. Time range: {dt_ngp[0]:.4f} to {dt_ngp[-1]:.4f} ps")
                    else:
                        # Parse dt from bomd.in
                        timestep_ps = None
                        in_files = [f for f in os.listdir(data_dir) if f.endswith('.in')]
                        if in_files:
                            with open(os.path.join(data_dir, in_files[0]), 'r') as f:
                                for line in f:
                                    if 'dt' in line and '=' in line:
                                        try:
                                            dt_au = float(line.split('=')[1].split(',')[0].strip())
                                            timestep_ps = dt_au * 0.0241888435 * 1e-3  # a.u. to ps
                                            print(f"BOMD: Found dt={dt_au} a.u. in .in file, timestep={timestep_ps:.8f} ps")
                                            break
                                        except Exception:
                                            continue
                        # Try to infer stride from the .mdtrj file time array if available
                        if t_full is not None and len(t_full) > 1:
                            stride_ps = t_full[1] - t_full[0]
                            if timestep_ps is not None and abs(stride_ps - timestep_ps) > 1e-6:
                                # Likely output stride > 1
                                n_stride = round(stride_ps / timestep_ps)
                                timestep_ps = stride_ps  # Use actual output interval
                                print(f"BOMD: Detected output stride of {n_stride}, using timestep={timestep_ps:.8f} ps")
                        if timestep_ps is None:
                            timestep_ps = 0.001  # fallback to 1 fs
                        dt_ngp = np.arange(num_frames) * timestep_ps
                        print(f"BOMD: Estimated timestep {timestep_ps:.8f} ps, time range: {dt_ngp[0]:.4f} to {dt_ngp[-1]:.4f} ps")
                else:
                    dt_ngp = dt  # For QE, dt is already correct
                    
                ngp_data_dict = cal.calculate_ngp([ele], diffusivity_direction_choices,
                                                pos_full, conduct_rectified_structure_array,
                                                conduct_ions_array, dt_ngp, initial_time, final_time)
            # if mode == "ngp":
            #     # --- FIX: Construct dt as a time array for LAMMPS ---
            #     if format_info['format'] == 'lammps':
            #         num_frames = conduct_rectified_structure_array.shape[2]
            #         timestep = dt_value  # dt_value is set above from dt_full[0] or lammps_timestep
            #         dt_ngp = np.arange(num_frames) * timestep
            #     else:
            #         dt_ngp = dt  # For QE, dt is already correct
            #     ngp_data_dict = cal.calculate_ngp([ele], diffusivity_direction_choices,
            #                                       pos_full, conduct_rectified_structure_array,
            #                                       conduct_ions_array, dt_ngp, initial_time, final_time)
            
            # Van Hove correlation functions (unchanged)
            if mode == "vh":
                if format_info['format'] == 'lammps':
                    # unwrapped & drift-corrected coordinates returned by
                    # data_processing_lammps.data_evaluator_lammps
                    vh_pos = conduct_rectified_structure_array[0]
                else:
                    # original path for QE
                    vh_pos = conduct_pos_array[0]
                for correlation_type in ['Self', 'Distinct']:
                    if correlation_type == 'Self':
                        dist_interval, reduced_nt, grt = corr.Van_Hove_self(
                            avg_step=min(100, len(dt)//4),
                            dt=dt,
                            rmax=rmax,
                            step_skip=step_skip,
                            sigma=sigma,
                            ngrid=ngrid,
                            moving_ion_pos=vh_pos
                        )
                        evaluated_corr_dict[correlation_type] = {
                            'grt': grt.tolist(),
                            'dist_interval': dist_interval.tolist(),
                            'reduced_nt': reduced_nt
                        }
                    elif correlation_type == 'Distinct':
                        avg_cell = np.mean(cell_param, axis=0)
                        volume = np.linalg.det(avg_cell.reshape(3, 3))
                        x1, x2, x3 = avg_cell[0], avg_cell[1], avg_cell[2]
                        y1, y2, y3 = avg_cell[3], avg_cell[4], avg_cell[5]
                        z1, z2, z3 = avg_cell[6], avg_cell[7], avg_cell[8]
                        try:
                            dist_interval, reduced_nt, grt = corr.Van_Hove_distinct(
                                avg_step=min(50, len(dt)//8),
                                dt=dt,
                                rmax=rmax,
                                step_skip=step_skip,
                                sigma=sigma,
                                ngrid=ngrid,
                                moving_ion_pos=vh_pos,
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
                            evaluated_corr_dict[correlation_type] = {
                                'grt': [],
                                'dist_interval': [],
                                'reduced_nt': 0
                            }

            # Store processed data in dictionaries with (temperature, element) keys
            temp_input_dict[(temp, ele)] = {'evaluated_data': ele_dict, 'evaluated_corr': evaluated_corr_dict}
            temp_output_dict[(temp, ele)] = {
                # 'dt_dict': dt,
                'dt_dict': t.tolist(),
                'msd_data': msd_data_dict,
                'ngp_data': ngp_data_dict,
                'evaluated_corr': evaluated_corr_dict
            }
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
        ngp: Non-Gaussian Parameter analysis for dynamic heterogeneity
        ionic-density: 3D ionic density mapping
        rdf: Radial Distribution Function calculation
        vaf: Velocity Autocorrelation Function analysis
        vdos: Vibrational Density of States calculation.

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
    ngp = sub.add_parser("ngp", help="Non-Gaussian Parameter analysis for dynamic heterogeneity")
    vh = sub.add_parser("vh", help="Van Hove correlation function analysis")
    ionic_density = sub.add_parser("ionic-density", help="3D ionic density mapping")
    rdf = sub.add_parser("rdf", help="Radial Distribution Function calculation")
    vaf = sub.add_parser("vaf", help="Velocity Autocorrelation Function analysis")
    vdos = sub.add_parser("vdos", help="Vibrational Density of States calculation")

    # Add common arguments for MSD and Van Hove analyses
    for sp in (msd, vh, ngp):
        sp.add_argument(
        "--export-verification",
        action="store_true",
        help="Export verification XYZ trajectory (first 100 frames)"
        )
        sp.add_argument(
            "--show-recommendations",
            action="store_true", 
            help="Display analysis parameter recommendations based on trajectory"
        )
        sp.add_argument(
            "--enhanced-processing",
            action="store_true",
            default=True,
            help="Use enhanced LAMMPS processing with test_parser improvements (default: True)"
        )
        sp.add_argument(
            "--lammps-elements", 
            nargs="+", 
            help="Element symbols for LAMMPS atom types (e.g., Li La Ti O)"
        )
        
        sp.add_argument(
            "--lammps-timestep", 
            type=float, 
            help="LAMMPS timestep in picoseconds (required for LAMMPS trajectories)"
        )
        
        sp.add_argument(
            "--lammps-units", 
            default="metal", 
            choices=["metal", "real", "si", "lj"],
            help="LAMMPS unit system (default: metal)"
        )
        
        sp.add_argument(
            "--format", 
            choices=["auto", "lammps", "quantum_espresso", "ase"],
            default="auto",
            help="Force specific trajectory format (default: auto-detect)"
        )
        
        sp.add_argument(
            "--element-mapping", 
            nargs="+", 
            help="LAMMPS type to element mapping (e.g., 1:Li 2:La 3:Ti 4:O)"
        )
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
            type=int, default=10,
            help="Number of trajectory steps to skip between analysis frames (default: 10)"
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
    # Non-Gaussian Parameter analysis arguments
    ngp.add_argument(
        "--plot-data",
        action="append", nargs="+",
        default=None,
        help="Specific data to plot as tuples: T element direction. If omitted, plot all."
    )
    ngp.add_argument(
        "--first-time",
        type=float, default=2.0,
        help="Start time for NGP plot display in picoseconds (default: 2.0)"
    )
    ngp.add_argument(
        "--last-time",
        type=float, default=200.0,
        help="End time for NGP plot display in picoseconds (default: 200.0)"
    )
    ngp.add_argument(
        "--save-path",
        default="NGP.jpg",
        help="File path to save NGP plot (default: 'NGP.jpg')"
    )

    # Ionic density analysis arguments
    ionic_density.add_argument(
        "--bomd-elements", nargs="+",
        help="Element symbols for BOMD atom order (e.g., Li O Ti)"
    )
    ionic_density.add_argument(
        "--bomd-timestep", type=float,
        help="BOMD timestep in picoseconds"
    )
    ionic_density.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel) - REQUIRED"
    )
    ionic_density.add_argument(
    "--lammps-elements", nargs="+",
    help="Element symbols for LAMMPS atom types (e.g., Li La Ti O)"
    )
    ionic_density.add_argument(
        "--element-mapping", nargs="+",
        help="LAMMPS type to element mapping (e.g., 1:Li 2:La 3:Ti 4:O)"
    )
    ionic_density.add_argument(
        "--lammps-timestep", type=float,
        help="LAMMPS timestep in picoseconds"
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
        "--bomd-elements", nargs="+",
        help="Element symbols for BOMD atom order (e.g., Li O Ti)"
    )
    rdf.add_argument(
        "--bomd-timestep", type=float,
        help="BOMD timestep in picoseconds"
    )
    rdf.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel) - REQUIRED"
    )
    rdf.add_argument(
    "--lammps-elements", nargs="+",
    help="Element symbols for LAMMPS atom types (e.g., Li La Ti O)"
    )
    rdf.add_argument(
        "--element-mapping", nargs="+",
        help="LAMMPS type to element mapping (e.g., 1:Li 2:La 3:Ti 4:O)"
    )
    rdf.add_argument(
        "--lammps-timestep", type=float,
        help="LAMMPS timestep in picoseconds"
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
        "--central-atom", nargs="+", default=None,
        help="Central atom types for RDF calculation"
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
        "--bomd-elements", nargs="+",
        help="Element symbols for BOMD atom order (e.g., Li O Ti)"
    )
    vaf.add_argument(
        "--bomd-timestep", type=float,
        help="BOMD timestep in picoseconds"
    )
    vaf.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel, optionally .evp) - REQUIRED"
    )
    vaf.add_argument(
        "--lammps-elements", nargs="+",
        help="Element symbols for LAMMPS atom types (e.g., Li S Al P O)"
    )
    vaf.add_argument(
        "--element-mapping", nargs="+",
        help="LAMMPS type to element mapping (e.g., 1:Li 2:S 3:Al)"
    )
    vaf.add_argument(
        "--lammps-timestep", type=float,
        help="LAMMPS timestep in picoseconds"
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
        "--bomd-elements", nargs="+",
        help="Element symbols for BOMD atom order (e.g., Li O Ti)"
    )
    vdos.add_argument(
        "--bomd-timestep", type=float,
        help="BOMD timestep in picoseconds"
    )
    vdos.add_argument(
        "--data-dir", required=True,
        help="Directory containing trajectory files (.pos, .in, .cel, optionally .evp) - REQUIRED"
    )
    vdos.add_argument(
        "--lammps-elements", nargs="+",
        help="Element symbols for LAMMPS atom types (e.g., Li S Al P O)"
    )
    vdos.add_argument(
        "--element-mapping", nargs="+",
        help="LAMMPS type to element mapping (e.g., 1:Li 2:S 3:Al)"
    )
    vdos.add_argument(
        "--lammps-timestep", type=float,
        help="LAMMPS timestep in picoseconds"
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
    Main entry point for CPDyAna CLI.

    Parses arguments, detects trajectory format, and dispatches to the appropriate
    analysis workflow (MSD, NGP, Van Hove, RDF, VAF, VDOS, ionic density).
    Handles BOMD (.trj), QE, and LAMMPS files, and manages plotting and output.

    Returns:
        None
    """
    a = parser()
    format_info = inp.detect_trajectory_format(a.data_dir)

    if a.mode in ("msd", "vh", "ngp"):
        if format_info['format'] is None:
            sys.exit("ERROR: No recognized trajectory files found in data directory")
        print(f"Detected trajectory format: {format_info['format']}")

        # Process element mapping for LAMMPS
        element_map = {}
        if hasattr(a, 'element_mapping') and a.element_mapping:
            for mapping in a.element_mapping:
                try:
                    type_id, element = mapping.split(':')
                    element_map[int(type_id)] = element
                    print(f"Mapped LAMMPS type {type_id} to element {element}")
                except ValueError:
                    print(f"Warning: Invalid element mapping '{mapping}', skipping")

        # Unified Job() call for all formats
        if format_info['format'] == 'bomd':
            bomd_elements = getattr(a, 'bomd_elements', None)
            bomd_timestep = getattr(a, 'bomd_timestep', None)
            Temp_inp_data, Temp_out_data = Job(
                a.temperature, a.diffusing_elements, a.diffusivity_direction_choices,
                a.diffusivity_choices, a.correlation, a.data_dir,
                a.Conv_factor, a.initial_time, a.final_time, a.initial_slope_time,
                a.final_slope_time, a.block, a.rmax, a.step_skip, a.sigma, a.ngrid,
                a.mode,
                lammps_elements=None,
                lammps_timestep=None,
                element_mapping=None,
                export_verification=getattr(a, 'export_verification', False),
                show_recommendations=getattr(a, 'show_recommendations', False),
                lammps_units=getattr(a, 'lammps_units', 'metal')
            )
        else:
            Temp_inp_data, Temp_out_data = Job(
                a.temperature, a.diffusing_elements, a.diffusivity_direction_choices,
                a.diffusivity_choices, a.correlation, a.data_dir,
                a.Conv_factor, a.initial_time, a.final_time, a.initial_slope_time,
                a.final_slope_time, a.block, a.rmax, a.step_skip, a.sigma, a.ngrid,
                a.mode, 
                lammps_elements=a.lammps_elements, 
                lammps_timestep=a.lammps_timestep,
                element_mapping=element_map,
                export_verification=getattr(a, 'export_verification', False),
                show_recommendations=getattr(a, 'show_recommendations', False),
                lammps_units=getattr(a, 'lammps_units', 'metal')
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
        # if a.mode == "msd":
        #     if a.plot_data is None:
        #         Plot_data_tracer = []
        #         for (temp, ele) in [(t, e) for t in a.temperature for e in a.diffusing_elements]:
        #             for direction in a.diffusivity_direction_choices:
        #                 for diff_type in a.diffusivity_choices:
        #                     Plot_data_tracer.append([temp, ele, diff_type, direction])
        #         pdata = Plot_data_tracer
        #     else:
        #         pdata = [[float(x[0]), x[1], x[2], x[3]] for x in a.plot_data]
        #     print(f"Generating MSD plot with {len(pdata)} data series...")
        #     format_info = inp.detect_trajectory_format(a.data_dir)
        #     is_lammps = format_info['format'] == 'lammps'
        #     p.msd_plot(data_source, pdata, a.first_time, a.last_time, save_path=a.save_path, is_lammps=is_lammps)
        #     print(f"MSD plot saved to: {a.save_path}")
        if a.mode == "msd":
            if a.plot_data is None:
                Plot_data_tracer = []
                for (temp, ele) in [(t, e) for t in a.temperature for e in a.diffusing_elements]:
                    for direction in a.diffusivity_direction_choices:
                        for diff_type in a.diffusivity_choices:
                            Plot_data_tracer.append([temp, ele, diff_type, direction])
                pdata = Plot_data_tracer
            else:
                pdata = [[float(x[0]), x[1], x[2], x[3]] for x in a.plot_data]
            print(f"Generating MSD plot with {len(pdata)} data series...")
            format_info = inp.detect_trajectory_format(a.data_dir)
            is_lammps = format_info['format'] == 'lammps'
            is_bomd = format_info['format'] == 'bomd'
            is_qe = format_info['format'] == 'quantum_espresso'
            p.msd_plot(data_source, pdata, a.first_time, a.last_time, save_path=a.save_path, 
                    is_lammps=is_lammps, is_bomd=is_bomd, is_qe=is_qe)
            print(f"MSD plot saved to: {a.save_path}")
        elif a.mode == "ngp":
            if a.plot_data is None:
                Plot_data_ngp = [[temp, ele, direction] for temp in a.temperature
                                 for ele in a.diffusing_elements for direction in a.diffusivity_direction_choices]
                pdata = Plot_data_ngp
            else:
                pdata = [[float(x[0]), x[1], x[2]] for x in a.plot_data]
            print(f"Generating NGP plot with {len(pdata)} data series...")
            p.ngp_plot(data_source, pdata, a.first_time, a.last_time, save_path=a.save_path)
            print(f"NGP plot saved to: {a.save_path}")
        elif a.mode == "vh":
            if a.plot_data is None:
                pdata = []
                for (T, ele), blob in Temp_out_data.items():
                    for corr_type in blob["evaluated_corr"]:
                        pdata.append([T, ele, corr_type])
            else:
                pdata = [[float(x[0]), x[1], x[2]] for x in a.plot_data]
            print(f"Generating Van Hove plot with {len(pdata)} correlation functions...")
            # p.van_hove_plot(
            #     data_source,
            #     pdata,
            #     save_path=a.save_path,
            #     figsize=tuple(a.figsize)
            # )
            dt_vh = None
            for (T, ele), blob in Temp_out_data.items():
                try:
                    dt_arr = np.array(blob['dt_dict'])
                    if len(dt_arr) > 1:
                        dt_vh = dt_arr[1] - dt_arr[0]
                        break
                except Exception:
                    continue
            if dt_vh is None:
                dt_vh = 0.01  # fallback
            p.van_hove_plot(
                data_source,
                pdata,
                save_path=a.save_path,
                figsize=tuple(a.figsize),
                first_time=a.initial_time,
                last_time=a.final_time,
                step_skip=a.step_skip,
                dt=dt_vh
            )
            # p.van_hove_plot(
            #     data_source,
            #     pdata,
            #     save_path=a.save_path,
            #     figsize=tuple(a.figsize),
            #     first_time=a.initial_time,
            #     last_time=a.final_time
            # )
            print(f"Van Hove plot saved to: {a.save_path}")

    # Handle ionic density mapping
    elif a.mode == "ionic-density":
        format_info = inp.detect_trajectory_format(a.data_dir)
        if format_info['format'] == 'lammps':
            fmt = inp.detect_trajectory_format(a.data_dir)
            if fmt['format'] != 'lammps':
                sys.exit("ERROR: Only LAMMPS trajectories supported for ionic-density")
            lf = fmt.get('lammps_files', [])
            if not lf:
                lf = glob.glob(os.path.join(a.data_dir, "*.lammpstrj"))
            if not lf:
                sys.exit("ERROR: No LAMMPS dump file found")
            lfile = os.path.abspath(lf[0])
            base = os.path.splitext(os.path.basename(lfile))[0]
            out = f"{base}_density.xsf"
            cmd = [
                sys.executable, "-m", "target.probability_density",
                "--lammps-file", lfile,
                "--output", out,
                "--element", a.element,
                "--sigma", str(a.sigma),
                "--n-sigma", str(a.n_sigma),
                "--density", str(a.density),
                "--step-skip", str(a.step_skip),
                "--num-frames", str(a.num_frames),
            ]
            if a.lammps_elements:
                cmd += ["--lammps-elements"] + a.lammps_elements
            if a.element_mapping:
                cmd += ["--element-mapping"] + a.element_mapping
            if a.lammps_timestep is not None:
                cmd += ["--lammps-timestep", str(a.lammps_timestep)]
            if a.mask:
                cmd += ["--mask", a.mask]
            if a.recenter:
                cmd += ["--recenter"]
            if a.bbox:
                cmd += ["--bbox", a.bbox]
            print(f"Processing LAMMPS file for ionic density: {base}")
            try:
                subprocess.run(cmd, check=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
                print(f"→ Density file created: {out}")
            except subprocess.CalledProcessError as e:
                print(f"→ Processing failed: {e}")
        elif format_info['format'] == 'bomd':
            # --- BOMD support for ionic density ---
            bomd_files = format_info.get('bomd_files', [])
            if not bomd_files:
                sys.exit("ERROR: No BOMD .mdtrj file found for ionic density analysis")
            bomd_trj = os.path.abspath(bomd_files[0])
            in_files = [f for f in os.listdir(a.data_dir) if f.endswith('.in')]
            if not in_files:
                sys.exit("ERROR: No .in file found for BOMD ionic density analysis")
            # Use BOMD elements if provided, else None
            bomd_elements = getattr(a, "bomd_elements", None)
            output_file = a.output if hasattr(a, "output") else "density.xsf"
            cmd = [
                sys.executable, "-m", "target.probability_density",
                "--bomd-trj", bomd_trj,
                "--output", output_file,
                "--element", a.element,
                "--sigma", str(a.sigma),
                "--n-sigma", str(a.n_sigma),
                "--density", str(a.density),
                "--step-skip", str(getattr(a, "step_skip", 1)),
                "--num-frames", str(getattr(a, "num_frames", 0))
            ]
            if bomd_elements:
                cmd += ["--bomd-elements"] + bomd_elements
            if getattr(a, "mask", None):
                cmd += ["--mask", a.mask]
            if getattr(a, "recenter", False):
                cmd += ["--recenter"]
            if getattr(a, "bbox", None):
                cmd += ["--bbox", a.bbox]
            print(f"Processing BOMD file for ionic density: {os.path.basename(bomd_trj)}")
            try:
                subprocess.run(cmd, check=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
                print(f"Density file created: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"Processing failed: {e}")
        else:
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
            for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
                pos_file = os.path.abspath(pos_file)
                ion_file = os.path.abspath(ion_file)
                cel_file = os.path.abspath(cel_file)
                missing_files = []
                for f in [pos_file, ion_file, cel_file]:
                    if not os.path.exists(f):
                        missing_files.append(f)
                if missing_files:
                    print(f"Skipping file set {i+1}: Missing files {missing_files}")
                    continue
                base_name = os.path.splitext(os.path.basename(pos_file))[0]
                output_file = f"{base_name}_density.xsf"
                try:
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
                    print(f" → Density file created: {output_file}")
                except subprocess.CalledProcessError as e:
                    print(f" → Failed to process {base_name}: {e}")
                    continue
                except Exception as e:
                    print(f" → Unexpected error for {base_name}: {e}")
                    continue
            print(f"Ionic density analysis completed. Generated {len(results)} density files.")

    elif a.mode == "rdf":
        format_info = inp.detect_trajectory_format(a.data_dir)
        if format_info['format'] == 'lammps':
            lammps_files = format_info.get('lammps_files', [])
            if not lammps_files:
                lammps_files = glob.glob(os.path.join(a.data_dir, "*.lammpstrj"))
            if not lammps_files:
                sys.exit("ERROR: No LAMMPS trajectory file found in data directory for RDF analysis")
            lammps_file = os.path.abspath(lammps_files[0])
            output_prefix = "rdf_plot_lammps"
            cmd = [
                sys.executable, "-m", "target.compute_rdf",
                "--lammps-file", lammps_file,
                "--output-prefix", output_prefix,
                "--central-atoms"
            ] + a.central_atom
            if a.pair_atoms:
                cmd += ["--pair-atoms"] + a.pair_atoms
            cmd += [
                "--ngrid", str(a.ngrid),
                "--rmax", str(a.rmax),
                "--sigma", str(a.sigma),
                "--xlim", str(a.xlim[0]), str(a.xlim[1]),
                "--num-frames", str(a.num_frames)
            ]
            if a.lammps_elements:
                cmd += ["--lammps-elements"] + a.lammps_elements
            if a.element_mapping:
                cmd += ["--element-mapping"] + a.element_mapping
            if a.lammps_timestep:
                cmd += ["--lammps-timestep", str(a.lammps_timestep)]
            print(f"Processing LAMMPS file for RDF: {lammps_file}")
            try:
                subprocess.run(cmd, check=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
                print(f"→ RDF plots created with prefix: {output_prefix}")
            except subprocess.CalledProcessError as e:
                print(f"→ Failed to process LAMMPS RDF: {e}")
        elif format_info['format'] == 'bomd':
            # --- BOMD support for RDF ---
            bomd_files = format_info.get('bomd_files', [])
            if not bomd_files:
                sys.exit("ERROR: No BOMD .mdtrj file found for RDF analysis")
            bomd_trj = os.path.abspath(bomd_files[0])
            in_files = [f for f in os.listdir(a.data_dir) if f.endswith('.in')]
            if not in_files:
                sys.exit("ERROR: No .in file found for BOMD RDF analysis")
            in_file = os.path.join(a.data_dir, in_files[0])
            output_prefix = "rdf_plot_bomd"
            cmd = [
                sys.executable, "-m", "target.compute_rdf",
                "--bomd-trj", bomd_trj,
                "--bomd-in", in_file,
                "--output-prefix", output_prefix,
                "--central-atoms"
            ] + a.central_atom
            if a.pair_atoms:
                cmd += ["--pair-atoms"] + a.pair_atoms
            cmd += [
                "--ngrid", str(a.ngrid),
                "--rmax", str(a.rmax),
                "--sigma", str(a.sigma),
                "--xlim", str(a.xlim[0]), str(a.xlim[1]),
                "--num-frames", str(a.num_frames)
            ]
            print(f"Processing BOMD file for RDF: {os.path.basename(bomd_trj)}")
            try:
                subprocess.run(cmd, check=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
                print(f"RDF plots created with prefix: {output_prefix}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process BOMD RDF: {e}")
        else:
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
            for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
                try:
                    base_name = os.path.splitext(os.path.basename(pos_file))[0]
                    output_prefix = f"rdf_plot_{base_name}"
                    print(f"Processing file set {i+1}/{len(pos_files)}: {base_name}")
                    extracted_frames = rdf.build_ase_trajectory(
                        ion_file, pos_file, cel_file,
                        time_after_start=getattr(a, "time_after_start", 60),
                        num_frames=getattr(a, "num_frames", 100),
                        time_interval=getattr(a, "time_interval", 0.00193511)
                    )
                    rdf.compute_rdf(
                        extracted_frames,
                        output_prefix=output_prefix,
                        time_after_start=getattr(a, "time_after_start", 60),
                        central_atoms=getattr(a, "central_atom", ['Li', 'Al', 'P', 'S']),
                        pair_atoms=getattr(a, "pair_atoms", None),
                        ngrid=getattr(a, "ngrid", 1001),
                        rmax=getattr(a, "rmax", 10.0),
                        sigma=getattr(a, "sigma", 0.2),
                        xlim=tuple(getattr(a, "xlim", [1.5, 8.0]))
                    )
                    results[base_name] = {"rdf_plots": f"{output_prefix}_*.png"}
                    print(f" → RDF plots created with prefix: {output_prefix}")
                except Exception as e:
                    print(f" → Failed to process {base_name}: {e}")
                    continue

    elif a.mode == "vaf":
        files = os.listdir(a.data_dir)
        files_lower = [f.lower() for f in files]
        is_qe = any(f.endswith('.pos') for f in files_lower) and any(f.endswith('.cel') for f in files_lower)
        is_lammps = any(f.endswith('.dump') or f.endswith('.lammpstrj') or f.endswith('.extxyz') for f in files_lower)
        if is_qe:
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
            for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
                base_name = os.path.splitext(os.path.basename(pos_file))[0]
                evp_file = evp_files[i] if i < len(evp_files) else None
                cmd = [
                    sys.executable, os.path.join(os.path.dirname(__file__), "compute_vaf.py"),
                    "--data-dir", a.data_dir,
                    "--element"
                ] + a.element
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
                    print(f" → VAF analysis completed with prefix: {a.out_prefix}_{base_name}")
                except subprocess.CalledProcessError as e:
                    print(f" → VAF analysis failed for {base_name}: {e}")
                    continue
                except Exception as e:
                    print(f" → Unexpected error in VAF analysis for {base_name}: {e}")
                    continue
            print(f"VAF analysis completed for {len(results)} file sets.")
        elif is_lammps:
            lammps_file = None
            for ext in ('*.dump', '*.lammpstrj', '*.extxyz'):
                found = glob.glob(os.path.join(a.data_dir, ext))
                if found:
                    lammps_file = found[0]
                    break
            if not lammps_file:
                sys.exit("ERROR: No LAMMPS trajectory file found in data-dir for VAF analysis")
            base_name = os.path.splitext(os.path.basename(lammps_file))[0]
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "compute_vaf.py"),
                "--data-dir", a.data_dir,
                "--element"
            ] + a.element
            if a.lammps_elements:
                cmd += ["--lammps-elements"] + a.lammps_elements
            if a.element_mapping:
                cmd += ["--element-mapping"] + a.element_mapping
            if a.lammps_timestep:
                cmd += ["--lammps-timestep", str(a.lammps_timestep)]
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
            print(f"Processing LAMMPS file for VAF: {base_name}")
            try:
                subprocess.run(cmd, check=True)
                print(f" → VAF analysis completed for LAMMPS file: {base_name}")
            except subprocess.CalledProcessError as e:
                print(f" → VAF analysis failed for LAMMPS: {e}")
        elif format_info['format'] == 'bomd':
            bomd_files = format_info.get('bomd_files', [])
            if not bomd_files:
                sys.exit("ERROR: No BOMD .mdtrj file found for VAF analysis")
            bomd_trj = os.path.abspath(bomd_files[0])
            in_files = [f for f in os.listdir(a.data_dir) if f.endswith('.in')]
            if not in_files:
                sys.exit("ERROR: No .in file found for BOMD VAF analysis")
            in_file = os.path.join(a.data_dir, in_files[0])
            base_name = os.path.splitext(os.path.basename(bomd_trj))[0]
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "compute_vaf.py"),
                "--bomd-trj", bomd_trj,
                "--bomd-in", in_file,
                "--element"
            ] + a.element
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
            print(f"Processing BOMD file for VAF: {base_name}")
            try:
                subprocess.run(cmd, check=True)
                print(f" → VAF analysis completed for BOMD file: {base_name}")
            except subprocess.CalledProcessError as e:
                print(f" → VAF analysis failed for BOMD: {e}")
            return
        else:
            sys.exit("ERROR: Could not detect QE or LAMMPS trajectory files in data-dir.")

    elif a.mode == "vdos":
        files = os.listdir(a.data_dir)
        files_lower = [f.lower() for f in files]
        is_qe = any(f.endswith('.pos') for f in files_lower) and any(f.endswith('.cel') for f in files_lower)
        is_lammps = any(f.endswith('.dump') or f.endswith('.lammpstrj') or f.endswith('.extxyz') for f in files_lower)
        if is_qe:
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
            for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
                base_name = os.path.splitext(os.path.basename(pos_file))[0]
                evp_file = evp_files[i] if i < len(evp_files) else None
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
                if evp_file:
                    cmd += ["--evp_file", evp_file]
                print(f"Processing file set {i+1}/{len(pos_files)}: {base_name}")
                try:
                    subprocess.run(cmd, check=True)
                    print(f" → VDOS analysis completed with prefix: {a.out_prefix}_{base_name}")
                except subprocess.CalledProcessError as e:
                    print(f" → VDOS analysis failed for {base_name}: {e}")
                    continue
            print("VDOS analysis completed.")
        elif is_lammps:
            lammps_file = None
            for ext in ('*.dump', '*.lammpstrj', '*.extxyz'):
                found = glob.glob(os.path.join(a.data_dir, ext))
                if found:
                    lammps_file = found[0]
                    break
            if not lammps_file:
                print("ERROR: No LAMMPS trajectory file found in data-dir for VDOS analysis")
                return
            base_name = os.path.splitext(os.path.basename(lammps_file))[0]
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "vdos.py"),
                "--data-dir", a.data_dir,
                "--out_prefix", f"{a.out_prefix}_{base_name}",
                "--start", str(a.start),
                "--nframes", str(a.nframes),
                "--stride", str(a.stride),
                "--time_interval", str(a.time_interval),
                "--elements"
            ] + a.elements
            if a.lammps_elements:
                cmd += ["--lammps-elements"] + a.lammps_elements
            if a.element_mapping:
                cmd += ["--element-mapping"] + a.element_mapping
            if a.lammps_timestep:
                cmd += ["--lammps-timestep", str(a.lammps_timestep)]
            print(f"Processing LAMMPS file for VDOS: {base_name}")
            try:
                subprocess.run(cmd, check=True)
                print(f"VDOS analysis completed for LAMMPS file: {base_name}")
            except subprocess.CalledProcessError as e:
                print(f"VDOS analysis failed for LAMMPS: {e}")
            print("VDOS analysis completed.")
        elif a.mode == "vdos":
            files = os.listdir(a.data_dir)
            files_lower = [f.lower() for f in files]
            is_qe = any(f.endswith('.pos') for f in files_lower) and any(f.endswith('.cel') for f in files_lower)
            is_lammps = any(f.endswith('.dump') or f.endswith('.lammpstrj') or f.endswith('.extxyz') for f in files_lower)
            if is_qe:
                # ...existing QE code...
                return
            elif is_lammps:
                # ...existing LAMMPS code...
                return
            elif format_info['format'] == 'bomd':
                bomd_files = format_info.get('bomd_files', [])
                if not bomd_files:
                    print("ERROR: No BOMD trajectory file found in data-dir for VDOS analysis")
                    return
                bomd_trj = os.path.abspath(bomd_files[0])
                in_files = [f for f in os.listdir(a.data_dir) if f.endswith('.in')]
                if not in_files:
                    print("ERROR: No .in file found for BOMD VDOS analysis")
                    return
                in_file = os.path.join(a.data_dir, in_files[0])
                base_name = os.path.splitext(os.path.basename(bomd_trj))[0]
                cmd = [
                    sys.executable, os.path.join(os.path.dirname(__file__), "vdos.py"),
                    "--bomd-trj", bomd_trj
                ]
                if hasattr(a, "bomd_elements") and a.bomd_elements:
                    cmd += ["--bomd-elements"] + a.bomd_elements
                cmd += [
                    "--elements"
                ] + a.elements
                cmd += [
                    "--out_prefix", f"{a.out_prefix}_{base_name}",
                    "--nframes", str(a.nframes),
                    "--stride", str(a.stride),
                    "--time_interval", str(a.time_interval)
                ]
                print(f"Processing BOMD file for VDOS: {base_name}")
                try:
                    subprocess.run(cmd, check=True)
                    print(f" → VDOS analysis completed for BOMD file: {base_name}")
                except subprocess.CalledProcessError as e:
                    print(f" → VDOS analysis failed for BOMD: {e}")
                return
        else:
            print("ERROR: Could not detect QE or LAMMPS trajectory files in data-dir.")
        return

# Entry point for script execution
if __name__ == "__main__":
    main()