#!/usr/bin/env python3
"""
prcss.py - Runs the Job() function from the command line, optionally saving output to JSON.

To run from the command line, ensure you are one directory ABOVE 'target' so that
target/ is recognized as a package (contains __init__.py). Then:
    python -m target.prcss --temperature 600 --diffusing-elements Li ...
"""

import argparse
import sys
import json
import numpy as np

# Relative imports from the same 'target' folder
from . import correrelation_analysis as corr
from . import data_processing as dp
from . import input_reader as inp
from . import calculations as cal
from . import serializable  # For convert_to_serializable()

def parse_arguments():
    """
    Parse command-line arguments for prcss.py
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface to run Job() for MD analysis (MSD, Van Hove, etc.), then optionally save JSON."
    )
    # REQUIRED ARGS
    parser.add_argument("--temperature", "-T", nargs='+', type=float, required=True,
                        help="List of temperatures. Example: --temperature 300 400")
    parser.add_argument("--diffusing-elements", "-e", nargs='+', required=True,
                        help="List of diffusing elements. Example: --diffusing-elements Li+ Na+")
    parser.add_argument("--pos-file", nargs='+', required=True,
                        help="List of position (POSCAR) file paths, one per temperature.")
    parser.add_argument("--cel-file", nargs='+', required=True,
                        help="List of CEL file paths, one per temperature.")
    parser.add_argument("--evp-file", nargs='+', required=True,
                        help="List of EVP file paths, one per temperature.")
    parser.add_argument("--ion-file", nargs='+', required=True,
                        help="List of ion file paths, one per temperature.")
    
    # OPTIONAL ARGS
    parser.add_argument("--diffusivity-direction-choices", nargs='+', default=["XYZ"],
                        help="Directions for diffusivity calculations, e.g. X Y Z. Default: XYZ")
    parser.add_argument("--diffusivity-choices", nargs='+', default=["Tracer", "Charged"],
                        help="Diffusivity types to calculate. Default: Tracer Charged")
    parser.add_argument("--correlation", nargs='*', default=["Self", "Distinct"],
                        help="Correlation types to evaluate. Example: --correlation Self Distinct")
    parser.add_argument("--initial-time", nargs='+', type=float, default=[0.0],
                        help="Start times for data analysis, one per temperature. Default: 0.0")
    parser.add_argument("--final-time", nargs='+', type=float, default=[10.0],
                        help="End times for data analysis, one per temperature. Default: 10.0")
    parser.add_argument("--initial-slope-time", nargs='+', type=float, default=[2.0],
                        help="Start times for slope calculations, one per temperature. Default: 2.0")
    parser.add_argument("--final-slope-time", nargs='+', type=float, default=[8.0],
                        help="End times for slope calculations, one per temperature. Default: 8.0")
    parser.add_argument("--Conv-factor", type=float, default=1.0,
                        help="Conversion factor for units. Default: 1.0")
    parser.add_argument("--block", type=int, default=500,
                        help="Block size for MSD calculations. Default: 500")
    parser.add_argument("--rmax", type=float, default=10.0,
                        help="rmax for Van Hove. Default: 10.0")
    parser.add_argument("--step-skip", type=int, default=500,
                        help="Step skipping factor for Van Hove. Default: 500")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Smoothing parameter for Van Hove. Default: 0.1")
    parser.add_argument("--ngrid", type=int, default=101,
                        help="Number of grid points for Van Hove. Default: 101")
    parser.add_argument("--output-file", default="log.txt",
                        help="Log file name. Default: log.txt")
    parser.add_argument("--json-output", default=None,
                        help="If provided, temp_output_dict is saved to this JSON file.")

    return parser.parse_args()

def Job(temperature, diffusing_elements, diffusivity_direction_choices, diffusivity_choices, correlation,
        pos_file, cel_file, evp_file, ion_file, Conv_factor, initial_time, final_time,
        initial_slope_time, final_slope_time, block, rmax, step_skip, sigma, ngrid, output_file='log.txt'):
    """
    Comprehensive workflow for analyzing molecular dynamics simulation data
    to extract diffusion-related properties and correlations.

    1. Reads input files, including ion trajectories, cell parameters, and position info.
    2. Segments the data based on user-defined time ranges.
    3. Evaluates properties like diffusing ions, rectified structures, conductive ions.
    4. Calculates diffusivity (MSD), correlation functions (Van Hove).
    5. Returns dictionaries with evaluated data and results.

    Parameters
    ----------
    temperature : list of float
    diffusing_elements : list of str
    diffusivity_direction_choices : list of str
    diffusivity_choices : list of str
    correlation : list of str
    pos_file : list of str
    cel_file : list of str
    evp_file : list of str
    ion_file : list of str
    Conv_factor : float
    initial_time : list of float
    final_time : list of float
    initial_slope_time : list of float
    final_slope_time : list of float
    block : int
    rmax : float
    step_skip : int
    sigma : float
    ngrid : int
    output_file : str

    Returns
    -------
    temp_input_dict, temp_output_dict : dict, dict
    """
    # --------------------- CHECKS ---------------------
    if not temperature:
        with open(output_file, 'a') as f:
            f.write("Temperature list is empty. Exiting.\n")
        return None, None

    if not diffusing_elements:
        with open(output_file, 'a') as f:
            f.write("Diffusing elements list is empty. Exiting.\n")
        return None, None

    temp_input_dict = {}
    temp_output_dict = {}
    dt_dict = {}
    evaluated_data_dict = {}
    evaluated_corr_dict = {}

    # --------------------- MAIN LOOP ---------------------
    for temp_count in range(len(temperature)):
        temp = temperature[temp_count]
        # times for this temperature
        initial_times = initial_time[temp_count]
        final_times = final_time[temp_count]
        initial_slope_times = initial_slope_time[temp_count]
        final_slope_times = final_slope_time[temp_count]

        with open(output_file, 'a') as f:
            f.write(f"\nProcessing temperature: {temp}\n")

        for ele_count in range(len(diffusing_elements)):
            ele = diffusing_elements[ele_count]
            with open(output_file, 'a') as f:
                f.write(f"  Processing diffusing element: {ele}\n")

            # --------------------- READ FILES ---------------------
            inp_array = inp.read_ion_file(ion_file[temp_count])
            with open(output_file, 'a') as f:
                f.write("  Ion file read successfully.\n")

            cell_param_full = inp.read_cel_file(cel_file[temp_count], Conv_factor)
            (ke_elec_data_full, cell_temp_data_full, ion_temp_data_full, tot_energy_data_full, enthalpy_data_full,
             tot_energy_ke_ion_data_full, tot_energy_ke_ion_ke_elec_data_full, vol_data_full, pressure_data_full,
             number_of_time_frames, time_difference) = inp.read_evp_file(evp_file[temp_count], Conv_factor)

            pos_full, steps_full, dt_full, t_full = inp.read_pos_file(
                pos_file[temp_count],
                inp_array,
                Conv_factor,
                number_of_time_frames,
                time_difference
            )
            with open(output_file, 'a') as f:
                f.write("  CEL, EVP, and POS files read successfully.\n")

            # --------------------- SEGMENT DATA ---------------------
            First_term, Last_term = dp.find_terms(t_full.tolist(), initial_times, final_times)
            First_slope_term, Last_slope_term = dp.find_terms(dt_full.tolist(), initial_slope_times, final_slope_times)
            (pos, steps, dt, t, cell_param, ke_elec_data, cell_temp_data, ion_temp_data,
             tot_energy_data, enthalpy_data, tot_energy_ke_ion_data, tot_energy_ke_ion_ke_elec_data_full,
             vol_data, pressure_data) = dp.segmenter_func(
                First_term, Last_term, pos_full, dt_full, t_full,
                cell_param_full, ke_elec_data_full, cell_temp_data_full, ion_temp_data_full,
                tot_energy_data_full, enthalpy_data_full, tot_energy_ke_ion_data_full,
                tot_energy_ke_ion_ke_elec_data_full, vol_data_full, pressure_data_full
            )
            with open(output_file, 'a') as f:
                f.write("  Data segmented successfully.\n")

            avg_nsteps = int((5 / 100) * (len(dt)))
            (pos_array, rectified_structure_array, conduct_ions_array, frame_ions_array,
             frame_pos_array, conduct_pos_array, conduct_rectified_structure_array,
             frame_rectified_structure_array) = dp.data_evaluator(
                diffusivity_direction_choices, diffusing_elements[ele_count],
                pos, inp_array, steps
            )
            with open(output_file, 'a') as f:
                f.write("  Data evaluated successfully.\n")

            # --------------------- AVERAGE CELL PARAMS & VOLUME ---------------------
            volume = np.mean(vol_data)
            x1, x2, x3 = np.mean(cell_param[:, 0]), np.mean(cell_param[:, 1]), np.mean(cell_param[:, 2])
            y1, y2, y3 = np.mean(cell_param[:, 3]), np.mean(cell_param[:, 4]), np.mean(cell_param[:, 5])
            z1, z2, z3 = np.mean(cell_param[:, 6]), np.mean(cell_param[:, 7]), np.mean(cell_param[:, 8])

            ele_dict = {}
            ele_corr = {}

            # --------------------- DIFFUSIVITY DIRECTION DICTS ---------------------
            if diffusivity_direction_choices:
                for direction_idx, direction in enumerate(diffusivity_direction_choices):
                    ele_dict[direction] = {
                        'pos_array': pos_array[direction_idx, :, :, :],
                        'rectified_structure_array': rectified_structure_array[direction_idx, :, :, :],
                        'conduct_ions_array': conduct_ions_array[direction_idx, :],
                        'frame_ions_array': frame_ions_array[direction_idx, :],
                        'frame_pos_array': frame_pos_array[direction_idx, :, :, :],
                        'conduct_pos_array': conduct_pos_array[direction_idx, :, :, :],
                        'conduct_rectified_structure_array': conduct_rectified_structure_array[direction_idx, :, :, :],
                        'frame_rectified_structure_array': frame_rectified_structure_array[direction_idx, :, :, :]
                    }
                with open(output_file, 'a') as f:
                    f.write("  Diffusivity direction loop executed successfully.\n")
            else:
                with open(output_file, 'a') as f:
                    f.write("  Skipping diffusivity_direction_choices calculations (none provided).\n")

            # --------------------- CORRELATION ---------------------
            if correlation:
                for corr_type in correlation:
                    (pos_array_corr, rectified_structure_array_corr,
                     conduct_ions_array_corr, frame_ions_array_corr,
                     frame_pos_array_corr, conduct_pos_array_corr,
                     conduct_rectified_structure_array_corr,
                     frame_rectified_structure_array_corr) = dp.data_evaluator(
                        ['XYZ'], diffusing_elements[ele_count], pos, inp_array, steps
                    )
                    size_x = len(conduct_rectified_structure_array_corr[0, :, 0, 0])
                    size_y = len(conduct_rectified_structure_array_corr[0, 0, :, 0])
                    size_z = len(conduct_rectified_structure_array_corr[0, 0, 0, :])
                    structure = np.zeros((size_x, size_y, size_z))
                    structure[:, :, :] = np.array(conduct_rectified_structure_array_corr[0, :, :, :])

                    if corr_type == "Self":
                        dist_interval, reduced_nt, grt = corr.Van_Hove_self(
                            avg_nsteps, dt, rmax, step_skip, sigma, ngrid, structure
                        )
                    elif corr_type == "Distinct":
                        dist_interval, reduced_nt, grt = corr.Van_Hove_distinct(
                            avg_nsteps, dt, rmax, step_skip, sigma, ngrid, structure,
                            volume, x1, x2, x3, y1, y2, y3, z1, z2, z3
                        )
                    else:
                        # If you have more correlation types, handle them here
                        continue

                    ele_corr[corr_type] = {
                        'dist_interval': dist_interval,
                        'reduced_nt': reduced_nt,
                        'grt': grt
                    }
            else:
                with open(output_file, 'a') as f:
                    f.write("  Skipping correlation calculations (none provided).\n")

            evaluated_data_dict[ele] = ele_dict
            evaluated_corr_dict[ele] = ele_corr

            # --------------------- MSD CALCULATION ---------------------
            if diffusivity_choices and diffusivity_direction_choices:
                msd_data_dict = cal.calculate_msd(
                    diffusing_elements, diffusivity_direction_choices, diffusivity_choices,
                    pos_full, conduct_pos_array, conduct_ions_array,
                    dt, Last_term, First_slope_term, Last_slope_term, block
                )
            else:
                with open(output_file, 'a') as f:
                    f.write("  Skipping calculate_msd (missing directions or choices).\n")
                msd_data_dict = None

            # --------------------- STORE RESULTS ---------------------
            temp_input_dict[(temp, ele)] = {'evaluated_data': evaluated_data_dict[ele]}
            temp_output_dict[(temp, ele)] = {
                'dt_dict': dt,
                'msd_data': msd_data_dict,
                'evaluated_corr': evaluated_corr_dict[ele]
            }

            with open(output_file, 'a') as f:
                f.write(f"  Processing completed for diffusing element: {ele}\n")

        with open(output_file, 'a') as f:
            f.write(f"Processing completed for temperature: {temp}\n")

    return temp_input_dict, temp_output_dict

def main():
    # 1) Parse command-line arguments
    args = parse_arguments()

    # 2) (Optional) Basic sanity checks 
    ntemps = len(args.temperature)
    if (len(args.pos_file) != ntemps or 
        len(args.cel_file) != ntemps or
        len(args.evp_file) != ntemps or
        len(args.ion_file) != ntemps):
        sys.exit("Error: pos-file/cel-file/evp-file/ion-file counts must match number of temperatures!")

    # Also check initial/final time arrays match
    if not (len(args.initial_time) == ntemps == len(args.final_time) ==
            len(args.initial_slope_time) == len(args.final_slope_time)):
        sys.exit("Error: initial_time/final_time arrays must match number of temperatures!")

    # 3) Call the Job() function
    temp_input_dict, temp_output_dict = Job(
        temperature=args.temperature,
        diffusing_elements=args.diffusing_elements,
        diffusivity_direction_choices=args.diffusivity_direction_choices,
        diffusivity_choices=args.diffusivity_choices,
        correlation=args.correlation,
        pos_file=args.pos_file,
        cel_file=args.cel_file,
        evp_file=args.evp_file,
        ion_file=args.ion_file,
        Conv_factor=args.Conv_factor,
        initial_time=args.initial_time,
        final_time=args.final_time,
        initial_slope_time=args.initial_slope_time,
        final_slope_time=args.final_slope_time,
        block=args.block,
        rmax=args.rmax,
        step_skip=args.step_skip,
        sigma=args.sigma,
        ngrid=args.ngrid,
        output_file=args.output_file
    )

    print(f"Job() complete. Log written to: {args.output_file}")
    
    # 4) If user specified a --json-output, dump temp_output_dict to JSON
    if args.json_output:
        print(f"Saving temp_output_dict to JSON: {args.json_output}")
        temp_out_data_serializable = serializable.convert_to_serializable(temp_output_dict)
        with open(args.json_output, 'w') as f:
            json.dump(temp_out_data_serializable, f, indent=2)
        print("JSON saved.")

if __name__ == "__main__":
    main()
