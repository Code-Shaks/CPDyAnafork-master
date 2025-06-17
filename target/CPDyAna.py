#!/usr/bin/env python3
"""
CPDyAna – Combined version
=========================
This single‑file variant merges the original CPDyAna.py driver with the helper
functions that used to live in data_processing.py so that the whole workflow
can be imported or executed without relying on a separate local package.

Usage remains identical to the two‑file setup:
    python CPDyAna.py msd -T 800 --data-dir . ...
    python CPDyAna.py vh -T 800 ...
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

# --- internal imports (external to this file) ---------------------------------
from . import correrelation_analysis as corr
from . import input_reader as inp
from . import calculations as cal
from . import json_serializable as js
from . import plotting as p
from . import data_processing as dp
from . import probability_density as prob
from . import compute_rdf as rdf

def Job(temperature, diffusing_elements, diffusivity_direction_choices, diffusivity_choices, correlation, 
        pos_file, cel_file, evp_file, ion_file, Conv_factor, initial_time, final_time, 
        initial_slope_time, final_slope_time, block, rmax, step_skip, sigma, ngrid, mode=None):
    """
    Main job function for MSD, VH, and related analyses.

    Args:
        temperature (list): List of temperatures.
        diffusing_elements (list): Elements to analyze.
        diffusivity_direction_choices (list): Directions for diffusivity.
        diffusivity_choices (list): Types of diffusivity.
        correlation (list): Correlation types.
        pos_file, cel_file, evp_file, ion_file (list): File paths.
        Conv_factor (float): Conversion factor.
        initial_time, final_time, initial_slope_time, final_slope_time (float): Time windows.
        block (int): Block size.
        rmax (float): Maximum radius.
        step_skip (int): Step skip.
        sigma (float): Sigma for Gaussian.
        ngrid (int): Number of grid points.
        mode (str): Analysis mode.

    Returns:
        tuple: (input_dict, output_dict) with analysis results.
    """
    temp_input_dict, temp_output_dict = {}, {}
    for temp_count, temp in enumerate(temperature):
        inp_array = inp.read_ion_file(ion_file[temp_count])
        cell_param_full = inp.read_cel_file(cel_file[temp_count], Conv_factor)
        (ke_elec_full, cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, 
         tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full, 
         n_frames, time_diff) = inp.read_evp_file(evp_file[temp_count], Conv_factor)
        pos_full, steps_full, dt_full, t_full = inp.read_pos_file(pos_file[temp_count], inp_array, 
                                                              Conv_factor, n_frames, time_diff)
        
        First_term, Last_term = dp.find_terms(t_full.tolist(), initial_time, final_time)
        (pos, steps, dt, t, cell_param, ke_elec, cell_temp, ion_temp, tot_energy, 
         enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure) = dp.segmenter_func(
            First_term, Last_term, pos_full, dt_full, t_full, cell_param_full, ke_elec_full, 
            cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, 
            tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full)
        
        for ele in diffusing_elements:
            (pos_array, rectified_structure_array, conduct_ions_array, frame_ions_array, 
             frame_pos_array, conduct_pos_array, conduct_rectified_structure_array, 
             frame_rectified_structure_array) = dp.data_evaluator(diffusivity_direction_choices, 
                                                               [ele], pos, inp_array, steps)
            
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
            
            msd_data_dict = {}
            if mode == "msd":
                msd_data_dict = cal.calculate_msd([ele], diffusivity_direction_choices, diffusivity_choices, 
                                          pos_full, conduct_rectified_structure_array, 
                                          conduct_ions_array, dt, Last_term, initial_slope_time, 
                                          final_slope_time, block)
            
            evaluated_corr_dict = {}
            if mode == "vh":
                for correlation_type in ['Self', 'Distinct']:
                    if correlation_type == 'Self':
                        dist_interval, reduced_nt, grt = corr.Van_Hove_self(
                            avg_step=min(100, len(dt)//4),
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
                            evaluated_corr_dict[correlation_type] = {
                                'grt': [],
                                'dist_interval': [],
                                'reduced_nt': 0
                            }

            temp_input_dict[(temp, ele)] = {'evaluated_data': ele_dict, 'evaluated_corr': evaluated_corr_dict}
            temp_output_dict[(temp, ele)] = {'dt_dict': dt, 'msd_data': msd_data_dict, 'evaluated_corr': evaluated_corr_dict}
    return temp_input_dict, temp_output_dict

def parser():
    """
    Argument parser for CPDyAna CLI.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="CPDyAna CLI – combined version")
    sub = p.add_subparsers(dest="mode", required=True)

    msd = sub.add_parser("msd")
    vh = sub.add_parser("vh")
    ionic_density = sub.add_parser("ionic-density")
    rdf = sub.add_parser("rdf")
    vaf = sub.add_parser("vaf")
    vdos = sub.add_parser("vdos")

    for sp in (msd, vh):
        sp.add_argument(
            "-T", "--temperature",
            nargs="+", type=float,
            default=[800.0],
            help="Temperature(s) in K (default: 800)"
        )
        sp.add_argument(
            "-e", "--diffusing-elements",
            nargs="+", default=["Li"],
            help="Diffusing elements (default: ['Li'])"
        )
        sp.add_argument(
            "--data-dir", required=True,
            help="Directory containing data files (required)"
        )
        sp.add_argument(
            "--diffusivity-direction-choices",
            nargs="+", default=["XYZ"],
            help="Diffusivity direction choices (default: ['XYZ'])"
        )
        sp.add_argument(
            "--diffusivity-choices",
            nargs="+", default=["Tracer"],
            help="Diffusivity choices (default: ['Tracer'])"
        )
        sp.add_argument(
            "--correlation",
            nargs="*", default=["Self"],
            help="Correlation type(s) (default: ['Self'])"
        )
        sp.add_argument(
            "--initial-time",
            type=float,
            default=2.0,
            help="Initial time for analysis (default: 2.0)"
        )
        sp.add_argument(
            "--final-time",
            type=float,
            default=200.0,
            help="Final time for analysis (default: 200.0)"
        )
        sp.add_argument(
            "--initial-slope-time",
            type=float,
            default=5.0,
            help="Initial slope time (default: 5.0)"
        )
        sp.add_argument(
            "--final-slope-time",
            type=float,
            default=100.0,
            help="Final slope time (default: 100.0)"
        )
        sp.add_argument(
            "--Conv-factor",
            type=float,
            default=0.529177249,
            help="Conversion factor (default: 0.529177249)"
        )
        sp.add_argument(
            "--block",
            type=int,
            default=500,
            help="Block size (default: 500)"
        )
        sp.add_argument(
            "--rmax",
            type=float,
            default=10,
            help="Maximum radius (rmax) for analysis (default: 10)"
        )
        sp.add_argument(
            "--step-skip",
            type=int,
            default=500,
            help="Number of steps to skip (default: 500)"
        )
        sp.add_argument(
            "--sigma",
            type=float,
            default=0.1,
            help="Sigma value (default: 0.1)"
        )
        sp.add_argument(
            "--ngrid",
            type=int,
            default=101,
            help="Number of grid points (default: 101)"
        )
        sp.add_argument(
            "--json-output",
            default="OUTPUT.json",
            help="JSON output file (default: 'OUTPUT.json')"
        )

    # MSD-specific arguments
    msd.add_argument(
        "--plot-data",
        action="append", nargs="+",
        default=None,
        help="Tuples T, element, dir, type to plot. If omitted, plot everything."
    )

    msd.add_argument(
        "--first-time",
        type=float,
        default=2.0,
        help="First time for MSD plot (default: 2.0)"
    )
    msd.add_argument(
        "--last-time",
        type=float,
        default=200.0,
        help="Last time for MSD plot (default: 200.0)"
    )
    msd.add_argument(
        "--save-path",
        default="MSD.jpg",
        help="File path to save MSD plot (default: 'MSD.jpg')"
    )

    # Van Hove-specific arguments
    vh.add_argument(
        "--plot-data",
        action="append", nargs="+",
        default=None,
        help="Tuples T, element, dir, type to plot. If omitted, plot everything."
    )

    vh.add_argument(
        "--figsize",
        nargs=2, type=float,
        default=[10, 8],
        help="Figure size for Van Hove plot (default: [10, 8])"
    )
    vh.add_argument(
        "--save-path",
        default="van_hove_plot.png",
        help="File path to save Van Hove plot (default: 'van_hove_plot.png')"
    )

    # ionic_density specific arguments
    ionic_density.add_argument(
        "--data-dir", required=True,
        help="Directory containing .pos, .in, and .cel files"
    )
    ionic_density.add_argument(
        "--time-after-start", type=float, default=0.0,
        help="Time (ps) after which to start extracting frames (default: 0.0)"
    )
    ionic_density.add_argument(
        "--num-frames", type=int, default=0,
        help="Number of frames to extract (default: 0 = all frames)"
    )
    ionic_density.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Time interval between frames (default: 0.00193511)"
    )
    ionic_density.add_argument(
        "--element", default="Li",
        help="Element to analyze (default: Li)"
    )
    ionic_density.add_argument(
        "--sigma", type=float, default=0.3,
        help="Gaussian sigma for density calculation (default: 0.3)"
    )
    ionic_density.add_argument(
        "--n-sigma", type=float, default=4.0,
        help="Number of sigma for cutoff (default: 4.0)"
    )
    ionic_density.add_argument(
        "--density", type=float, default=0.2,
        help="Density scaling factor (default: 0.2)"
    )
    ionic_density.add_argument(
        "--output", default="density.xsf",
        help="Output file name (default: density.xsf)"
    )
    ionic_density.add_argument(
        "--step-skip", type=int, default=1,
        help="Number of steps to skip between frames (default: 1)"
    )
    ionic_density.add_argument(
        "--mask", default=None,
        help="Atom mask for density calculation (e.g. '0,1,2,5-10')"
    )
    ionic_density.add_argument(
        "--recenter", action="store_true",
        help="Recenter trajectory before density calculation"
    )
    ionic_density.add_argument(
        "--bbox", default=None,
        help="Bounding box for grid as 'xmin,xmax,ymin,ymax,zmin,zmax' (comma-separated)"
    )
    # RDF specific arguments
    rdf.add_argument(
        "--data-dir", required=True,
        help="Directory containing .pos, .in, .cel files"
    )
    rdf.add_argument(
        "--time-after-start", type=float, default=60,
        help="Time after start for RDF analysis in ps (default: 60)"
    )
    rdf.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Time interval between frames in ps (default: 0.00193511)"
    )
    rdf.add_argument(
        "--num-frames", type=int, default=100,
        help="Number of frames to analyze (default: 100)"
    )
    rdf.add_argument(
        "--central-atom", nargs="+", default=['Li','Al','P','S'],
        help="Central atom(s) for RDF analysis (default: ['Li', 'Al', 'P', 'S'])"
    )
    rdf.add_argument('--pair-atoms', nargs='+', default=None,
                        help='List of pair atom types for RDF (default: same as central-atoms)'
    )
    rdf.add_argument('--ngrid', type=int, default=1001,
                        help='Number of radial grid points for RDF'
                        ' (default: 1001)')
    rdf.add_argument('--rmax', type=float, default=10.0,
                        help='Maximum distance for RDF calculation (Å)'
                        ' (default: 10.0)')
    rdf.add_argument('--sigma', type=float, default=0.2,
                        help='Gaussian broadening parameter for RDF'
                        ' (default: 0.2)')
    rdf.add_argument('--xlim', nargs=2, type=float, default=[1.5, 8.0],
                        help='x-axis limits for RDF plot (default: 1.5 8.0)'
                        ' (in Å)')

    # VAF specific arguments
    vaf.add_argument(
        "--data-dir", required=True,
        help="Directory containing .pos, .in, .cel files (and optionally .evp)"
    )
    vaf.add_argument(
        "--element", nargs="+", required=True,
        help="Atom symbol(s) for VAF (e.g. Li Na)"
    )
    vaf.add_argument(
        "--start", type=float, default=0.0,
        help="Time (ps) to start analysis"
    )
    vaf.add_argument(
        "--nframes", type=int, default=0,
        help="Number of frames (set 0 for all frames)"
    )
    vaf.add_argument(
        "--stride", type=int, default=1,
        help="Stride for frames (1=all, 2=every other, etc.)"
    )
    vaf.add_argument(
        "--blocks", type=int, default=4,
        help="Number of blocks for error estimates"
    )
    vaf.add_argument(
        "--out-prefix", default="vaf",
        help="Prefix for output files"
    )
    vaf.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Default time between frames (ps) if no .evp file"
    )
    vaf.add_argument(
        "--t-start-fit-ps", type=int, default=0,
        help="Start frame index for VAF"
    )
    vaf.add_argument(
        "--stepsize-t", type=int, default=1,
        help="Stride for t in VAF"
    )
    vaf.add_argument(
        "--stepsize-tau", type=int, default=1,
        help="Stride for tau in VAF"
    )
    vaf.add_argument(
        "--t-end-fit-ps", type=float, default=10,
        help="End of the fit in ps"
    )
     
    # VDOS specific arguments
    vdos.add_argument(
        "--data-dir", required=True,
        help="Directory containing .pos, .in, .cel files (and optionally .evp)"
    )
    vdos.add_argument(
        "--elements", nargs="+", default=["Li", "Al", "P", "S"],
        help="Elements to plot (default: Li Al P S)"
    )
    vdos.add_argument(
        "--out-prefix", default="vdos",
        help="Prefix for output files"
    )
    vdos.add_argument(
        "--start", type=float, default=0.0,
        help="Time (ps) to start analysis"
    )
    vdos.add_argument(
        "--nframes", type=int, default=0,
        help="Number of frames (set 0 for all frames)"
    )
    vdos.add_argument(
        "--stride", type=int, default=1,
        help="Stride for frames (1=all, 2=every other, etc.)"
    )
    vdos.add_argument(
        "--time-interval", type=float, default=0.00193511,
        help="Default time between frames (ps) if no .evp file"
    )
    return p.parse_args()

def main():
    """
    Main entry point for CPDyAna CLI.
    Handles dispatching to the correct analysis mode and manages file I/O.
    """
    a = parser()

    if a.mode in ("msd", "vh"):
        # Find files in data-dir
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))
        evp_files = sorted(glob.glob(os.path.join(a.data_dir, "*.evp")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))

        if not (pos_files and cel_files and evp_files and ion_files):
            sys.exit("Missing data files in data directory")
        if not (len(pos_files) == len(cel_files) == len(evp_files) == len(ion_files)):
            sys.exit("Mismatch in number of .pos, .cel, .evp, .in files")

        Temp_inp_data, Temp_out_data = Job(
            a.temperature, a.diffusing_elements, a.diffusivity_direction_choices,
            a.diffusivity_choices, a.correlation, pos_files, cel_files, evp_files, ion_files,
            a.Conv_factor, a.initial_time, a.final_time, a.initial_slope_time,
            a.final_slope_time, a.block, a.rmax, a.step_skip, a.sigma, a.ngrid, a.mode
        )

        if a.json_output:
            Temp_out_data_serializable = js.convert_to_serializable(Temp_out_data)
            with open(a.json_output, 'w') as output_file:
                json.dump(Temp_out_data_serializable, output_file)
            data_source = a.json_output
        else:
            data_source = Temp_out_data

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
        
        p.msd_plot(data_source, pdata, a.first_time, a.last_time, save_path=a.save_path)

    elif a.mode == "vh":
        if a.plot_data is None:
            pdata = []
            for (T, ele), blob in Temp_out_data.items():
                for corr_type in blob["evaluated_corr"]:
                    pdata.append([T, ele, corr_type])
        else:
            pdata = [[float(x[0]), x[1], x[2]] for x in a.plot_data]

        p.van_hove_plot(
            data_source,
            pdata,
            save_path=a.save_path,
            figsize=tuple(a.figsize)
        )

    elif a.mode == "ionic-density":
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))

        if not pos_files or not ion_files:
            sys.exit("Missing .pos or .in files in data directory")
        if not cel_files:
            sys.exit("Ionic density analysis requires .cel files. No .cel files found in data directory")
        if len(pos_files) != len(ion_files) or len(pos_files) != len(cel_files):
            sys.exit("Number of .pos, .in, and .cel files must match")

        results = {}
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            pos_file = os.path.abspath(pos_file)
            ion_file = os.path.abspath(ion_file)
            cel_file = os.path.abspath(cel_file)

            missing_files = []
            if not os.path.exists(pos_file):
                missing_files.append(pos_file)
            if not os.path.exists(ion_file):
                missing_files.append(ion_file)
            if not os.path.exists(cel_file):
                missing_files.append(cel_file)
            if missing_files:
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

                subprocess.run(cmd, check=True, capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
                results[base_name] = {"density_file": output_file}
            except Exception:
                continue

    elif a.mode == "rdf":
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))

        if not pos_files or not ion_files:
            sys.exit("Missing .pos or .in files in data directory")
        if len(pos_files) != len(ion_files):
            sys.exit("Number of .pos and .in files must match")
        if not cel_files:
            sys.exit("RDF analysis requires .cel files. No .cel files found in data directory")
        if len(cel_files) != len(pos_files):
            sys.exit("Number of .cel files must match .pos files for RDF analysis")    

        results = {}
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            try:
                base_name = os.path.splitext(os.path.basename(pos_file))[0]
                output_prefix = f"rdf_plot_{base_name}"

                # Pass all user CLI options to build_ase_trajectory and compute_rdf
                extracted_frames = rdf.build_ase_trajectory(
                    ion_file, pos_file, cel_file,
                    time_after_start=getattr(a, "time_after_start", 60),
                    num_frames=getattr(a, "num_frames", 100),
                    time_interval=getattr(a, "time_interval", 0.00193511),
                )

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
            except Exception:
                continue

    if a.mode == "vaf":
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))
        evp_files = sorted(glob.glob(os.path.join(a.data_dir, "*.evp")))

        if not pos_files or not ion_files or not cel_files:
            sys.exit("Missing .pos, .in, or .cel files in data directory")
        if not (len(pos_files) == len(ion_files) == len(cel_files)):
            sys.exit("Mismatch in number of .pos, .in, .cel files")

        results = {}
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            base_name = os.path.splitext(os.path.basename(pos_file))[0]
            evp_file = evp_files[i] if i < len(evp_files) else None

            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "vaf.py"),
                "--in_file", ion_file,
                "--pos_file", pos_file,
                "--cel_file", cel_file,
                "--element"
            ] + a.element

            if evp_file:
                cmd += ["--evp_file", evp_file]
            cmd += [
                "--start", str(a.start),
                "--nframes", str(a.nframes),
                "--stride", str(a.stride),
                "--blocks", str(a.blocks),
                "--out_prefix", f"{a.out_prefix}_{base_name}",
                "--t_end_fit_ps", str(a.t_end_fit_ps),
                "--time_interval", str(a.time_interval),
                "--t_start_fit_ps", str(a.t_start_fit_ps),
                "--stepsize_t", str(a.stepsize_t),
                "--stepsize_tau", str(a.stepsize_tau)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                results[base_name] = {"vaf_output_prefix": f"{a.out_prefix}_{base_name}"}
            except subprocess.CalledProcessError:
                continue
            except Exception:
                continue

    if a.mode == "vdos":
        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
        cel_files = sorted(glob.glob(os.path.join(a.data_dir, "*.cel")))
        evp_files = sorted(glob.glob(os.path.join(a.data_dir, "*.evp")))

        if not pos_files or not ion_files or not cel_files:
            return
        if not (len(pos_files) == len(ion_files) == len(cel_files)):
            return

        # For each set of files, call vdos.py
        for i, (pos_file, ion_file, cel_file) in enumerate(zip(pos_files, ion_files, cel_files)):
            evp_file = evp_files[i] if i < len(evp_files) else None
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "vdos.py"),
                "--in_file", ion_file,
                "--pos_file", pos_file,
                "--cel_file", cel_file,
                "--out_prefix", a.out_prefix,
                "--start", str(a.start),
                "--nframes", str(a.nframes),
                "--stride", str(a.stride),
                "--time_interval", str(a.time_interval),
                "--elements"
            ] + a.elements
            if evp_file:
                cmd += ["--evp_file", evp_file]
            subprocess.run(cmd, check=True)
        return

if __name__ == "__main__":
    main()