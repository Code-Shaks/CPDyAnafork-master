#!/usr/bin/env python3
"""
CPDyAna – Combined version
=========================
This single‑file variant merges the original **CPDyAna.py** driver with the helper
functions that used to live in **data_processing.py** so that the whole workflow
can be imported or executed without relying on a separate local package.

Usage remains identical to the two‑file setup:
    python CPDyAna_combined.py msd -T 300 400 ...
    python CPDyAna_combined.py van_hove -T 300 ...
"""
import argparse
import json
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm

# --- internal imports (external to this file) ---------------------------------
from . import correrelation_analysis as corr  # noqa: E402
from . import input_reader as inp             # noqa: E402
from . import calculations as cal             # noqa: E402
from . import serializable                    # noqa: E402
from . import Plotting as p                   # noqa: E402
from . import data_processing as dp           # noqa: E402
from . import probability_density as prob           # noqa: E402

# -----------------------------------------------------------------------------
#                             Data‑processing helpers
# -----------------------------------------------------------------------------

# def find_terms(array, first_value, last_value):
#     """Return the indices of the first and last elements in `array`
#        that fall within [first_value, last_value].
#     """
#     # Ensure we have a NumPy array
#     arr = np.array(array)

#     # Create a boolean mask for elements in the specified range
#     mask = (arr >= first_value) & (arr <= last_value)

#     # Extract the indices where the mask is True
#     idx = np.where(mask)[0]
#     if idx.size == 0:
#         # No elements found in range
#         raise ValueError(f"No elements in array between {first_value} and {last_value}.")

#     # Return the first and last such indices
#     return idx[0], idx[-1]


# def segmenter_func(first_term, last, pos_full, dt_full, time_full, cell_full,
#                    ke_elec_full, cell_temp_full, ion_temp_full, tot_energy_full,
#                    enthalpy_full, tot_energy_ke_ion_full,
#                    tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full):
#     """Cut out a trajectory window and return the sliced arrays."""
#     if last <= int(0.7 * len(pos_full[0, :, 0])):
#         last_term = int(last / 0.7)
#     else:
#         last_term = int(last)

#     pos = pos_full[:, first_term:last_term, :]
#     step_counts = len(pos[0, :, 0])
#     dt = dt_full[first_term:last_term - 1]
#     time = time_full[first_term:last_term]

#     cell = cell_full[first_term:last_term, :]
#     ke_elec = ke_elec_full[first_term:last_term]
#     cell_temp = cell_temp_full[first_term:last_term]
#     ion_temp = ion_temp_full[first_term:last_term]
#     tot_energy = tot_energy_full[first_term:last_term]
#     enthalpy = enthalpy_full[first_term:last_term]
#     tot_energy_ke_ion = tot_energy_ke_ion_full[first_term:last_term]
#     tot_energy_ke_ion_ke_elec = tot_energy_ke_ion_ke_elec_full[first_term:last_term]
#     vol = vol_full[first_term:last_term]
#     pressure = pressure_full[first_term:last_term]

#     return (pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp,
#             tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec,
#             vol, pressure)


# def data_evaluator(diffusivity_direction_choices, target_elements, pos,
#                    total_ion_array, steps):
#     """Generate drift‑corrected coordinate and index arrays for MSD analysis."""
#     position_data_list = []
#     drifted_rectified_structure_list = []
#     conductor_indices_list = []
#     framework_indices_list = []
#     framework_pos_list = []
#     mobile_pos_list = []
#     pos_list = []
#     mobile_drifted_rectified_structure_list = []
#     framework_drifted_rectified_structure_list = []

#     for direction in diffusivity_direction_choices:
#         position_data = np.zeros((len(total_ion_array), steps, 3))

#         # Select coordinate components ------------------------------------------------
#         if direction == "XYZ":
#             position_data = pos.copy()
#         elif direction == "XY":
#             position_data[:, :, :2] = pos[:, :, :2]
#         elif direction == "YZ":
#             position_data[:, :, 1:] = pos[:, :, 1:]
#         elif direction == "ZX":
#             position_data[:, :, 0] = pos[:, :, 0]
#             position_data[:, :, 2] = pos[:, :, 2]
#         elif direction == "X":
#             position_data[:, :, 0] = pos[:, :, 0]
#         elif direction == "Y":
#             position_data[:, :, 1] = pos[:, :, 1]
#         elif direction == "Z":
#             position_data[:, :, 2] = pos[:, :, 2]

#         # Displacements --------------------------------------------------------------
#         disp = position_data - position_data[:, 0, :][:, None, :]

#         # Separate mobile (conductor) and framework indices --------------------------
#         conductor_indices = [i for i, el in enumerate(total_ion_array)
#                              if el in target_elements]
#         framework_indices = [i for i in range(len(total_ion_array))
#                              if i not in conductor_indices]

#         framework_disp = disp[framework_indices]
#         framework_pos = position_data[framework_indices]
#         mobile_pos = position_data[conductor_indices]

#         # Drift correction -----------------------------------------------------------
#         drift = np.average(framework_disp, axis=0)
#         corrected_displacements = disp - drift
#         drifted_rectified_structure = (position_data[:, 0, :][:, None, :] +
#                                        corrected_displacements)

#         mobile_drifted_rectified_structure = drifted_rectified_structure[conductor_indices]
#         framework_drifted_rectified_structure = drifted_rectified_structure[framework_indices]

#         # Collect --------------------------------------------------------------------
#         position_data_list.append(position_data)
#         drifted_rectified_structure_list.append(drifted_rectified_structure)
#         conductor_indices_list.append(conductor_indices)
#         framework_indices_list.append(framework_indices)
#         framework_pos_list.append(framework_pos)
#         mobile_pos_list.append(mobile_pos)
#         pos_list.append(pos)
#         mobile_drifted_rectified_structure_list.append(mobile_drifted_rectified_structure)
#         framework_drifted_rectified_structure_list.append(framework_drifted_rectified_structure)

#     return (np.array(position_data_list),
#             np.array(drifted_rectified_structure_list),
#             np.array(conductor_indices_list),
#             np.array(framework_indices_list),
#             np.array(framework_pos_list),
#             np.array(mobile_pos_list),
#             np.array(mobile_drifted_rectified_structure_list),
#             np.array(framework_drifted_rectified_structure_list))

# -----------------------------------------------------------------------------
#                                   Core job
# -----------------------------------------------------------------------------

def Job(Ts, elems, dirs, diff_choices, cors, ddir, conv, it, ft, ist, fst, blk,
        rmax, skip, sig, ng, log="log.txt"):
    """Main routine that parses trajectory folders and performs analysis."""
    pos = sorted(glob.glob(os.path.join(ddir, "*.pos")))
    cel = sorted(glob.glob(os.path.join(ddir, "*.cel")))
    evp = sorted(glob.glob(os.path.join(ddir, "*.evp")))
    ion = sorted(glob.glob(os.path.join(ddir, "*.in")))

    for lst in (pos, cel, evp, ion):
        if len(lst) < len(Ts):
            sys.exit("Missing data files")

    tout = {}
    for i, T in enumerate(Ts):
        for ele in elems:
            ia = inp.read_ion_file(ion[i])
            cell = inp.read_cel_file(cel[i], conv)
            (ke_elec, cell_t, ion_t, totE, enth, ke_ion, ke_tot, vol,
             pres, nfrm, dtb) = inp.read_evp_file(evp[i], conv)
            pfull, _, dtfull, tfull = inp.read_pos_file(pos[i], ia, conv, nfrm, dtb)

            F, L = dp.find_terms(tfull, it[i], ft[i])
            Fs, Ls = dp.find_terms(dtfull, ist[i], fst[i])

            seg = dp.segmenter_func(F, L, pfull, dtfull, tfull, cell, ke_elec,
                                  cell_t, ion_t, totE, enth, ke_ion, ke_tot,
                                  vol, pres)
            (pos_s, steps, dt, _, cmat, *_ ) = seg

            avg = int(0.05 * len(dt))

            (p_arr, r_arr, cion_arr, fion_arr, fpos_arr, cpos_arr,
             crect_arr, frect_arr) = dp.data_evaluator(dirs, ele, pos_s, ia, steps)

            vol_avg = np.mean(seg[-2])
            cell_m = np.mean(cmat, axis=0)
            x1, x2, x3, y1, y2, y3, z1, z2, z3 = cell_m

            e_dict, c_dict = {}, {}
            for d_idx, d in enumerate(dirs):
                e_dict[d] = {
                    "pos_array": p_arr[d_idx],
                    "rectified_structure_array": r_arr[d_idx],
                    "conduct_ions_array": cion_arr[d_idx],
                    "frame_ions_array": fion_arr[d_idx],
                    "frame_pos_array": fpos_arr[d_idx],
                    "conduct_pos_array": cpos_arr[d_idx],
                    "conduct_rectified_structure_array": crect_arr[d_idx],
                    "frame_rectified_structure_array": frect_arr[d_idx],
                }

            for ctype in cors:
                struct = crect_arr[0]
                if ctype == "Self":
                    dist, red, grt = corr.Van_Hove_self(avg, dt, rmax, skip, sig,
                                                         ng, struct)
                else:
                    dist, red, grt = corr.Van_Hove_distinct(avg, dt, rmax, skip,
                                                            sig, ng, struct,
                                                            vol_avg, x1, x2,
                                                            x3, y1, y2, y3,
                                                            z1, z2, z3)
                c_dict[ctype] = {
                    "dist_interval": dist,
                    "reduced_nt": red,
                    "grt": grt,
                }

            msd = (cal.calculate_msd(elems, dirs, diff_choices, pfull, cpos_arr,
                                      cion_arr, dt, L, Fs, Ls, blk,(Fs-Ls)/10)
                   if diff_choices else None)

            tout[(T, ele)] = {
                "dt_dict": dt,
                "msd_data": msd,
                "evaluated_corr": c_dict,
            }

    return tout

# -----------------------------------------------------------------------------
#                                Plot helpers
# -----------------------------------------------------------------------------

# def msd_plot(data, pdata, ft, lt, save=None):
#     d = data if isinstance(data, dict) else json.load(open(data))

#     fig, ax = plt.subplots()

#     for T, ele, dim, lab in pdata:        # note: order already swapped
#         suf = "" if dim == "XYZ" else f"_{dim}"
#         k   = f"({T}, '{ele}')"           # dict key for this run/element

#         if k not in d or ele not in d[k]["msd_data"]:
#             continue

#         # --- pull MSD and its matching time axis ---------------------------
#         mkey = f"{lab}_msd_array{suf}"
#         tkey = f"{lab}_time_array{suf}"
#         dkey = f"{lab}_diffusivity{suf}"
#         skey = f"{lab}_slope_sem{suf}"
#         ekey = f"{lab}_diffusivity_error{suf}"

#         try:
#             msd = d[k]["msd_data"][ele][mkey][0]
#             t   = d[k]["msd_data"][ele][tkey][0]
#         except KeyError:
#             continue         # skip if that combo wasn’t computed

#         # --- window indices on the *block* grid ----------------------------
#         F, L = find_terms(t, ft[0], lt[0])

#         sem = np.full_like(t, d[k]["msd_data"][ele][skey][0])

#         ax.plot(t[F:L], msd[F:L],
#                 label=(f"{dim}: D="
#                        f"{d[k]['msd_data'][ele][dkey][0]:.3e}±"
#                        f"{d[k]['msd_data'][ele][ekey][0]:.1e}"))
#         ax.fill_between(t[F:L], msd[F:L]-sem[F:L], msd[F:L]+sem[F:L], alpha=.2)

#     ax.set_xlabel("Time [ps]")
#     ax.set_ylabel("MSD [Å²]")
#     handles, labels = ax.get_legend_handles_labels()
#     unique = dict(zip(labels, handles))
#     ax.legend(unique.values(), unique.keys(), loc='best') 
#     if save:
#         plt.savefig(save, dpi=500)
#     plt.show()

# def van_hove_plot(data, pdata, save=None, figsize=(10, 8)):
#     d = data if isinstance(data, dict) else json.load(open(data))

#     for T, ele, h in pdata:
#         k = f"({T}, '{ele}')"
#         if k not in d or h not in d[k]["evaluated_corr"]:
#             continue

#         vh   = np.array(d[k]["evaluated_corr"][h]["grt"])
#         dist = d[k]["evaluated_corr"][h]["dist_interval"]

#         # ─── NEW: transpose if rows≠time ─────────────────────────────────
#         if vh.shape[0] == len(dist):      # r × t  →  t × r
#             vh = vh.T
#         # ────────────────────────────────────────────────────────────────

#         y  = np.linspace(0, dist[-1], len(dist))
#         dt_step = float(d[k]["dt_dict"][0])          # size of one MD step in ps
#         x = np.arange(vh.shape[0]) * dt_step * a.step_skip 

#         vmax, lab = (1, "Gs") if h == "Self" else (4, "Gd")
#         X, Y = np.meshgrid(x, y, indexing="ij")

#         plt.figure(figsize=figsize)
#         plt.pcolor(X, Y, vh, cmap="jet", vmin=vh.min(), vmax=vmax)
#         plt.xlabel("Time (ps)")
#         plt.ylabel("r (Å)")
#         plt.colorbar(label=lab)
#         plt.tight_layout()
#         if save:
#             plt.savefig(save, dpi=300)
#         plt.show()

# -----------------------------------------------------------------------------
#                                CLI interface
# -----------------------------------------------------------------------------

def parser():
    import argparse
    p = argparse.ArgumentParser(description="CPDyAna CLI – combined version")
    sub = p.add_subparsers(dest="mode", required=True)

    msd = sub.add_parser("msd")
    vh = sub.add_parser("vh")
    ionic_density = sub.add_parser("ionic-density")

    for sp in (msd, vh):
        sp.add_argument(
            "-T", "--temperature",
            nargs="+", type=float,
            default=[600.0],
            help="Temperature(s) in K (default: 600)"
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
            nargs="+", type=float,
            default=[0],
            help="Initial time(s) for analysis (default: [0])"
        )
        sp.add_argument(
            "--final-time",
            nargs="+", type=float,
            default=[10],
            help="Final time(s) for analysis (default: [10])"
        )
        sp.add_argument(
            "--initial-slope-time",
            nargs="+", type=float,
            default=[2],
            help="Initial slope time(s) (default: [2])"
        )
        sp.add_argument(
            "--final-slope-time",
            nargs="+", type=float,
            default=[8],
            help="Final slope time(s) (default: [8])"
        )
        sp.add_argument(
            "--Conv-factor",
            type=float,
            default=1.0,
            help="Conversion factor (default: 1.0)"
        )
        sp.add_argument(
            "--block",
            type=int,
            default=5,
            help="Block size (default: 5)"
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
            default="output.json",
            help="JSON output file (default: 'output.json')"
        )

    # MSD-specific arguments
    # parser()
    msd.add_argument(
    "--plot-data",
    action="append", nargs="+",
    default=None,          # <-- change from hard-coded list to None
    help="Tuples T, element, dir, type to plot. If omitted, plot everything."
    )

    msd.add_argument(
        "--first-time",
        nargs="+", type=float,
        default=[0],
        help="First time(s) for MSD plot (default: [0])"
    )
    msd.add_argument(
        "--last-time",
        nargs="+", type=float,
        default=[10],
        help="Last time(s) for MSD plot (default: [10])"
    )
    msd.add_argument(
        "--save-path",
        default="msd_plot.png",
        help="File path to save MSD plot (default: 'msd_plot.png')"
    )

    # Van Hove-specific arguments
    # parser()
    vh.add_argument(
    "--plot-data",
    action="append", nargs="+",
    default=None,          # <-- change from hard-coded list to None
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
        help="Directory containing .pos and .in files"
    )
    ionic_density.add_argument(
        "--formula", default="Li22Al2P4S24",
        help="Chemical formula of the system (default: Li22Al2P4S24)"
    )
    ionic_density.add_argument(
        "--rigid-lattice", nargs="+", default=["Al", "P", "S"],
        help="Elements considered as rigid lattice (default: ['Al', 'P', 'S'])"
    )
    ionic_density.add_argument(
        "--output-li", default="li_density.xsf",
        help="Output file for Li density (default: li_density.xsf)"
    )
    ionic_density.add_argument(
        "--output-rigid", default="rigid_density.xsf", 
        help="Output file for rigid lattice density (default: rigid_density.xsf)"
    )
    ionic_density.add_argument(
        "--conv-factor", type=float, default=1.0,
        help="Conversion factor for coordinates (default: 1.0)"
    )
    ionic_density.add_argument(
        "--cell-a", type=float, default=16.4294,
        help="Cell parameter a in bohr (default: 16.4294)"
    )
    ionic_density.add_argument(
        "--cell-b", type=float, 
        help="Cell parameter b in bohr (default: same as a)"
    )
    ionic_density.add_argument(
        "--cell-c-factor", type=float, default=1.44919,
        help="Factor for cell parameter c (c = a * factor) (default: 1.44919)"
    )

    # ... rest of existing parser code
    return p.parse_args()


def main():
    a = parser()

    if a.mode in ("msd", "vh"):
        tout = Job(a.temperature, a.diffusing_elements, a.diffusivity_direction_choices,
               a.diffusivity_choices, a.correlation, a.data_dir, a.Conv_factor,
               a.initial_time, a.final_time, a.initial_slope_time,
               a.final_slope_time, a.block, a.rmax, a.step_skip, a.sigma,
               a.ngrid)

        if a.json_output:
            with open(a.json_output, "w") as f:
                json.dump(serializable.convert_to_serializable(tout), f, indent=2)
            data_source = a.json_output
        else:
            data_source = tout

    if a.mode == "msd":
        if a.plot_data is None:        # ← if user gave nothing, build it
            pdata = []
            for (T, ele), blob in tout.items():
                for key in blob["msd_data"][ele]:
                    # key looks like 'Tracer_msd_array_XY' etc.
                    typ, _, *suf = key.split('_')
                    dir_label = suf[-1] if suf else "XYZ"
                    pdata.append([T, ele, dir_label, typ])
        else:
            pdata = [[float(x[0]), x[1], x[2], x[3]] for x in a.plot_data]
        p.msd_plot(data_source, pdata, a.first_time, a.last_time, a.save_path)

    elif a.mode == "vh":
        # ------------------------------------------- build the list of curves
        # If the user didn’t give --plot-data we auto-discover every (T, ele, h)
        # pair that was actually computed.  h is “Self” or “Distinct”.
        if a.plot_data is None:
            pdata = []
            for (T, ele), blob in tout.items():
                for corr_type in blob["evaluated_corr"]:
                    pdata.append([T, ele, corr_type])
        else:
            # Expect triples:  600  Li  Self          (all strings on CLI)
            pdata = [[float(x[0]), x[1], x[2]] for x in a.plot_data]

        # ------------------------------------------- draw the plot
        p.van_hove_plot(
            data_source,              # JSON file or in-memory dict
            pdata,                    # [[T, element, “Self”/“Distinct”], ...]
            save_path=a.save_path,         # --save-path
            figsize=tuple(a.figsize)  # --figsize 10 8   →  (10.0, 8.0)
        )

    elif a.mode == "ionic-density":
        from ase import Atoms
        from samos.trajectory import Trajectory

        pos_files = sorted(glob.glob(os.path.join(a.data_dir, "*.pos")))
        ion_files = sorted(glob.glob(os.path.join(a.data_dir, "*.in")))
    
        if not pos_files or not ion_files:
            sys.exit("Missing .pos or .in files in data directory")
        if len(pos_files) != len(ion_files):
            sys.exit("Number of .pos and .in files must match")
    
        results = {}
        for i, (pos_file, ion_file) in enumerate(zip(pos_files, ion_files)):
            print(f"\nProcessing file pair {i+1}: {pos_file}, {ion_file}")
        
            # Generate unique output filenames
            base_name = os.path.splitext(os.path.basename(pos_file))[0]
            output_li = f"{base_name}_{a.output_li}"
            output_rigid = f"{base_name}_{a.output_rigid}"
        
            #    Replicate logic from probability_density.py's main()
            # Set up cell parameters
            cell_a = a.cell_a * a.conv_factor * prob.bohr_to_ang
            cell_b = a.cell_b * a.conv_factor * prob.bohr_to_ang if a.cell_b is not None else cell_a
            cell_c = a.cell_a * a.cell_c_factor * a.conv_factor * prob.bohr_to_ang
            simulation_cell = [[cell_a, 0, 0], [0, cell_b, 0], [0, 0, cell_c]]
        
            # Load configurations and trajectory
            nat, start_positions = prob.load_start_configuration_from_qe_file(ion_file)
        
            # Load trajectory (pass nat explicitly)
            positionsArr, timestep = prob.load_trajectory_from_cp_file(pos_file, nat=nat, format='bohr')

            # Initialize atoms and trajectory
            atoms = Atoms(a.formula)
            atoms.set_positions(start_positions)
            atoms.cell = np.array(simulation_cell)
            t = Trajectory()
            t.set_timestep(timestep)
            t.set_atoms(atoms)
            t.set_positions(np.array(positionsArr))
        
            # Recenter trajectory
            t.recenter(a.rigid_lattice, mode='geometric')
            
            # Evaluate center of mass
            com = prob.evaluate_com(start_positions, t, mode='geometric')
            
            # Shift positions
            pos = t.get_positions()
            nstep, nat, ncoord = pos.shape
            for i in range(nstep):
                for j in range(nat):
                    pos[i, j, :] = pos[i, j, :] + com
            t.set_positions(np.array(pos))
        
            # Get indices for Li and rigid lattice
            indices_li = t.get_indices_of_species('Li', start=1)
            indices_rigid = np.array(
                list(t.get_indices_of_species('Al', start=1)) +
                list(t.get_indices_of_species('P', start=1)) +
                list(t.get_indices_of_species('S', start=1))
            )
        
            # Generate density files
            prob.get_gaussian_density(t, element='Li', outputfile=output_li, indices_i_care=indices_li)
            prob.get_gaussian_density(t, element=None, outputfile=output_rigid, indices_i_care=indices_rigid)
            
            results[base_name] = {"li_density_file": output_li, "rigid_density_file": output_rigid}
        
        print("\nIonic density analysis completed!")
        print("Results:", results)
        return


if __name__ == "__main__":
    main()
