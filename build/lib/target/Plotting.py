#!/usr/bin/env python3
"""
Plotting.py

A command-line tool with subcommands to create MSD or Van Hove plots
from JSON data using the msd_plot() and van_hove_plot() functions.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings


from . import data_processing as dp

# def msd_plot(data_path, Plot_data, First_time, Last_time, save_path=None):
#     """
#     Plots Mean Squared Displacement (MSD) vs Time for specified particles and time ranges.
#     See your existing docstring...
#     """
#     with open(data_path, 'r') as file:
#         data = json.load(file)
#     fig, ax = plt.subplots()
    
#     # Find common particles in Plot_data entries
#     common_parts_set = set(Plot_data[0])
#     for entry in Plot_data[1:]:
#         common_parts_set.intersection_update(set(entry))
#     common_parts = list(common_parts_set)
#     common_parts_str = ', '.join(map(str, common_parts))

#     msd_sem_array = []
#     # No direct usage of the range i in your snippet, but you keep them for indexing.
#     for i in range(len(First_time)):
#         frst = First_time[i]
#         lst = Last_time[i]
#         # For each range, we handle the plotting inside the next loop

#     # Now loop over the instructions
#     for info in Plot_data:
#         # info might look like [600, 'Li', 'msd_tracer', 'XYZ']
#         # Adjust if your format is different
#         if info[3] == "XYZ":
#             suffix = ""
#         elif info[3] == "XY":
#             suffix = "_XY"
#         elif info[3] == "YZ":
#             suffix = "_YZ"
#         elif info[3] == "ZX":
#             suffix = "_ZX"
#         elif info[3] in ["X","Y","Z"]:
#             suffix = f"_{info[3]}"
#         else:
#             suffix = ""

#         Temp = f"({info[0]}, '{info[1]}')"
#         Element = str(info[1])
#         dt = data[Temp]['dt_dict']

#         # We'll assume just the first range is used; 
#         # or if you want multiple, you'd iterate them
#         frst = First_time[0]
#         lst = Last_time[0]
#         First, Last = dp.find_terms(dt, frst, lst)
        
#         msd_array_key = f"{info[2]}_msd_array{suffix}"
#         diffusivity_key = f"{info[2]}_diffusivity{suffix}"
#         slope_sem_key = f"{info[2]}_slope_sem{suffix}"
#         diffusivity_error_key = f"{info[2]}_diffusivity_error{suffix}"

#         msd_sem = np.array([
#             data[Temp]['msd_data'][Element][slope_sem_key][0]*x
#             for x in dt
#         ])
#         msd_sem_array = np.array(msd_sem)

#         # Create a label from the "uncommon" parts
#         uncommon_parts = [
#             str(part) for part in info
#             if part not in common_parts_set and part not in ['', []]
#         ]
#         Label_marker = (
#             f"{', '.join(uncommon_parts)}: Diffusivity: "
#             f"{data[Temp]['msd_data'][Element][diffusivity_key][0]:.4e} ± "
#             f"{data[Temp]['msd_data'][Element][diffusivity_error_key][0]:.4e} cm²/s"
#         )

#         # Plot
#         ax.plot(
#             dt[First:Last],
#             data[Temp]['msd_data'][Element][msd_array_key][0][First:Last],
#             label=Label_marker
#         )
#         ax.fill_between(
#             dt[First:Last],
#             np.array(data[Temp]['msd_data'][Element][msd_array_key][0][First:Last]) - msd_sem_array[First:Last],
#             np.array(data[Temp]['msd_data'][Element][msd_array_key][0][First:Last]) + msd_sem_array[First:Last],
#             alpha=0.2, linewidth=1
#         )

#     ax.grid(False)
#     ax.set_xlabel('Time [ps]', fontweight='bold')
#     ax.set_ylabel('Mean Squared Displacement (MSD) [Å²]', fontweight='bold')
#     ax.set_title(f'MSD vs Time ({common_parts_str})', fontweight='bold')

#     ax.xaxis.set_major_locator(ticker.AutoLocator())
#     ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#     ax.yaxis.set_major_locator(ticker.AutoLocator())
#     ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

#     ax.legend()

#     if save_path:
#         plt.savefig(save_path, format='jpeg', dpi=300)
#     plt.show()

def msd_plot(data, pdata, ft, lt, save=None):
    
    d = data if isinstance(data, dict) else json.load(open(data))
    fig, ax = plt.subplots()

    # Define distinct colors for 7 directions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2']
    
    color_map = {
        'XYZ': colors[0], 'XY': colors[1], 'YZ': colors[2], 
        'ZX': colors[3], 'X': colors[4], 'Y': colors[5], 'Z': colors[6]
    }

    for T, ele, dim, lab in pdata:
        suf = "" if dim == "XYZ" else f"_{dim}"
        k = f"({T}, '{ele}')"

        if k not in d or ele not in d[k]["msd_data"]:
            continue

        # Extract data (same as before)
        mkey = f"{lab}_msd_array{suf}"
        tkey = f"{lab}_time_array{suf}"
        dkey = f"{lab}_diffusivity{suf}"
        skey = f"{lab}_slope_sem{suf}"
        ekey = f"{lab}_diffusivity_error{suf}"

        try:
            msd = d[k]["msd_data"][ele][mkey][0]
            t = d[k]["msd_data"][ele][tkey][0]
        except KeyError:
            continue

        from . import data_processing as dp
        F, L = dp.find_terms(t, ft[0], lt[0])
        sem = np.full_like(t, d[k]["msd_data"][ele][skey][0])

        # Use specific color for this direction
        line_color = color_map.get(dim, colors[0])
        
        ax.plot(t[F:L], msd[F:L], linewidth=2, color=line_color,
                label=(f"{dim}: D: "
                       f"{d[k]['msd_data'][ele][dkey][0]:.2e} ± "
                       f"{d[k]['msd_data'][ele][ekey][0]:.2e} cm²/s"))
        
        ax.fill_between(t[F:L], msd[F:L]-sem[F:L], msd[F:L]+sem[F:L], 
                       alpha=0.2, color=line_color)
    
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xlabel("Time (ps)", fontsize=14)
    ax.set_ylabel("MSD [Å²]", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='best') 
    
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=500, bbox_inches='tight')
    plt.show()


def van_hove_plot(data_path, Plot_data, save_path=None, figsize=(10, 8)):
    """
    Plots the Van Hove function (Self or Distinct) from a JSON file.
    See your existing docstring...
    """
    with open(data_path, 'r') as file:
        data = json.load(file)

    for temp, element, h_type in Plot_data:
        if h_type == "Self":
            vmax = 1.0
            cb_label = r"$4\pi r^2 G_s(t, r)$"
            cb_ticks = [0, 1]
        elif h_type == "Distinct":
            vmax = 4.0
            cb_label = r"$G_d(t, r)$"
            cb_ticks = [0, 1, 2, 3, 4]
        else:
            # Handle other possible types?
            vmax = np.max(np.array(1.0))
            cb_label = "VanHove"
            cb_ticks = None

        # data structure
        data_key = f"({temp}, '{element}')"
        van_hove = data[data_key]["evaluated_corr"][h_type]["grt"]

        dist_array = data[data_key]["evaluated_corr"][h_type]["dist_interval"]
        y = np.arange(np.shape(van_hove)[1]) * dist_array[-1] / float(len(dist_array)-1)

        dt_array = data[data_key]['dt_dict']
        reduced_nt = data[data_key]["evaluated_corr"][h_type]["reduced_nt"]
        x = np.linspace(dt_array[0], dt_array[-1], reduced_nt)

        X, Y = np.meshgrid(x, y, indexing="ij")
        ticksize = int(figsize[0] * 2.5)
        labelsize = int(figsize[0] * 3)

        plt.figure(figsize=figsize, facecolor="w")
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.pcolor(X, Y, van_hove, cmap="jet", vmin=np.min(van_hove), vmax=vmax)
        plt.xlabel("Time (ps)", size=labelsize)
        plt.ylabel(r"$r$ ($\AA$)", size=labelsize)
        plt.axis([x.min(), x.max(), y.min(), y.max()])

        cbar = plt.colorbar(ticks=cb_ticks) if cb_ticks else plt.colorbar()
        cbar.set_label(label=cb_label, size=labelsize)
        cbar.ax.tick_params(labelsize=ticksize)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format='jpeg', dpi=300)
        plt.show()

def parse_cli_args():
    """
    Creates a subcommand-based CLI for msd_plot vs van_hove_plot.
    """
    parser = argparse.ArgumentParser(
        description="Make MSD or Van Hove plots from JSON data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Subcommand: 'msd' or 'van_hove'")

    # --------------- MSD Subcommand ---------------
    msd_parser = subparsers.add_parser("msd", help="Generate MSD plots from JSON data.")
    msd_parser.add_argument("--data-path", required=True,
                            help="Path to the JSON file with simulation data.")
    msd_parser.add_argument("--save-path", default=None,
                            help="If provided, the plot is saved to this file (JPEG).")
    # We assume Plot_data is a repeated argument for each set. For example:
    #   --plot-data 600 Li msd_tracer XYZ
    msd_parser.add_argument("--plot-data", action="append", nargs="+", required=True,
                            help="Plot instructions. E.g: --plot-data 600 Li msd_tracer XYZ")
    
    # For your msd_plot, you also had First_time & Last_time arrays
    msd_parser.add_argument("--first-time", nargs="+", type=float, default=[0],
                            help="List of start times for each time window.")
    msd_parser.add_argument("--last-time", nargs="+", type=float, default=[10],
                            help="List of end times for each time window.")

    # --------------- Van Hove Subcommand ---------------
    vh_parser = subparsers.add_parser("van_hove", help="Generate Van Hove plots from JSON data.")
    vh_parser.add_argument("--data-path", required=True,
                           help="Path to the JSON file with simulation data.")
    vh_parser.add_argument("--save-path", default=None,
                           help="If provided, the plot is saved to this file (JPEG).")
    vh_parser.add_argument("--figsize", nargs=2, type=float, default=[10,8],
                           help="Figure size (width height). Default=10 8")
    
    # We'll let the user specify sets of [temp element h_type], e.g.:
    #   --plot-data 600 Li Self
    vh_parser.add_argument("--plot-data", action="append", nargs="+", required=True,
                           help="Van Hove instructions. E.g: --plot-data 600 Li Self")

    return parser.parse_args()

def main():
    args = parse_cli_args()
    
    if args.command == "msd":
        # Convert each sub-list in --plot-data to the correct format
        # Example: ["600","Li","msd_tracer","XYZ"] => [600, 'Li', 'msd_tracer', 'XYZ']
        # Need to handle type conversions if needed
        final_plot_data = []
        for item in args.plot_data:  # item is e.g. ["600","Li","msd_tracer","XYZ"]
            # item[0] -> temperature as int or float
            # item[1] -> element as string
            # item[2] -> e.g. "msd_tracer"
            # item[3] -> e.g. "XYZ"
            converted = [int_or_float(item[0]), item[1], item[2], item[3]]
            final_plot_data.append(converted)

        # Now call msd_plot
        msd_plot(
            data_path=args.data_path,
            Plot_data=final_plot_data,
            First_time=args.first_time,
            Last_time=args.last_time,
            save_path=args.save_path
        )

    elif args.command == "van_hove":
        # We'll do similarly
        final_plot_data = []
        for item in args.plot_data:
            # item might be ["600","Li","Self"]
            # convert item[0] to float or int
            # item[1] is element
            # item[2] is "Self" or "Distinct"
            converted = [int_or_float(item[0]), item[1], item[2]]
            final_plot_data.append(converted)

        van_hove_plot(
            data_path=args.data_path,
            Plot_data=final_plot_data,
            save_path=args.save_path,
            figsize=tuple(args.figsize)
        )

def int_or_float(val):
    """Helper: try int first, then float if needed."""
    try:
        return int(val)
    except ValueError:
        return float(val)

if __name__ == "__main__":
    main()
