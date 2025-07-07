"""
Plotting and Visualization Module for CPDyAna
=============================================

This module provides comprehensive plotting functions for visualizing molecular
dynamics analysis results. It creates publication-quality figures for various
analysis types including MSD plots, Van Hove correlation functions, and other
trajectory analysis visualizations.

The module uses matplotlib for plotting with customizable styling, automatic
legend generation, and support for multiple data series. All plots include
proper axis labels, units, and can be saved in various formats.

Key Features:
- Mean Square Displacement (MSD) plots with linear fits
- Van Hove correlation function visualizations
- Multi-element and multi-temperature comparisons
- Automatic color schemes and styling
- Publication-ready output formats

Functions:
    msd_plot: Create MSD vs time plots with diffusion coefficient fitting
    van_hove_plot: Visualize Van Hove correlation functions
    
Author: CPDyAna Development Team
Version: 01-02-2024
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import warnings
from copy import deepcopy
from itertools import cycle
from . import data_processing as dp
from . import data_processing_lammps as dpl

import os

# def msd_plot_lammps(data_path, Plot_data, First_time, Last_time, save_path=None):
#     with open(data_path, 'r') as file:
#         data = json.load(file)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     common_parts_set = set(Plot_data[0])
#     for entry in Plot_data[1:]:
#         common_parts_set.intersection_update(set(entry))
#     common_parts = list(common_parts_set)

#     for info in Plot_data:
#         suffix = '' if info[3] == "XYZ" else f"_{info[3]}"
#         Temp = f"({info[0]}, '{info[1]}')"
#         Element = str(info[1])
#         msd_array_key = f"{info[2]}_msd_array{suffix}"
#         time_array_key = f"{info[2]}_time_array{suffix}"
#         diffusivity_key = f"{info[2]}_diffusivity{suffix}"
#         diffusivity_error_key = f"{info[2]}_diffusivity_error{suffix}"
#         slope_sem_key = f"{info[2]}_slope_sem{suffix}"

#         msd_blob = data[Temp]['msd_data'][Element]

#         msd_mean = np.array(msd_blob[msd_array_key][0])
#         time_array = np.array(msd_blob[time_array_key][0])

#         min_length = min(len(time_array), len(msd_mean))
#         if len(time_array) != len(msd_mean):
#             print(f"Info: Array length adjustment for {Element} {info[3]}. Time: {len(time_array)}, MSD: {len(msd_mean)}")
#             print(f"Using first {min_length} points from both arrays")
#             time_array = time_array[:min_length]
#             msd_mean = msd_mean[:min_length]

#         if np.all(np.isnan(msd_mean)):
#             print(f"Warning: All MSD values are NaN for {Element} {info[3]}. Skipping plot.")
#             continue

#         valid_mask = (~np.isnan(msd_mean)) & (~np.isnan(time_array)) & (np.isfinite(msd_mean)) & (np.isfinite(time_array))
#         if not np.any(valid_mask):
#             print(f"Warning: No valid data points for {Element} {info[3]}. Skipping plot.")
#             continue

#         time_array = time_array[valid_mask]
#         msd_mean = msd_mean[valid_mask]

#         if len(time_array) == 0:
#             print(f"Warning: No data points for {Element} {info[3]} after filtering. Skipping plot.")
#             continue

#         print(f"Plotting {Element} {info[3]}: time range {time_array[0]:.3f}-{time_array[-1]:.3f} ps, "
#               f"MSD range {msd_mean[0]:.3f}-{msd_mean[-1]:.3f} Ų")

#         D = msd_blob[diffusivity_key][0] if diffusivity_key in msd_blob else np.nan
#         D_err = msd_blob[diffusivity_error_key][0] if diffusivity_error_key in msd_blob else np.nan
#         slope_sem = msd_blob[slope_sem_key][0] if slope_sem_key in msd_blob else np.nan

#         First = np.searchsorted(time_array, First_time, side='left')
#         Last = np.searchsorted(time_array, Last_time, side='right') - 1

#         if First >= len(time_array) or Last < 0 or First > Last:
#             print(f"Warning: Time window {First_time}-{Last_time} ps is outside data range for {Element} {info[3]}")
#             print(f"Data range: {time_array[0]:.3f}-{time_array[-1]:.3f} ps")
#             plot_times = time_array
#             plot_msd = msd_mean
#         else:
#             plot_times = time_array[First:Last+1]
#             plot_msd = msd_mean[First:Last+1]

#         # Use block error for shading (like QE)
#         if not np.isnan(slope_sem):
#             msd_sem = slope_sem * plot_times
#         else:
#             msd_sem = np.zeros_like(plot_times)

#         if np.isnan(D_err) or np.isnan(D):
#             Label_marker = f"{info[3]}: {Element} (D: calculation failed)"
#         else:
#             Label_marker = f"{info[3]}: {Element} D: {D:.2e} ± {D_err:.2e} cm²/s"

#         uncommon_parts = [str(part) for part in info if part not in common_parts_set and part not in ['', []]]
#         if uncommon_parts:
#             Label_marker = f"{', '.join(uncommon_parts)}: {Label_marker}"

#         ax.plot(plot_times, plot_msd, label=Label_marker, linewidth=2)
#         if np.any(msd_sem > 0):
#             ax.fill_between(plot_times, plot_msd - msd_sem, plot_msd + msd_sem, alpha=0.2)

#     ax.grid(True, alpha=0.3)
#     ax.set_xlabel('Time (ps)', fontsize=14)
#     ax.set_ylabel(r'MSD [$\mathrm{\AA^2}$]', fontsize=14)
#     ax.set_title('Mean Square Displacement', fontsize=16)
#     ax.xaxis.set_major_locator(ticker.AutoLocator())
#     ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#     ax.yaxis.set_major_locator(ticker.AutoLocator())
#     ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#     ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight')
#         print(f"MSD plot saved as {save_path}")
#     plt.show()
#     plt.close()

def msd_plot_lammps(data_path, Plot_data, First_time, Last_time, save_path=None):
    with open(data_path, 'r') as file:
        data = json.load(file)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
    sns.set_context("notebook", font_scale=1.2)
    color_palette = sns.color_palette("muted", len(Plot_data))
    
    common_parts_set = set(Plot_data[0])
    for entry in Plot_data[1:]:
        common_parts_set.intersection_update(set(entry))
    
    for idx, info in enumerate(Plot_data):
        suffix = '' if info[3] == "XYZ" else f"_{info[3]}"
        Temp = f"({info[0]}, '{info[1]}')"
        Element = str(info[1])
        msd_array_key = f"{info[2]}_msd_array{suffix}"
        time_array_key = f"{info[2]}_time_array{suffix}"
        diffusivity_key = f"{info[2]}_diffusivity{suffix}"
        diffusivity_error_key = f"{info[2]}_diffusivity_error{suffix}"
        slope_sem_key = f"{info[2]}_slope_sem{suffix}"
        
        msd_blob = data[Temp]['msd_data'][Element]
        time_array = np.array(msd_blob[time_array_key][0])
        msd_mean = np.array(msd_blob[msd_array_key][0])
        
        min_length = min(len(time_array), len(msd_mean))
        if len(time_array) != len(msd_mean):
            print(f"Info: Array length adjustment for {Element} {info[3]}. Time: {len(time_array)}, MSD: {len(msd_mean)}")
            time_array = time_array[:min_length]
            msd_mean = msd_mean[:min_length]
        
        if np.all(np.isnan(msd_mean)):
            print(f"Warning: All MSD values are NaN for {Element} {info[3]}. Skipping plot.")
            continue
        
        valid_mask = (~np.isnan(msd_mean)) & (~np.isnan(time_array)) & (np.isfinite(msd_mean)) & (np.isfinite(time_array))
        if not np.any(valid_mask):
            print(f"Warning: No valid data points for {Element} {info[3]}. Skipping plot.")
            continue
        
        time_array = time_array[valid_mask]
        msd_mean = msd_mean[valid_mask]
        
        if len(time_array) == 0:
            print(f"Warning: No data points for {Element} {info[3]} after filtering. Skipping plot.")
            continue
        
        D = msd_blob[diffusivity_key][0] if diffusivity_key in msd_blob else np.nan
        D_err = msd_blob[diffusivity_error_key][0] if diffusivity_error_key in msd_blob else np.nan
        slope_sem = msd_blob[slope_sem_key][0] if slope_sem_key in msd_blob else np.nan
        
        First = np.searchsorted(time_array, First_time, side='left')
        Last = np.searchsorted(time_array, Last_time, side='right') - 1
        
        if First >= len(time_array) or Last < 0 or First > Last:
            print(f"Warning: Time window {First_time}-{Last_time} ps is outside data range for {Element} {info[3]}")
            plot_times = time_array
            plot_msd = msd_mean
        else:
            plot_times = time_array[First:Last+1]
            plot_msd = msd_mean[First:Last+1]
        
        msd_sem = slope_sem * plot_times if not np.isnan(slope_sem) else np.zeros_like(plot_times)
        label = f"{Element} {info[3]}: D={D:.2e} ± {D_err:.2e} cm²/s" if not np.isnan(D) else f"{Element} {info[3]}: D calculation failed"
        
        sns.lineplot(x=plot_times, y=plot_msd, ax=ax, color=color_palette[idx], label=label, linewidth=2)
        if np.any(msd_sem > 0):
            ax.fill_between(plot_times, plot_msd - msd_sem, plot_msd + msd_sem, color=color_palette[idx], alpha=0.2)
    
    ax.set_xlabel('Time (ps)', fontsize=14)
    ax.set_ylabel(r'MSD [$\mathrm{\AA^2}$]', fontsize=14)
    ax.set_title('Mean Square Displacement (LAMMPS)', fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    sns.despine(ax=ax, trim=True)
    if save_path:
        plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# def msd_plot_qe(data_path, Plot_data, First_time, Last_time, save_path=None):
#     import json
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as ticker

#     with open(data_path, 'r') as file:
#         data = json.load(file)
#     fig, ax = plt.subplots()
#     common_parts_set = set(Plot_data[0])
#     for entry in Plot_data[1:]:
#         common_parts_set.intersection_update(set(entry))
#     common_parts = list(common_parts_set)

#     for info in Plot_data:
#         suffix = '' if info[3] == "XYZ" else f"_{info[3]}"
#         Temp = f"({info[0]}, '{info[1]}')"
#         Element = str(info[1])
#         dt = data[Temp]['dt_dict']
#         # Find indices for time window
#         First = np.searchsorted(dt, First_time, side='left')
#         Last = np.searchsorted(dt, Last_time, side='right') - 1
#         msd_array_key = f"{info[2]}_msd_array{suffix}"
#         diffusivity_key = f"{info[2]}_diffusivity{suffix}"
#         slope_sem_key = f"{info[2]}_slope_sem{suffix}"
#         diffusivity_error_key = f"{info[2]}_diffusivity_error{suffix}"

#         msd_sem = np.array([data[Temp]['msd_data'][Element][slope_sem_key][0] * x for x in dt])
#         uncommon_parts = [str(part) for part in info if part not in common_parts_set and part not in ['', []]]
#         Label_marker = (f"{', '.join(uncommon_parts)}: D: "
#                         f"{data[Temp]['msd_data'][Element][diffusivity_key][0]:.2e} ± "
#                         f"{data[Temp]['msd_data'][Element][diffusivity_error_key][0]:.2e} cm²/s")

#         ax.plot(dt[First:Last+1], data[Temp]['msd_data'][Element][msd_array_key][0][First:Last+1], label=Label_marker)
#         ax.fill_between(dt[First:Last+1],
#                         np.array(data[Temp]['msd_data'][Element][msd_array_key][0][First:Last+1]) - msd_sem[First:Last+1],
#                         np.array(data[Temp]['msd_data'][Element][msd_array_key][0][First:Last+1]) + msd_sem[First:Last+1],
#                         alpha=0.2)

#     ax.grid(False)
#     ax.set_xlabel('Time (ps)', fontsize=14)
#     ax.set_ylabel('MSD [$\mathrm{\AA^2}$]', fontsize=14)
#     ax.xaxis.set_major_locator(ticker.AutoLocator())
#     ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#     ax.yaxis.set_major_locator(ticker.AutoLocator())
#     ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#     ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
#     if save_path:
#         plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()

def msd_plot_qe(data_path, Plot_data, First_time, Last_time, save_path=None):
    with open(data_path, 'r') as file:
        data = json.load(file)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
    sns.set_context("notebook", font_scale=1.2)
    color_palette = sns.color_palette("muted", len(Plot_data))
    
    common_parts_set = set(Plot_data[0])
    for entry in Plot_data[1:]:
        common_parts_set.intersection_update(set(entry))
    
    for idx, info in enumerate(Plot_data):
        suffix = '' if info[3] == "XYZ" else f"_{info[3]}"
        Temp = f"({info[0]}, '{info[1]}')"
        Element = str(info[1])
        dt = data[Temp]['dt_dict']
        First = np.searchsorted(dt, First_time, side='left')
        Last = np.searchsorted(dt, Last_time, side='right') - 1
        msd_array_key = f"{info[2]}_msd_array{suffix}"
        diffusivity_key = f"{info[2]}_diffusivity{suffix}"
        slope_sem_key = f"{info[2]}_slope_sem{suffix}"
        diffusivity_error_key = f"{info[2]}_diffusivity_error{suffix}"
        
        msd_mean = np.array(data[Temp]['msd_data'][Element][msd_array_key][0])
        msd_sem = np.array([data[Temp]['msd_data'][Element][slope_sem_key][0] * x for x in dt])
        D = data[Temp]['msd_data'][Element][diffusivity_key][0]
        D_err = data[Temp]['msd_data'][Element][diffusivity_error_key][0]
        
        uncommon_parts = [str(part) for part in info if part not in common_parts_set and part not in ['', []]]
        label = f"{', '.join(uncommon_parts)}: D={D:.2e} ± {D_err:.2e} cm²/s"
        
        sns.lineplot(x=dt[First:Last+1], y=msd_mean[First:Last+1], ax=ax, color=color_palette[idx], label=label, linewidth=2)
        ax.fill_between(dt[First:Last+1], msd_mean[First:Last+1] - msd_sem[First:Last+1], 
                        msd_mean[First:Last+1] + msd_sem[First:Last+1], color=color_palette[idx], alpha=0.2)
    
    ax.set_xlabel('Time (ps)', fontsize=14)
    ax.set_ylabel(r'MSD [$\mathrm{\AA^2}$]', fontsize=14)
    ax.set_title('Mean Square Displacement (QE)', fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    sns.despine(ax=ax, trim=True)
    if save_path:
        plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def msd_plot(data_path, Plot_data, First_time, Last_time, save_path=None, is_lammps=False):
    if is_lammps:
        msd_plot_lammps(data_path, Plot_data, First_time, Last_time, save_path)
    else:
        msd_plot_qe(data_path, Plot_data, First_time, Last_time, save_path)
    
def centers_to_edges(centers):
    """
    Convert bin centers to bin edges for pcolormesh plotting.
    """
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges

def van_hove_plot(data_path, Plot_data, save_path=None, figsize=(10, 8)):

    with open(data_path, 'r') as file:
        data = json.load(file)

    for temp, element, h_type in Plot_data:
        if h_type == "Self":
            vmax = 1.0
            cb_label = r"$4\pi r^2 G_s(t, r)$"
        elif h_type == "Distinct":
            vmax = 4.0
            cb_label = r"$G_d(t, r)$"
        else:
            vmax = np.max(np.array(1.0))
            cb_label = "VanHove"

        data_key = f"({temp}, '{element}')"
        van_hove = np.array(data[data_key]["evaluated_corr"][h_type]["grt"])
        dist_array = np.array(data[data_key]["evaluated_corr"][h_type]["dist_interval"])
        dt_array = np.array(data[data_key]['dt_dict'])
        reduced_nt = data[data_key]["evaluated_corr"][h_type]["reduced_nt"]

        # Defensive: check shapes
        if van_hove.ndim != 2:
            print(f"Warning: Van Hove array is not 2D for {element} {h_type}")
            continue
        if van_hove.shape[1] != len(dist_array):
            print(f"Warning: Van Hove shape {van_hove.shape} does not match dist_interval length {len(dist_array)}")
            continue

        # Build time axis: use reduced_nt and dt_array
        # dt_array is the time difference between frames, so cumulative sum gives time
        # But for LAMMPS, dt_array is all 1.0, so just use np.arange(reduced_nt)
        x = np.arange(reduced_nt)
        # If you want time in ps, multiply by step_skip * dt (dt=1.0 for LAMMPS)
        x = x * 10  # step_skip=10 in your run, dt=1.0
        y = dist_array
        
        x_edges = centers_to_edges(x)
        y_edges = centers_to_edges(y)

        plt.figure(figsize=figsize, facecolor="w")
        # pcolormesh expects (len(x), len(y)) for Z
        plt.pcolormesh(x_edges, y_edges, van_hove.T, cmap="jet", shading="auto", vmin=np.min(van_hove), vmax=vmax)
        plt.xlabel("Time (ps)", fontsize=14)
        plt.ylabel(r"$r$ ($\AA$)", fontsize=14)
        plt.colorbar(label=cb_label)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='jpeg', dpi=300)
        plt.show()

def ngp_plot(data_source, plot_data, first_time, last_time, save_path="NGP.jpg"):
    """
    Generate Non-Gaussian Parameter (NGP) plots for specified data.
    
    Args:
        data_source (dict or str): Dictionary with NGP data or path to JSON file.
        plot_data (list): List of [temperature, element, direction] to plot.
        first_time (float): Start time for plot display in picoseconds.
        last_time (float): End time for plot display in picoseconds.
        save_path (str): File path to save the plot.
    """
    # Load data if a file path is provided
    if isinstance(data_source, str):
        with open(data_source, 'r') as file:
            data = json.load(file)
    else:
        data = data_source
    
    plt.figure(figsize=(10, 8))
    
    for temp, ele, direction in plot_data:
        try:
            # Construct the key as a string tuple
            key = str((temp, ele))  # e.g., "(600.0, 'Li')"
            
            # Determine the suffix based on direction
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            ngp_key = f'NGP_array{suffix}'
            time_lags_key = f'time_lags{suffix}'
            
            # Access the NGP data and time lags (first element of the list)
            time_lags = np.array(data[key]['ngp_data'][ele][time_lags_key][0])
            ngp_values = np.array(data[key]['ngp_data'][ele][ngp_key][0])
            
            # Filter data within the specified time range
            mask = (time_lags >= first_time) & (time_lags <= last_time)
            if np.any(mask):
                plt.plot(time_lags[mask], ngp_values[mask], label=f'T={temp}K, {ele}, {direction}')
            else:
                print(f"Warning: No data within time range {first_time}-{last_time} ps for T={temp}, {ele}, {direction}")
        except (KeyError, TypeError, IndexError) as e:
            print(f"Warning: Could not plot NGP for T={temp}, {ele}, {direction}: {e}")
            # Optional: Print available keys for debugging
            if key in data and 'ngp_data' in data[key] and ele in data[key]['ngp_data']:
                print(f"Available keys: {list(data[key]['ngp_data'][ele].keys())}")
            continue
    
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('Time Lag (ps)', fontsize=12)
    plt.ylabel('Non-Gaussian Parameter, α₂(t)', fontsize=12)
    plt.title('Non-Gaussian Parameter vs Time Lag', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(-0.5, 2.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_vaf_isotropic(data_obj, axis=None, hide_legend=False, target_species=None, display=False, **options):
    """
    Plot isotropic velocity autocorrelation function (VAF) and its time integral with enhanced Seaborn styling.

    Parameters
    ----------
    data_obj : object
        Data object containing VAF arrays and attributes.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on.
    hide_legend : bool, optional
        If True, suppress legend display.
    target_species : list, optional
        List of species to plot.
    display : bool, optional
        If True, call plt.show().
    """
    if axis is None:
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    # Apply sophisticated Seaborn styling
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
    sns.set_palette("deep")  # Richer color palette
    
    properties = data_obj.get_attrs()
    num_traj = properties['nr_of_trajectories']
    fit_start = properties['t_start_fit_dt']
    fit_end = properties['t_end_fit_dt']
    time_step = properties.get('stepsize_t', 1)
    fs_per_step = properties['timestep_fs']
    
    time_points = fs_per_step * time_step * np.arange(
        properties.get('t_start_dt') / time_step,
        properties.get('t_end_dt') / time_step
    )
    if target_species is None:
        target_species = properties['species_of_interest']
    
    axis.set_ylabel(r'VAF [$\mathrm{\AA}^2\,\mathrm{fs}^{-2}$]', fontsize=16, weight='bold')
    axis.set_xlabel('Time [fs]', fontsize=16, weight='bold')
    integral_axis = axis.twinx()
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-1, 1))
    integral_axis.yaxis.set_major_formatter(formatter)
    integral_axis.set_ylabel(r"Integrated VAF [$\mathrm{cm}^2/\mathrm{s}$]", fontsize=16, weight='bold')
    max_integral = 0
    
    # Use a sophisticated Seaborn color palette
    color_palette = sns.color_palette("muted", len(target_species))
    for idx, element in enumerate(target_species):
        line_color = color_palette[idx]
        vaf_avg = data_obj.get_array(f'vaf_isotropic_{element}_mean')
        vaf_err = data_obj.get_array(f'vaf_isotropic_{element}_sem')
        int_avg = data_obj.get_array(f'vaf_integral_isotropic_{element}_mean')
        int_err = data_obj.get_array(f'vaf_integral_isotropic_{element}_sem')
        
        # Enhanced lineplot with shadow effect
        sns.lineplot(x=time_points, y=vaf_avg, ax=axis, color=line_color, linewidth=3,
                     label=f'VAF ({element})', zorder=10)
        axis.fill_between(time_points, vaf_avg - vaf_err, vaf_avg + vaf_err,
                          facecolor=line_color, alpha=0.2, edgecolor='none', zorder=5)
        
        max_integral = max(max_integral, (int_avg + int_err).max())
        for traj_idx in range(num_traj):
            traj_vaf = data_obj.get_array(f'vaf_isotropic_{element}_{traj_idx}')
            traj_int = data_obj.get_array(f'vaf_integral_isotropic_{element}_{traj_idx}')
            max_integral = max(max_integral, traj_int.max())
            for block in range(len(traj_vaf)):
                axis.plot(time_points, traj_vaf[block], color=line_color, alpha=0.03, linewidth=1, zorder=1)
                integral_axis.plot(time_points, traj_int[block], color=line_color,
                                  alpha=0.03, linestyle='-.', linewidth=1, zorder=1)
        
        integral_axis.plot(time_points, int_avg, color=line_color, linewidth=3, linestyle='-.', zorder=10)
        integral_axis.fill_between(time_points, int_avg - int_err, int_avg + int_err,
                                  facecolor=line_color, alpha=0.2, edgecolor='none', zorder=5)
    
    integral_axis.set_ylim(0, max_integral * 1.1)  # Slight padding for elegance
    integral_axis.axvline(fit_start * fs_per_step * time_step, color='k', linewidth=1, alpha=0.5, linestyle='--')
    integral_axis.axvline(fit_end * fs_per_step * time_step, color='k', linewidth=1, alpha=0.5, linestyle='--')
    
    if not hide_legend:
        axis.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.1, 1))
        integral_axis.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.1, 0.5))
    
    # Refine axes appearance
    sns.despine(ax=axis, trim=True)
    sns.despine(ax=integral_axis, left=True, right=False, trim=True)
    axis.grid(True, which="both", ls="--", alpha=0.5)
    
    if display:
        plt.tight_layout()
        plt.show()

def plot_power_spectrum(data_obj, axis=None, display=False, signal_alpha=0.1, fill_alpha=0.2, **options):
    """
    Plot power spectrum or vibrational density of states with enhanced Seaborn styling.

    Parameters
    ----------
    data_obj : object
        Data object with spectrum arrays and attributes.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on.
    display : bool, optional
        If True, call plt.show().
    signal_alpha : float, optional
        Transparency for individual signals.
    fill_alpha : float, optional
        Transparency for error fill.
    """
    if axis is None:
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    # Sophisticated Seaborn styling
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("muted")
    
    properties = data_obj.get_attrs()
    target_elements = properties['species_of_interest']
    num_traj = properties['nr_of_trajectories']
    freq_arrays = [data_obj.get_array(f'frequency_{i}') for i in range(num_traj)]
    freq_base = freq_arrays[0]
    
    axis.set_xlabel(r'Frequency [THz]', fontsize=16, weight='bold')
    axis.set_ylabel(r'Signal [$\mathrm{\AA}^2\,\mathrm{fs}^{-1}$]', fontsize=16, weight='bold')
    
    color_palette = sns.color_palette("muted", len(target_elements))
    
    for idx, element in enumerate(target_elements):
        line_color = color_palette[idx]
        if signal_alpha > 1e-4:
            for traj_idx in range(num_traj):
                spec_data = data_obj.get_array(f'periodogram_{element}_{traj_idx}')
                for signal in spec_data:
                    axis.plot(freq_base, signal, color=line_color, alpha=signal_alpha, linewidth=1, zorder=1)
        try:
            spec_avg = data_obj.get_array(f'periodogram_{element}_mean')
            spec_err = data_obj.get_array(f'periodogram_{element}_sem')
            sns.lineplot(x=freq_base, y=spec_avg, ax=axis, color=line_color, linewidth=3, zorder=10)
            axis.fill_between(freq_base, spec_avg - spec_err, spec_avg + spec_err,
                              facecolor=line_color, alpha=fill_alpha, edgecolor='none', zorder=5)
        except Exception as error:
            print(error)
    
    sns.despine(ax=axis, trim=True)
    axis.grid(True, which="both", ls="--", alpha=0.5)
    
    if display:
        plt.tight_layout()
        plt.show()

def plot_rdf(data_obj, axis=None, secondary_axis=None, hide_legend=False, target_pairs=None,
                display=False, hide_labels=False, fill_alpha=0.2, skip_integral=False,
                rdf_params={}, int_params={}, **options):
    """
    Plot radial distribution functions (RDF) for atom pairs with enhanced Seaborn styling.

    Parameters
    ----------
    data_obj : object
        Data object with RDF arrays and attributes.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on.
    secondary_axis : matplotlib.axes.Axes, optional
        Secondary axis for integrals.
    hide_legend : bool, optional
        If True, suppress legend.
    target_pairs : list, optional
        List of species pairs to plot.
    display : bool, optional
        If True, call plt.show().
    hide_labels : bool, optional
        If True, suppress labels.
    fill_alpha : float, optional
        Transparency for error fill.
    skip_integral : bool, optional
        If True, do not plot integrals.
    rdf_params : dict, optional
        Additional plotting parameters for RDF.
    int_params : dict, optional
        Additional plotting parameters for integrals.

    Returns
    -------
    list
        List of line handles.
    """
    if axis is None:
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("muted")
    
    if not skip_integral and secondary_axis is None:
        secondary_axis = axis.twinx()
    
    properties = data_obj.get_attrs()
    pair_list = properties['species_pairs']
    if target_pairs is not None:
        pair_list = [pair for pair in pair_list if pair in target_pairs]
    
    color_map = {}
    color_palette = sns.color_palette("muted", len(set([pair[0] for pair in pair_list])))
    color_iterator = cycle(color_palette)
    line_handles = []
    
    for atom1, atom2 in pair_list:
        try:
            rdf_data = data_obj.get_array(f'rdf_{atom1}_{atom2}')
            int_data = data_obj.get_array(f'int_{atom1}_{atom2}')
            distances = data_obj.get_array(f'radii_{atom1}_{atom2}')
        except KeyError:
            print(f'Warning: RDF for {atom1}-{atom2} was not calculated, skipping')
            continue
        
        if atom1 not in color_map:
            color_map[atom1] = next(color_iterator)
        line_color = color_map[atom1]
        
        rdf_options = deepcopy(rdf_params)
        int_options = deepcopy(int_params)
        if 'color' not in rdf_options:
            rdf_options['color'] = line_color
        if 'label' not in rdf_options and not hide_labels:
            rdf_options['label'] = f'{atom1}-{atom2}'
        if 'label' not in int_options and not hide_labels:
            int_options['label'] = f'Integral {atom1}-{atom2}'
        
        rdf_line = sns.lineplot(x=distances, y=rdf_data, ax=axis, linewidth=3, **rdf_options, zorder=10)
        line_handles.append(rdf_line)
        if not skip_integral:
            int_line = sns.lineplot(x=distances, y=int_data, ax=secondary_axis,
                                   linestyle='-.', linewidth=3, **int_options, zorder=10)
            line_handles.append(int_line)
    
    axis.set_xlabel(r'$r$ [$\mathrm{\AA}$]', fontsize=16, weight='bold')
    axis.set_ylabel(r'$g(r)$', fontsize=16, weight='bold')
    if not hide_legend:
        axis.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.1, 1))
    if not skip_integral:
        secondary_axis.set_ylabel('Integral', fontsize=16, weight='bold')
    
    sns.despine(ax=axis, trim=True)
    if not skip_integral:
        sns.despine(ax=secondary_axis, left=True, right=False, trim=True)
    axis.grid(True, which="both", ls="--", alpha=0.5)
    
    if display:
        plt.tight_layout()
        plt.show()
    return line_handles

def plot_angular_spec(data_obj, axis=None, hide_legend=False, display=False, hide_labels=False, **options):
    """
    Plot angular spectra for atom triplets with enhanced Seaborn styling.

    Parameters
    ----------
    data_obj : object
        Data object with angular spectrum arrays and attributes.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on.
    hide_legend : bool, optional
        If True, suppress legend.
    display : bool, optional
        If True, call plt.show().
    hide_labels : bool, optional
        If True, suppress labels.
    """
    if axis is None:
        options.pop('ax', None)
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("muted")
    
    properties = data_obj.get_attrs()
    triplet_list = properties['species_pairs']  # Assuming triplets are stored similarly
    line_handles = []
    color_palette = sns.color_palette("muted", len(triplet_list))
    
    for idx, (atom1, atom2, atom3) in enumerate(triplet_list):
        ang_data = data_obj.get_array(f'aspec_{atom1}_{atom2}_{atom3}')
        angle_values = data_obj.get_array(f'angles_{atom1}_{atom2}_{atom3}')
        line_color = color_palette[idx]
        label_text = f'{atom2}-{atom1}-{atom3}' if not hide_labels else None
        line = sns.lineplot(x=angle_values, y=ang_data, ax=axis, label=label_text, color=line_color, linewidth=3, zorder=10)
        line_handles.append(line)
    
    if not hide_legend:
        axis.legend(handles=line_handles, frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.despine(ax=axis, trim=True)
    axis.grid(True, which="both", ls="--", alpha=0.5)
    
    if display:
        plt.tight_layout()
        plt.show()

def parse_cli_args():
    """
    Parse command-line arguments for MSD and Van Hove plotting.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Make MSD or Van Hove plots from JSON data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Subcommand: 'msd' or 'van_hove'")

    # MSD Subcommand
    msd_parser = subparsers.add_parser("msd", help="Generate MSD plots from JSON data.")
    msd_parser.add_argument("--data-path", required=True,
                            help="Path to the JSON file with simulation data.")
    msd_parser.add_argument("--save-path", default=None,
                            help="If provided, the plot is saved to this file (JPEG).")
    msd_parser.add_argument("--plot-data", action="append", nargs="+", required=True,
                            help="Plot instructions. E.g: --plot-data 600 Li msd_tracer XYZ")
    msd_parser.add_argument("--first-time", nargs="+", type=float, default=[0],
                            help="List of start times for each time window.")
    msd_parser.add_argument("--last-time", nargs="+", type=float, default=[10],
                            help="List of end times for each time window.")

    # Van Hove Subcommand
    vh_parser = subparsers.add_parser("van_hove", help="Generate Van Hove plots from JSON data.")
    vh_parser.add_argument("--data-path", required=True,
                           help="Path to the JSON file with simulation data.")
    vh_parser.add_argument("--save-path", default=None,
                           help="If provided, the plot is saved to this file (JPEG).")
    vh_parser.add_argument("--figsize", nargs=2, type=float, default=[10,8],
                           help="Figure size (width height). Default=10 8")
    vh_parser.add_argument("--plot-data", action="append", nargs="+", required=True,
                           help="Van Hove instructions. E.g: --plot-data 600 Li Self")

    return parser.parse_args()

def main():
    """
    Main entry point for plotting CLI.
    Dispatches to MSD or Van Hove plotting based on subcommand.
    """
    args = parse_cli_args()
    
    if args.command == "msd":
        final_plot_data = []
        for item in args.plot_data:
            converted = [int_or_float(item[0]), item[1], item[2], item[3]]
            final_plot_data.append(converted)

        msd_plot(
            data_path=args.data_path,
            Plot_data=final_plot_data,
            First_time=args.first_time,
            Last_time=args.last_time,
            save_path=args.save_path
        )

    elif args.command == "van_hove":
        final_plot_data = []
        for item in args.plot_data:
            converted = [int_or_float(item[0]), item[1], item[2]]
            final_plot_data.append(converted)

        van_hove_plot(
            data_path=args.data_path,
            Plot_data=final_plot_data,
            save_path=args.save_path,
            figsize=tuple(args.figsize)
        )

def int_or_float(val):
    """
    Helper function to convert a string to int if possible, else float.

    Args:
        val (str): Value to convert.

    Returns:
        int or float: Converted value.
    """
    try:
        return int(val)
    except ValueError:
        return float(val)

if __name__ == "__main__":
    main()