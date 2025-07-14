"""
Plotting and Visualization Module for CPDyAna
=============================================

This module provides comprehensive plotting functions for visualizing molecular
dynamics analysis results using Seaborn's enhanced styling capabilities.

Key Features:
- Publication-quality figures with consistent styling
- Enhanced color palettes and visual aesthetics
- Statistical visualization of uncertainty
- Responsive layouts and multi-panel figures
- Accessible and customizable plot elements

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

def msd_plot_lammps(data_path, Plot_data, First_time, Last_time, save_path=None):
    """
    Create enhanced MSD plots from LAMMPS data using Seaborn styling.

    Args:
        data_path (str): Path to JSON file with MSD data.
        Plot_data (list): List of plot specifications (TEMP, ELEMENT, DATA_TYPE, DIRECTION).
        First_time (float): Start time for plotting window (ps).
        Last_time (float): End time for plotting window (ps).
        save_path (str, optional): File path to save the plot.

    Returns:
        None. Displays and optionally saves the plot.
    """
    # Load data
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    # Set up figure with Seaborn styling
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("tab10", len(Plot_data))
    
    # Find common elements across plot data for labeling
    common_parts_set = set(Plot_data[0])
    for entry in Plot_data[1:]:
        common_parts_set.intersection_update(set(entry))
    
    # Plot each data series
    for idx, info in enumerate(Plot_data):
        suffix = '' if info[3] == "XYZ" else f"_{info[3]}"
        Temp = f"({info[0]}, '{info[1]}')"
        Element = str(info[1])
        
        # Construct data keys
        msd_array_key = f"{info[2]}_msd_array{suffix}"
        time_array_key = f"{info[2]}_time_array{suffix}"
        diffusivity_key = f"{info[2]}_diffusivity{suffix}"
        diffusivity_error_key = f"{info[2]}_diffusivity_error{suffix}"
        slope_sem_key = f"{info[2]}_slope_sem{suffix}"
        
        # Retrieve and validate data
        try:
            msd_blob = data[Temp]['msd_data'][Element]
            time_array = np.array(msd_blob[time_array_key][0])
            msd_mean = np.array(msd_blob[msd_array_key][0])
            
            # Check array lengths
            min_length = min(len(time_array), len(msd_mean))
            if len(time_array) != len(msd_mean):
                print(f"Info: Array length adjustment for {Element} {info[3]}. Using first {min_length} points.")
                time_array = time_array[:min_length]
                msd_mean = msd_mean[:min_length]
            
            # Validate data quality
            valid_mask = (~np.isnan(msd_mean)) & (~np.isnan(time_array)) & (np.isfinite(msd_mean)) & (np.isfinite(time_array))
            if not np.any(valid_mask):
                print(f"Warning: No valid data points for {Element} {info[3]}. Skipping plot.")
                continue
            
            time_array = time_array[valid_mask]
            msd_mean = msd_mean[valid_mask]
            
            # Get diffusion data
            D = msd_blob[diffusivity_key][0] if diffusivity_key in msd_blob else np.nan
            D_err = msd_blob[diffusivity_error_key][0] if diffusivity_error_key in msd_blob else np.nan
            slope_sem = msd_blob[slope_sem_key][0] if slope_sem_key in msd_blob else np.nan
            
            # Apply time window
            First = np.searchsorted(time_array, First_time, side='left')
            Last = np.searchsorted(time_array, Last_time, side='right') - 1
            
            if First >= len(time_array) or Last < 0 or First > Last:
                print(f"Warning: Time window {First_time}-{Last_time} ps is outside data range for {Element} {info[3]}")
                plot_times = time_array
                plot_msd = msd_mean
            else:
                plot_times = time_array[First:Last+1]
                plot_msd = msd_mean[First:Last+1]
            
            # Calculate error bounds
            msd_sem = slope_sem * plot_times if not np.isnan(slope_sem) else np.zeros_like(plot_times)
            
            # Create label
            if np.isnan(D_err) or np.isnan(D):
                label = f"{Element} {info[3]}: D calculation failed"
            else:
                label = f"{Element} {info[3]}: D = {D:.2e} ± {D_err:.2e} cm²/s"
            
            # Plot with Seaborn
            sns.lineplot(x=plot_times, y=plot_msd, label=label, color=palette[idx], linewidth=2)
            if np.any(msd_sem > 0):
                plt.fill_between(plot_times, plot_msd - msd_sem, plot_msd + msd_sem, color=palette[idx], alpha=0.2)
                
        except KeyError as e:
            print(f"Warning: Missing data for {Element} {info[3]}: {e}")
            continue
    
    # Enhance plot appearance
    plt.xlabel('Time (ps)', fontweight='bold', fontsize=14)
    plt.ylabel(r'MSD [$\mathrm{\AA^2}$]', fontweight='bold', fontsize=14)
    plt.title('Mean Square Displacement (LAMMPS)', fontsize=16)
    sns.despine()
    
    # Add legend with better positioning
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def msd_plot_qe(data_path, Plot_data, First_time, Last_time, save_path=None):
    """
    Create enhanced MSD plots from QE data using Seaborn styling.

    Args:
        data_path (str): Path to JSON file with MSD data.
        Plot_data (list): List of plot specifications (TEMP, ELEMENT, DATA_TYPE, DIRECTION).
        First_time (float): Start time for plotting window (ps).
        Last_time (float): End time for plotting window (ps).
        save_path (str, optional): File path to save the plot.

    Returns:
        None. Displays and optionally saves the plot.
    """
    # Load data
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    # Set up figure with Seaborn styling
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("tab10", len(Plot_data))
    
    # Find common elements across plot data
    common_parts_set = set(Plot_data[0])
    for entry in Plot_data[1:]:
        common_parts_set.intersection_update(set(entry))
    
    # Plot each data series
    for idx, info in enumerate(Plot_data):
        suffix = '' if info[3] == "XYZ" else f"_{info[3]}"
        Temp = f"({info[0]}, '{info[1]}')"
        Element = str(info[1])
        
        try:
            dt = data[Temp]['dt_dict']
            First = np.searchsorted(dt, First_time, side='left')
            Last = np.searchsorted(dt, Last_time, side='right') - 1
            
            # Construct data keys
            msd_array_key = f"{info[2]}_msd_array{suffix}"
            diffusivity_key = f"{info[2]}_diffusivity{suffix}"
            slope_sem_key = f"{info[2]}_slope_sem{suffix}"
            diffusivity_error_key = f"{info[2]}_diffusivity_error{suffix}"
            
            # Get MSD data
            msd_mean = np.array(data[Temp]['msd_data'][Element][msd_array_key][0])
            msd_sem = np.array([data[Temp]['msd_data'][Element][slope_sem_key][0] * x for x in dt])
            D = data[Temp]['msd_data'][Element][diffusivity_key][0]
            D_err = data[Temp]['msd_data'][Element][diffusivity_error_key][0]
            
            # Create label
            uncommon_parts = [str(part) for part in info if part not in common_parts_set and part not in ['', []]]
            label_prefix = f"{', '.join(uncommon_parts)}: " if uncommon_parts else ""
            label = f"{label_prefix}{Element} {info[3]}: D = {D:.2e} ± {D_err:.2e} cm²/s"
            
            # Plot with Seaborn
            sns.lineplot(x=dt[First:Last+1], y=msd_mean[First:Last+1], label=label, color=palette[idx], linewidth=2)
            plt.fill_between(dt[First:Last+1], msd_mean[First:Last+1] - msd_sem[First:Last+1], 
                             msd_mean[First:Last+1] + msd_sem[First:Last+1], color=palette[idx], alpha=0.2)
        
        except (KeyError, IndexError) as e:
            print(f"Warning: Missing data for {Element} {info[3]}: {e}")
            continue
    
    # Enhance plot appearance
    plt.xlabel('Time (ps)', fontweight='bold', fontsize=14)
    plt.ylabel(r'MSD [$\mathrm{\AA^2}$]', fontweight='bold', fontsize=14)
    plt.title('Mean Square Displacement (QE)', fontsize=16)
    sns.despine()
    
    # Add legend with better positioning
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def msd_plot_bomd(data_path, Plot_data, First_time, Last_time, save_path=None):
    """
    Create MSD plots for BOMD data using dt_dict and msd_data keys.

    Args:
        data_path (str): Path to JSON file with MSD data.
        Plot_data (list): List of plot specifications (TEMP, ELEMENT, DATA_TYPE, DIRECTION).
        First_time (float): Start time for plotting window (ps).
        Last_time (float): End time for plotting window (ps).
        save_path (str, optional): File path to save the plot.

    Returns:
        None. Displays and optionally saves the plot.
    """
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data
    with open(data_path, 'r') as file:
        data = json.load(file)

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("tab10", len(Plot_data))

    for idx, info in enumerate(Plot_data):
        Temp = f"({info[0]}, '{info[1]}')"
        Element = str(info[1])
        msd_type = info[2]
        direction = info[3]
        suffix = '' if direction == "XYZ" else f"_{direction}"

        try:
            dt = np.array(data[Temp]['dt_dict'])
            msd_blob = data[Temp]['msd_data'][Element]
            msd_array_key = f"{msd_type}_msd_array{suffix}"
            time_array_key = f"{msd_type}_time_array{suffix}"
            diffusivity_key = f"{msd_type}_diffusivity{suffix}"
            diffusivity_error_key = f"{msd_type}_diffusivity_error{suffix}"
            slope_sem_key = f"{msd_type}_slope_sem{suffix}"

            # Use Tracer_time_array if present, else dt_dict
            if time_array_key in msd_blob:
                time_array = np.array(msd_blob[time_array_key][0])
            else:
                time_array = dt

            msd_mean = np.array(msd_blob[msd_array_key][0])
            msd_sem = np.array([msd_blob.get(slope_sem_key, [0])[0] * x for x in time_array])
            D = msd_blob.get(diffusivity_key, [np.nan])[0]
            D_err = msd_blob.get(diffusivity_error_key, [np.nan])[0]

            # Apply time window
            First = np.searchsorted(time_array, First_time, side='left')
            Last = np.searchsorted(time_array, Last_time, side='right') - 1

            # Label
            label = f"{Element} {direction}: D = {D:.2e} ± {D_err:.2e} cm²/s"

            # Plot
            sns.lineplot(x=time_array[First:Last+1], y=msd_mean[First:Last+1], label=label, color=palette[idx], linewidth=2)
            plt.fill_between(time_array[First:Last+1], msd_mean[First:Last+1] - msd_sem[First:Last+1],
                             msd_mean[First:Last+1] + msd_sem[First:Last+1], color=palette[idx], alpha=0.2)
        except Exception as e:
            print(f"Warning: Could not plot {Element} {direction}: {e}")
            continue

    plt.xlabel('Time (ps)', fontweight='bold', fontsize=14)
    plt.ylabel(r'MSD [$\mathrm{\AA^2}$]', fontweight='bold', fontsize=14)
    plt.title('Mean Square Displacement (BOMD)', fontsize=16)
    sns.despine()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.tight_layout()
    if save_path:
        print(f"Saving to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# def msd_plot(data_path, Plot_data, First_time, Last_time, save_path=None, is_lammps=False):
#     """
#     Dispatcher function for MSD plotting based on data source.

#     Args:
#         data_path (str): Path to JSON file with MSD data.
#         Plot_data (list): List of plot specifications.
#         First_time (float): Start time for plotting window (ps).
#         Last_time (float): End time for plotting window (ps).
#         save_path (str, optional): File path to save the plot.
#         is_lammps (bool): If True, use LAMMPS plotting; else QE.

#     Returns:
#         None.
#     """
#     if is_lammps:
#         msd_plot_lammps(data_path, Plot_data, First_time, Last_time, save_path)
#     else:
#         msd_plot_qe(data_path, Plot_data, First_time, Last_time, save_path)

def msd_plot(data_path, Plot_data, First_time, Last_time, save_path=None, is_lammps=False, is_bomd=False, is_qe=False):
    """
    Dispatcher function for MSD plotting based on data source.

    Args:
        data_path (str): Path to JSON file with MSD data.
        Plot_data (list): List of plot specifications.
        First_time (float): Start time for plotting window (ps).
        Last_time (float): End time for plotting window (ps).
        save_path (str, optional): File path to save the plot.
        is_lammps (bool): If True, use LAMMPS plotting.
        is_bomd (bool): If True, use BOMD plotting.
        is_qe (bool): If True, use QE plotting.

    Returns:
        None.
    """
    if is_lammps:
        msd_plot_lammps(data_path, Plot_data, First_time, Last_time, save_path)
    elif is_bomd:
        msd_plot_bomd(data_path, Plot_data, First_time, Last_time, save_path)
    elif is_qe:
        msd_plot_qe(data_path, Plot_data, First_time, Last_time, save_path)

def centers_to_edges_time(centers):
    """
    Convert bin centers to bin edges for time axis plotting,
    ensuring the first edge is always at 0.
    Handles single-bin case robustly.
    """
    centers = np.asarray(centers)
    if len(centers) == 1:
        width = 1.0  # or any small value
        return np.array([centers[0] - width/2, centers[0] + width/2])
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = 0  # Force first edge to be 0 for time axis
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges

def centers_to_edges(centers):
    """
    Convert bin centers to bin edges for general plotting.
    Handles single-bin case robustly.
    """
    centers = np.asarray(centers)
    if len(centers) == 1:
        width = 1.0  # or any small value
        return np.array([centers[0] - width/2, centers[0] + width/2])
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges

# def centers_to_edges_time(centers):
#     """
#     Convert bin centers to bin edges for time axis plotting,
#     ensuring the first edge is always at 0.

#     Args:
#         centers (np.ndarray): Array of bin centers.

#     Returns:
#         np.ndarray: Array of bin edges.
#     """
#     edges = np.zeros(len(centers) + 1)
#     edges[1:-1] = (centers[:-1] + centers[1:]) / 2
#     edges[0] = 0  # Force first edge to be 0 for time axis
#     edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
#     return edges

# def centers_to_edges(centers):
#     """
#     Convert bin centers to bin edges for general plotting.

#     Args:
#         centers (np.ndarray): Array of bin centers.

#     Returns:
#         np.ndarray: Array of bin edges.
#     """
#     edges = np.zeros(len(centers) + 1)
#     edges[1:-1] = (centers[:-1] + centers[1:]) / 2
#     edges[0] = centers[0] - (centers[1] - centers[0]) / 2
#     edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
#     return edges

def van_hove_plot(data_path, Plot_data, save_path=None, figsize=(10, 8), first_time=None, last_time=None, step_skip=10, dt=1.0):
    """
    Create enhanced Van Hove correlation function plots using Seaborn styling.

    Args:
        data_path (str): Path to JSON file with Van Hove data.
        Plot_data (list): List of plot specifications (TEMP, ELEMENT, TYPE).
        save_path (str, optional): Base file path for saving plots.
        figsize (tuple): Figure size in inches.

    Returns:
        None. Displays and optionally saves the plot.
    """
    # Load data
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    # Process each plot specification
    for temp, element, h_type in Plot_data:
        # Configure plot based on type
        if h_type == "Self":
            vmax = 1.0
            cb_label = r"$4\pi r^2 G_s(t, r)$"
            title = f"Self Van Hove Function - {element} at {temp}K"
        elif h_type == "Distinct":
            vmax = 4.0
            cb_label = r"$G_d(t, r)$"
            title = f"Distinct Van Hove Function - {element} at {temp}K"
        else:
            vmax = 1.0
            cb_label = "Van Hove Function"
            title = f"Van Hove Function - {element} at {temp}K"
        
        # Get data
        data_key = f"({temp}, '{element}')"
        try:
            van_hove = np.array(data[data_key]["evaluated_corr"][h_type]["grt"])
            dist_array = np.array(data[data_key]["evaluated_corr"][h_type]["dist_interval"])
            reduced_nt = data[data_key]["evaluated_corr"][h_type]["reduced_nt"]
            
            # Validate data
            if van_hove.ndim != 2 or van_hove.shape[1] != len(dist_array):
                print(f"Warning: Invalid Van Hove data shape for {element} {h_type}")
                continue
            
            # Set up plot with Seaborn styling
            plt.figure(figsize=figsize)
            sns.set_theme(style="ticks")
            
            # Prepare axes
            # x = np.arange(reduced_nt) * 10  # step_skip=10, dt=1.0
            dt_dict = np.array(data[data_key].get('dt_dict', []))
            step_skip_used = 10  # or get from your analysis parameters or JSON
            if len(dt_dict) > 0:
                max_it = (len(dt_dict) - 1) // step_skip_used + 1
                x = np.array([dt_dict[it * step_skip_used] - dt_dict[0] for it in range(min(reduced_nt, max_it))])
                van_hove = van_hove[:len(x), :]
            else:
                x = np.arange(reduced_nt) * step_skip_used * 1.0  # fallback
            if last_time is not None:
                mask = x <= last_time
                x = x[mask]
                van_hove = van_hove[:len(x), :]
            else:
                # fallback: plot all
                mask = slice(None)
            y = dist_array
            x_edges = centers_to_edges_time(x)
            y_edges = centers_to_edges(y)
            
            # Create heatmap
            heatmap = plt.pcolormesh(x_edges, y_edges, van_hove.T, 
                                   cmap="viridis", shading="auto", 
                                   vmin=np.min(van_hove), vmax=vmax)
           
            # Enhance appearance
            plt.xlabel("Time (ps)", fontweight='bold', fontsize=14)
            plt.ylabel(r"$r$ ($\AA$)", fontweight='bold', fontsize=14)
            plt.title(title, fontsize=16)
            
            # Add colorbar with better formatting
            cbar = plt.colorbar(heatmap)
            cbar.set_label(cb_label, fontsize=14, fontweight='bold')
            
            sns.despine()
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                actual_save_path = save_path.replace('.', f'_{element}_{h_type}.')
                plt.savefig(actual_save_path, dpi=300, bbox_inches='tight')
                print(f"Saved Van Hove plot to {actual_save_path}")
            
            plt.show()
            plt.close()
            
        except (KeyError, ValueError) as e:
            print(f"Error processing Van Hove data for {element} {h_type}: {e}")
            continue

def ngp_plot(data_source, plot_data, first_time, last_time, save_path="NGP.jpg"):
    """
    Generate enhanced Non-Gaussian Parameter (NGP) plots using Seaborn styling.

    Args:
        data_source (str or dict): Path to JSON file or loaded data dict.
        plot_data (list): List of plot specifications (TEMP, ELEMENT, DIRECTION).
        first_time (float): Start time for plotting window (ps).
        last_time (float): End time for plotting window (ps).
        save_path (str): File path to save the plot.

    Returns:
        None. Displays and optionally saves the plot.
    """
    # Load data if a file path is provided
    if isinstance(data_source, str):
        with open(data_source, 'r') as file:
            data = json.load(file)
    else:
        data = data_source
    
    # Set up figure with Seaborn styling
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("tab10", len(plot_data))
    
    # Plot each data series
    for idx, (temp, ele, direction) in enumerate(plot_data):
        try:
            # Construct key and retrieve data
            key = str((temp, ele))
            suffix = '' if direction == 'XYZ' else f'_{direction}'
            ngp_key = f'NGP_array{suffix}'
            time_lags_key = f'time_lags{suffix}'
            
            # Get data arrays
            time_lags = np.array(data[key]['ngp_data'][ele][time_lags_key][0])
            ngp_values = np.array(data[key]['ngp_data'][ele][ngp_key][0])
            
            # Filter by time range
            mask = (time_lags >= first_time) & (time_lags <= last_time)
            if np.any(mask):
                # Plot with Seaborn
                sns.lineplot(x=time_lags[mask], y=ngp_values[mask], 
                           label=f'T={temp}K, {ele}, {direction}', 
                           color=palette[idx], linewidth=2)
            else:
                print(f"Warning: No data within time range {first_time}-{last_time} ps for T={temp}, {ele}, {direction}")
                
        except (KeyError, TypeError, IndexError) as e:
            print(f"Warning: Could not plot NGP for T={temp}, {ele}, {direction}: {e}")
            continue
    
    # Enhance appearance
    print("NGP array:", ngp_values)
    print("Time lags:", time_lags)
    plt.xscale('log')
    plt.xlabel('Time Lag (ps)', fontweight='bold', fontsize=14)
    plt.ylabel('Non-Gaussian Parameter, α₂(t)', fontweight='bold', fontsize=14)
    plt.title('Non-Gaussian Parameter vs Time Lag', fontsize=16)
    plt.ylim(-0.5, 2.0)
    
    # Add grid with better styling
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.5)
    
    # Improve legend
    plt.legend(fontsize=12, framealpha=0.9, edgecolor='gray')
    sns.despine()
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def plot_vaf_isotropic(data_obj, axis=None, hide_legend=False, target_species=None, display=False, **options):
    """
    Plot isotropic velocity autocorrelation function (VAF) with enhanced Seaborn styling.

    Args:
        data_obj: DynamicsAnalyzer or similar object with VAF data.
        axis (matplotlib.axes.Axes, optional): Axis to plot on.
        hide_legend (bool): If True, do not display legend.
        target_species (list, optional): List of species to plot.
        display (bool): If True, show the plot.
        **options: Additional matplotlib options.

    Returns:
        None.
    """
    if axis is None:
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    # Apply Seaborn styling
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Get data properties
    properties = data_obj.get_attrs()
    num_traj = properties['nr_of_trajectories']
    fit_start = properties['t_start_fit_dt']
    fit_end = properties['t_end_fit_dt']
    time_step = properties.get('stepsize_t', 1)
    fs_per_step = properties['timestep_fs']
    
    # Calculate time points
    time_points = fs_per_step * time_step * np.arange(
        properties.get('t_start_dt') / time_step,
        properties.get('t_end_dt') / time_step
    )
    
    # Set target species if not specified
    if target_species is None:
        target_species = properties['species_of_interest']
    
    # Set up axes
    axis.set_ylabel(r'VAF [$\mathrm{\AA}^2\,\mathrm{fs}^{-2}$]', fontweight='bold', fontsize=14)
    axis.set_xlabel('Time [fs]', fontweight='bold', fontsize=14)
    integral_axis = axis.twinx()
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-1, 1))
    integral_axis.yaxis.set_major_formatter(formatter)
    integral_axis.set_ylabel(r"Integrated VAF [$\mathrm{cm}^2/\mathrm{s}$]", fontweight='bold', fontsize=14)
    max_integral = 0
    
    # Generate color palette
    palette = sns.color_palette("tab10", len(target_species))
    
    # Plot each species
    for idx, element in enumerate(target_species):
        color = palette[idx]
        
        # Get VAF data
        vaf_avg = data_obj.get_array(f'vaf_isotropic_{element}_mean')
        vaf_err = data_obj.get_array(f'vaf_isotropic_{element}_sem')
        int_avg = data_obj.get_array(f'vaf_integral_isotropic_{element}_mean')
        int_err = data_obj.get_array(f'vaf_integral_isotropic_{element}_sem')
        
        # Get diffusion coefficient for legend
        attrs = data_obj.get_attrs()
        D = None
        D_err = None
        if element in attrs and "diffusion_mean_cm2_s" in attrs[element]:
            D = attrs[element]["diffusion_mean_cm2_s"]
            D_err = attrs[element].get("diffusion_sem_cm2_s", None)
            if D_err is not None:
                label_vaf = f'VAF ({element})'
                label_int = (r"$D^{\mathrm{VAF}}_{%s} = (%.2e \pm %.2e)\,\mathrm{cm}^2/\mathrm{s}$"
                             % (element, D, D_err))
            else:
                label_vaf = f'VAF ({element})'
                label_int = (r"$D^{\mathrm{VAF}}_{%s} = %.2e\,\mathrm{cm}^2/\mathrm{s}$"
                             % (element, D))
        else:
            label_vaf = f'VAF ({element})'
            label_int = f'Integral ({element})'
        
        # Plot main VAF
        sns.lineplot(x=time_points, y=vaf_avg, ax=axis, color=color, 
                   linewidth=2.5, label=label_vaf, zorder=10)
        axis.fill_between(time_points, vaf_avg - vaf_err, vaf_avg + vaf_err,
                        facecolor=color, alpha=0.2, edgecolor='none', zorder=5)
        
        # Track max integral value for scaling
        max_integral = max(max_integral, (int_avg + int_err).max())
        
        # Plot individual trajectories with low opacity
        for traj_idx in range(num_traj):
            traj_vaf = data_obj.get_array(f'vaf_isotropic_{element}_{traj_idx}')
            traj_int = data_obj.get_array(f'vaf_integral_isotropic_{element}_{traj_idx}')
            max_integral = max(max_integral, traj_int.max())
            
            # Plot each block with low opacity
            for block in range(len(traj_vaf)):
                sns.lineplot(x=time_points, y=traj_vaf[block], ax=axis, color=color, 
                           alpha=0.05, linewidth=0.5, zorder=1, legend=False)
                sns.lineplot(x=time_points, y=traj_int[block], ax=integral_axis, color=color,
                           alpha=0.05, linestyle='-.', linewidth=0.5, zorder=1, legend=False)
        
        # Plot integral VAF with diffusion coefficient in label
        sns.lineplot(x=time_points, y=int_avg, ax=integral_axis, color=color, 
                   linewidth=2.5, linestyle='-.', label=label_int, zorder=10)
        integral_axis.fill_between(time_points, int_avg - int_err, int_avg + int_err,
                                 facecolor=color, alpha=0.2, edgecolor='none', zorder=5)
    
    # Set y-axis limit for integral axis
    integral_axis.set_ylim(0, max_integral * 1.1)
    
    # Add fitting region indicators
    integral_axis.axvline(fit_start * fs_per_step * time_step, color='k', linewidth=1, alpha=0.5, linestyle='--')
    integral_axis.axvline(fit_end * fs_per_step * time_step, color='k', linewidth=1, alpha=0.5, linestyle='--')
    
    # Add legends if not hidden
    if not hide_legend:
        handles1, labels1 = axis.get_legend_handles_labels()
        handles2, labels2 = integral_axis.get_legend_handles_labels()
        # Combine and remove duplicates while preserving order
        seen = set()
        handles_labels = []
        for h, l in zip(handles1 + handles2, labels1 + labels2):
            if l not in seen:
                handles_labels.append((h, l))
                seen.add(l)
        handles, labels = zip(*handles_labels)
        axis.legend(
            handles, labels,
            loc='upper left',
            frameon=True,
            edgecolor='black',
            fontsize=9,           # smaller font
            bbox_to_anchor=(0.01, 0.99),  # slightly inside the plot
            borderaxespad=0.2,
            labelspacing=0.4,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.5,
            fancybox=True
        )
        # Remove the integral axis legend
        integral_axis.legend_.remove() if integral_axis.legend_ else None

    # Clean up appearance
    sns.despine(ax=axis, right=True, left=False)
    sns.despine(ax=integral_axis, left=True, right=False)
    
    # Display if requested
    if display:
        plt.tight_layout()
        plt.show()

def plot_power_spectrum(data_obj, axis=None, display=False, signal_alpha=0.1, fill_alpha=0.2, **options):
    """
    Plot power spectrum with enhanced Seaborn styling.

    Args:
        data_obj: DynamicsAnalyzer or similar object with spectrum data.
        axis (matplotlib.axes.Axes, optional): Axis to plot on.
        display (bool): If True, show the plot.
        signal_alpha (float): Opacity for individual signals.
        fill_alpha (float): Opacity for error fill.
        **options: Additional matplotlib options.

    Returns:
        None.
    """
    if axis is None:
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    # Apply Seaborn styling
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Get properties
    properties = data_obj.get_attrs()
    target_elements = properties['species_of_interest']
    num_traj = properties['nr_of_trajectories']
    
    # Get frequency arrays
    freq_arrays = [data_obj.get_array(f'frequency_{i}') for i in range(num_traj)]
    freq_base = freq_arrays[0]
    
    # Set up axes
    axis.set_xlabel(r'Frequency [THz]', fontweight='bold', fontsize=14)
    axis.set_ylabel(r'Signal [$\mathrm{\AA}^2\,\mathrm{fs}^{-1}$]', fontweight='bold', fontsize=14)
    axis.set_title('Vibrational Density of States', fontsize=16)
    
    # Generate color palette
    palette = sns.color_palette("tab10", len(target_elements))
    
    # Plot each element
    for idx, element in enumerate(target_elements):
        color = palette[idx]
        
        # Plot individual signals if alpha is significant
        if signal_alpha > 1e-4:
            for traj_idx in range(num_traj):
                try:
                    spec_data = data_obj.get_array(f'periodogram_{element}_{traj_idx}')
                    for signal in spec_data:
                        sns.lineplot(x=freq_base, y=signal, ax=axis, color=color, 
                                   alpha=signal_alpha, linewidth=0.5, zorder=1, legend=False)
                except:
                    pass
                    
        # Plot mean and error
        try:
            spec_avg = data_obj.get_array(f'periodogram_{element}_mean')
            spec_err = data_obj.get_array(f'periodogram_{element}_sem')
            
            sns.lineplot(x=freq_base, y=spec_avg, ax=axis, color=color, 
                       linewidth=2.5, label=element, zorder=10)
            axis.fill_between(freq_base, spec_avg - spec_err, spec_avg + spec_err,
                            facecolor=color, alpha=fill_alpha, edgecolor='none', zorder=5)
        except Exception as e:
            print(f"Error plotting mean spectrum for {element}: {e}")
    
    # Clean up appearance
    sns.despine()
    axis.legend(fontsize=12, framealpha=0.9, edgecolor='gray')
    
    # Display if requested
    if display:
        plt.tight_layout()
        plt.show()

def plot_rdf(data_obj, axis=None, secondary_axis=None, hide_legend=False, target_pairs=None,
             display=False, hide_labels=False, fill_alpha=0.2, skip_integral=False,
             rdf_params={}, int_params={}, **options):
    """
    Plot radial distribution functions (RDF) with enhanced Seaborn styling.

    Args:
        data_obj: RDF analysis object with RDF data.
        axis (matplotlib.axes.Axes, optional): Axis to plot on.
        secondary_axis (matplotlib.axes.Axes, optional): Axis for integral.
        hide_legend (bool): If True, do not display legend.
        target_pairs (list, optional): List of atom pairs to plot.
        display (bool): If True, show the plot.
        hide_labels (bool): If True, do not display labels.
        fill_alpha (float): Opacity for error fill.
        skip_integral (bool): If True, do not plot integral.
        rdf_params (dict): Additional plotting options for RDF.
        int_params (dict): Additional plotting options for integral.
        **options: Additional matplotlib options.

    Returns:
        list: List of line handles for the plotted curves.
    """
    if axis is None:
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    # Apply Seaborn styling
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Set up secondary axis if needed
    if not skip_integral and secondary_axis is None:
        secondary_axis = axis.twinx()
    
    # Get properties
    properties = data_obj.get_attrs()
    pair_list = properties['species_pairs']
    if target_pairs is not None:
        pair_list = [pair for pair in pair_list if pair in target_pairs]
    
    # Generate color mapping for consistent colors by first element
    unique_first_elements = sorted(set([pair[0] for pair in pair_list]))
    palette = sns.color_palette("tab10", len(unique_first_elements))
    color_map = {elem: palette[i] for i, elem in enumerate(unique_first_elements)}
    
    # For returning line handles
    line_handles = []
    
    # Plot each pair
    for atom1, atom2 in pair_list:
        try:
            # Get RDF data
            rdf_data = data_obj.get_array(f'rdf_{atom1}_{atom2}')
            int_data = data_obj.get_array(f'int_{atom1}_{atom2}')
            distances = data_obj.get_array(f'radii_{atom1}_{atom2}')
            
            # Set color based on first atom
            color = color_map[atom1]
            
            # Prepare plotting options
            rdf_opts = deepcopy(rdf_params)
            int_opts = deepcopy(int_params)
            
            # Set color if not specified
            if 'color' not in rdf_opts:
                rdf_opts['color'] = color
                
            # Set labels if not hidden
            if 'label' not in rdf_opts and not hide_labels:
                rdf_opts['label'] = f'{atom1}-{atom2}'
            if 'label' not in int_opts and not hide_labels:
                int_opts['label'] = f'Integral {atom1}-{atom2}'
            
            # Plot RDF
            rdf_line = sns.lineplot(x=distances, y=rdf_data, ax=axis, linewidth=2.5, **rdf_opts, zorder=10)
            line_handles.append(rdf_line)
            
            # Plot integral if not skipped
            if not skip_integral:
                int_line = sns.lineplot(x=distances, y=int_data, ax=secondary_axis,
                                      linestyle='-.', linewidth=2.5, **int_opts, zorder=10)
                line_handles.append(int_line)
                
        except KeyError:
            print(f'Warning: RDF for {atom1}-{atom2} was not calculated, skipping')
            continue
    
    # Set axis labels and appearance
    axis.set_xlabel(r'$r$ [$\mathrm{\AA}$]', fontweight='bold', fontsize=14)
    axis.set_ylabel(r'$g(r)$', fontweight='bold', fontsize=14)
    axis.set_title('Radial Distribution Function', fontsize=16)
    
    # Add legend if not hidden
    if not hide_legend:
        axis.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.1, 1))
        
    # Set secondary axis label if used
    if not skip_integral:
        secondary_axis.set_ylabel('Coordination Number', fontweight='bold', fontsize=14)
    
    # Clean up appearance
    sns.despine(ax=axis, right=True, left=False)
    if not skip_integral:
        sns.despine(ax=secondary_axis, left=True, right=False)
    
    # Display if requested
    if display:
        plt.tight_layout()
        plt.show()
        
    return line_handles

def plot_angular_spec(data_obj, axis=None, hide_legend=False, display=False, hide_labels=False, **options):
    """
    Plot angular spectra for atom triplets with enhanced Seaborn styling.

    Args:
        data_obj: Angular spectrum analysis object.
        axis (matplotlib.axes.Axes, optional): Axis to plot on.
        hide_legend (bool): If True, do not display legend.
        display (bool): If True, show the plot.
        hide_labels (bool): If True, do not display labels.
        **options: Additional matplotlib options.

    Returns:
        None.
    """
    if axis is None:
        options.pop('ax', None)
        fig = plt.figure(**options)
        axis = fig.add_subplot(1, 1, 1)
    
    # Apply Seaborn styling
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Get properties
    properties = data_obj.get_attrs()
    triplet_list = properties['species_pairs']  # Triplets stored as species_pairs
    
    # Generate color palette
    palette = sns.color_palette("tab10", len(triplet_list))
    line_handles = []
    
    # Plot each triplet
    for idx, (atom1, atom2, atom3) in enumerate(triplet_list):
        try:
            # Get angular data
            ang_data = data_obj.get_array(f'aspec_{atom1}_{atom2}_{atom3}')
            angle_values = data_obj.get_array(f'angles_{atom1}_{atom2}_{atom3}')
            
            # Create label if not hidden
            label = None if hide_labels else f'{atom1}-{atom2}-{atom3}'
            
            # Plot with Seaborn
            line = sns.lineplot(x=angle_values, y=ang_data, ax=axis, 
                              label=label, color=palette[idx], linewidth=2.5)
            line_handles.append(line)
            
        except KeyError:
            print(f'Warning: Angular spectrum for {atom1}-{atom2}-{atom3} was not calculated, skipping')
            continue
    
    # Set axis labels and appearance
    axis.set_xlabel('Angle [degrees]', fontweight='bold', fontsize=14)
    axis.set_ylabel('Probability Density', fontweight='bold', fontsize=14)
    axis.set_title('Angular Distribution', fontsize=16)
    
    # Add legend if not hidden
    if not hide_legend:
        axis.legend(handles=line_handles, frameon=True, edgecolor='black', 
                  fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Clean up appearance
    sns.despine()
    
    # Display if requested
    if display:
        plt.tight_layout()
        plt.show()

def parse_cli_args():
    """
    Parse command-line arguments for plotting utilities.

    Returns:
        argparse.Namespace: Parsed arguments for plotting.
    """
    parser = argparse.ArgumentParser(
        description="Create publication-quality plots for molecular dynamics analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Select plot type: 'msd', 'van_hove', or 'ngp'")

    # MSD Subcommand
    msd_parser = subparsers.add_parser("msd", help="Generate MSD plots from JSON data.")
    msd_parser.add_argument("--data-path", required=True,
                            help="Path to the JSON file with simulation data.")
    msd_parser.add_argument("--save-path", default=None,
                            help="File path to save the plot (JPEG format).")
    msd_parser.add_argument("--plot-data", action="append", nargs="+", required=True,
                            help="Data to plot, format: --plot-data TEMP ELEMENT DATA_TYPE DIRECTION")
    msd_parser.add_argument("--first-time", nargs="+", type=float, default=[0],
                            help="Start time(s) for plotting window(s) in ps.")
    msd_parser.add_argument("--last-time", nargs="+", type=float, default=[10],
                            help="End time(s) for plotting window(s) in ps.")
    msd_parser.add_argument("--is-lammps", action="store_true",
                            help="Use LAMMPS data format instead of QE.")

    # Van Hove Subcommand
    vh_parser = subparsers.add_parser("van_hove", help="Generate Van Hove correlation plots.")
    vh_parser.add_argument("--data-path", required=True,
                           help="Path to the JSON file with Van Hove data.")
    vh_parser.add_argument("--save-path", default=None,
                           help="Base file path for saving plots (JPEG format).")
    vh_parser.add_argument("--figsize", nargs=2, type=float, default=[10, 8],
                           help="Figure dimensions (width height) in inches.")
    vh_parser.add_argument("--plot-data", action="append", nargs="+", required=True,
                           help="Data to plot, format: --plot-data TEMP ELEMENT TYPE")
    vh_parser.add_argument("--colormap", default="tab10",
                           help="Matplotlib colormap for heatmap (e.g., tab10, plasma, jet)")

    # NGP Subcommand
    ngp_parser = subparsers.add_parser("ngp", help="Generate Non-Gaussian Parameter plots.")
    ngp_parser.add_argument("--data-path", required=True,
                            help="Path to the JSON file with NGP data.")
    ngp_parser.add_argument("--save-path", default="NGP.jpg",
                            help="File path to save the plot (JPEG format).")
    ngp_parser.add_argument("--plot-data", action="append", nargs=3, required=True,
                            help="Data to plot, format: --plot-data TEMP ELEMENT DIRECTION")
    ngp_parser.add_argument("--first-time", type=float, default=0.1,
                            help="Start time for plotting window in ps.")
    ngp_parser.add_argument("--last-time", type=float, default=100,
                            help="End time for plotting window in ps.")
    ngp_parser.add_argument("--log-scale", action="store_true",
                            help="Use logarithmic scale for time axis.")

    return parser.parse_args()

def int_or_float(val):
    """
    Convert string to int if possible, else float.

    Args:
        val (str): Value to convert.

    Returns:
        int or float: Converted value.
    """
    try:
        return int(val)
    except ValueError:
        return float(val)

def main():
    """
    Main function for command-line plotting interface.

    Parses CLI arguments and dispatches to the appropriate plotting function.

    Returns:
        None.
    """
    args = parse_cli_args()
    
    # Apply global Seaborn styling
    sns.set(style="whitegrid", font_scale=1.2)
    
    if args.command == "msd":
        # Process MSD plotting
        final_plot_data = []
        for item in args.plot_data:
            converted = [int_or_float(item[0]), item[1], item[2], item[3]]
            final_plot_data.append(converted)

        msd_plot(
            data_path=args.data_path,
            Plot_data=final_plot_data,
            First_time=args.first_time[0],
            Last_time=args.last_time[0],
            save_path=args.save_path,
            is_lammps=args.is_lammps
        )

    elif args.command == "van_hove":
        # Process Van Hove plotting
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
    
    elif args.command == "ngp":
        # Process NGP plotting
        final_plot_data = []
        for item in args.plot_data:
            temp, element, direction = item
            final_plot_data.append((int_or_float(temp), element, direction))
            
        ngp_plot(
            data_source=args.data_path,
            plot_data=final_plot_data,
            first_time=args.first_time,
            last_time=args.last_time,
            save_path=args.save_path
        )

if __name__ == "__main__":
    main()