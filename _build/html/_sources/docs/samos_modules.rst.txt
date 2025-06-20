SAMOS Modules
=============

The SAMOS (Statistical Analysis of MOlecular Simulations) modules provide advanced trajectory analysis capabilities for molecular dynamics simulations. These modules offer high-performance analysis tools with support for multiple file formats and statistical methods.

Overview
--------

The SAMOS modules consist of:

* **Analysis Tools**: Core statistical analysis functions for MD trajectories
* **I/O Operations**: Reading and writing various trajectory formats (LAMMPS, XSF, etc.)
* **Plotting Utilities**: Visualization functions for analysis results
* **Trajectory Handling**: Advanced trajectory manipulation and processing
* **Utility Functions**: Helper functions for common MD analysis tasks

Core Analysis Module
--------------------

The main analysis module provides high-level functions for common MD analyses.

.. automodule:: samos_modules.samos_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

* :func:`samos_modules.samos_analysis.util_msd`: Mean Square Displacement calculation
* :func:`samos_modules.samos_analysis.util_rdf_and_plot`: Radial Distribution Function analysis
* :func:`samos_modules.samos_analysis.write_xsf_header`: XSF file writing utilities

I/O Operations Module
---------------------

Handles reading and writing of various trajectory and data formats.

.. automodule:: samos_modules.samos_io
   :members:
   :undoc-members:
   :show-inheritance:

Supported Formats
~~~~~~~~~~~~~~~~~

* **LAMMPS Dump Files**: :func:`samos_modules.samos_io.read_lammps_dump`
* **XSF Files**: :func:`samos_modules.samos_io.read_xsf`, :func:`samos_modules.samos_io.write_xsf`
* **Trajectory Formats**: Support for ASE-compatible formats

Plotting Module
---------------

Visualization and plotting utilities for analysis results.

.. automodule:: samos_modules.samos_plotting
   :members:
   :undoc-members:
   :show-inheritance:

Compiled Extensions
-------------------

The SAMOS modules include several compiled extensions for high-performance calculations:

* **gaussian_density.pyd**: Gaussian density calculations
* **mdutils.pyd**: MD utility functions
* **rdf.pyd**: Optimized RDF calculations

These compiled modules provide significant performance improvements for computationally intensive operations.

Usage Examples
--------------

Basic MSD Analysis with SAMOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from samos_modules.samos_analysis import util_msd
   
   # Calculate MSD for Li ions
   util_msd(
       trajectory_path='trajectory.extxyz',
       stepsize=1,
       species=['Li'],
       plot=True,
       savefig='msd_li.png',
       t_start_fit_ps=5,
       t_end_fit_ps=50
   )

RDF Calculation and Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from samos_modules.samos_analysis import util_rdf_and_plot
   
   # Compute and plot RDF for Li-O pairs
   util_rdf_and_plot(
       trajectory_path='trajectory.extxyz',
       radius=8.0,
       stepsize=1,
       bins=200,
       species_pairs=['Li-O', 'Li-Li'],
       plot=True,
       savefig='rdf_analysis.png'
   )

Reading LAMMPS Trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from samos_modules.samos_io import read_lammps_dump
   
   # Read LAMMPS dump file and convert to trajectory
   traj = read_lammps_dump(
       filename='dump.lammpstrj',
       elements=['Li', 'Al', 'P', 'S'],
       timestep=0.001,  # ps
       save_extxyz=True,
       outfile='converted.extxyz'
   )