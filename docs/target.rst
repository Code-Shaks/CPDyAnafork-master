Target Package
==============

The `target` package contains the core CPDyAna analysis functionality for molecular dynamics trajectory analysis. It provides a comprehensive suite of tools for calculating diffusion properties, correlation functions, and structural analysis.

Main Analysis Driver
--------------------

The main CPDyAna CLI interface and job orchestration.

.. automodule:: target.CPDyAna
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

* :func:`target.CPDyAna.main`: Main CLI entry point
* :func:`target.CPDyAna.Job`: Core analysis job orchestrator
* :func:`target.CPDyAna.parser`: Command-line argument parser

Data Processing
---------------

Utilities for processing and segmenting trajectory data.

.. automodule:: target.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

LAMMPS Data Processing
----------------------

Specialized functions for LAMMPS trajectory data.

.. automodule:: target.data_processing_lammps
   :members:
   :undoc-members:
   :show-inheritance:

Input/Output Operations
-----------------------

Reading and parsing of trajectory files and I/O utilities, including BOMD and LAMMPS.

.. automodule:: target.input_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: target.io
   :members:
   :undoc-members:
   :show-inheritance:

Supported File Types
~~~~~~~~~~~~~~~~~~~~

* **.pos files**: Position trajectories (Quantum ESPRESSO, etc.)
* **.cel files**: Unit cell parameters
* **.evp files**: Energy, volume, pressure data
* **.in files**: Ion definitions and atomic information
* **LAMMPS dump files**: .lammpstrj, .dump
* **BOMD trajectory files**: .trj (Born-Oppenheimer MD)
* **XSF files**: For visualization
* **Extended XYZ files**: .extxyz

BOMD Support
~~~~~~~~~~~~

CPDyAna supports Born-Oppenheimer Molecular Dynamics (BOMD) analysis via:

* :func:`target.input_reader.read_bomd_trajectory` — Read and parse BOMD .trj files
* BOMD-specific CLI arguments: `--bomd-elements`, `--bomd-timestep`, etc.

Core Calculations
-----------------

Main calculation routines for MSD, NGP, and diffusion analysis.

.. automodule:: target.calculations
   :members:
   :undoc-members:
   :show-inheritance:

Analysis Types
~~~~~~~~~~~~~~

* **Mean Square Displacement**: Tracer and collective diffusivity (LAMMPS and BOMD)
* **Non-Gaussian Parameter**: Dynamical heterogeneity
* **Diffusion Coefficients**: Multi-directional analysis (X, Y, Z, XY, XZ, YZ, XYZ)
* **Statistical Analysis**: Block averaging and error estimation

Correlation Analysis
--------------------

Van Hove correlation functions and related analysis.

.. automodule:: target.correrelation_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Correlation Types
~~~~~~~~~~~~~~~~~

* **Self Correlation**: Single particle dynamics (:func:`target.correrelation_analysis.Van_Hove_self`)
* **Distinct Correlation**: Pair dynamics (:func:`target.correrelation_analysis.Van_Hove_distinct`)

RDF Computation
---------------

Radial distribution function calculations (LAMMPS, BOMD, QE).

.. automodule:: target.compute_rdf
   :members:
   :undoc-members:
   :show-inheritance:

VAF Computation
---------------

Velocity autocorrelation function analysis.

.. automodule:: target.compute_vaf
   :members:
   :undoc-members:
   :show-inheritance:

VDOS Analysis
-------------

Vibrational Density of States calculations.

.. automodule:: target.vdos
   :members:
   :undoc-members:
   :show-inheritance:

Plotting and Visualization
--------------------------

Plotting utilities for analysis results.

.. automodule:: target.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Probability Density Analysis
----------------------------

3D ionic density mapping and spatial analysis.

.. automodule:: target.probability_density
   :members:
   :undoc-members:
   :show-inheritance:

Trajectory Utilities
--------------------

Trajectory object and helpers.

.. automodule:: target.trajectory
   :members:
   :undoc-members:
   :show-inheritance:

General Utilities
-----------------

General helper functions.

.. automodule:: target.utilities
   :members:
   :undoc-members:
   :show-inheritance:

Analysis Helpers
----------------

Additional analysis routines.

.. automodule:: target.analysis
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
-------------

Preprocessing and filtering routines.

.. automodule:: target.prcss
   :members:
   :undoc-members:
   :show-inheritance:

Utility Modules
---------------

JSON Serialization
~~~~~~~~~~~~~~~~~~

.. automodule:: target.json_serializable
   :members:
   :undoc-members:
   :show-inheritance:

Constants
~~~~~~~~~

.. automodule:: target.const
   :members:
   :undoc-members:
   :show-inheritance: