target package
==============

The `target` package provides tools for analyzing molecular dynamics simulations, focusing on the calculation of transport properties such as ionic conductivity, diffusion coefficients, vibrational density of states (VDOS), and correlation functions. It includes modules for data processing, plotting, input parsing, and more.

Usage Example
-------------

To analyze a trajectory and compute transport properties, you might use:

.. code-block:: python

    from target import calculations, input_reader, data_processing, Plotting

    # Read input files
    species = input_reader.read_ion_file('LiAlPS.in')
    positions = input_reader.read_pos('LiAlPS.pos', natoms=len(species))
    cell = input_reader.read_cel('LiAlPS.cel')

    # Preprocess data
    processed_data = data_processing.clean_and_slice(positions)

    # Calculate diffusion coefficient
    D = calculations.compute_diffusion_coefficient(processed_data)

    # Plot results
    Plotting.plot_diffusion(D)

Submodules
----------

target.Plotting module
----------------------
**Description:**  
Provides plotting utilities for visualizing simulation results, including time series, correlation functions, and transport property trends.

**Features:**  
- Customizable plotting functions for various properties  
- Support for publication-quality figures  
- Integration with analysis modules for seamless visualization  

**Usage Example:**

.. code-block:: python

    from target.Plotting import plot_vacf
    plot_vacf(vacf_data, time_axis)

.. automodule:: target.Plotting
   :members:
   :undoc-members:
   :show-inheritance:

target.calculations module
--------------------------
**Description:**  
Contains core routines for calculating transport properties from simulation data.

**Features:**  
- Calculation of diffusion coefficients  
- Ionic conductivity estimation  
- Mean squared displacement (MSD) and velocity autocorrelation functions (VACF)  
- Support for multi-component systems  

**Usage Example:**

.. code-block:: python

    from target.calculations import compute_diffusion_coefficient
    D = compute_diffusion_coefficient(msd_data, time_axis)

.. automodule:: target.calculations
   :members:
   :undoc-members:
   :show-inheritance:

target.const module
-------------------
**Description:**  
Defines physical constants and unit conversion factors used throughout the package.

**Features:**  
- SI and atomic unit constants  
- Conversion utilities for energy, length, and time  

**Usage Example:**

.. code-block:: python

    from target.const import BOHR2ANG
    angstrom = bohr_value * BOHR2ANG

.. automodule:: target.const
   :members:
   :undoc-members:
   :show-inheritance:

target.correrelation_analysis module
------------------------------------
**Description:**  
Implements correlation function analysis, essential for extracting transport properties from molecular dynamics trajectories.

**Features:**  
- Calculation of time correlation functions (e.g., VACF, MSD)  
- Tools for analyzing relaxation times and dynamic processes  

**Usage Example:**

.. code-block:: python

    from target.correrelation_analysis import compute_vacf
    vacf = compute_vacf(velocity_data)

.. automodule:: target.correrelation_analysis
   :members:
   :undoc-members:
   :show-inheritance:

target.data_processing module
-----------------------------
**Description:**  
Handles preprocessing and manipulation of raw simulation data for further analysis.

**Features:**  
- Data cleaning and normalization  
- Trajectory slicing and selection  
- Preparation of data for correlation and transport analysis  

**Usage Example:**

.. code-block:: python

    from target.data_processing import clean_and_slice
    processed = clean_and_slice(raw_data)

.. automodule:: target.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

target.input_reader module
--------------------------
**Description:**  
Parses input files from various simulation packages and converts them into formats compatible with the analysis routines.

**Features:**  
- Support for multiple input formats (.in, .pos, .cel, etc.)  
- Extraction of atomic species, positions, and cell parameters  
- Error handling for incomplete or inconsistent input files  

**Usage Example:**

.. code-block:: python

    from target.input_reader import read_ion_file
    species = read_ion_file('LiAlPS.in')

.. automodule:: target.input_reader
   :members:
   :undoc-members:
   :show-inheritance:

target.prcss module
-------------------
**Description:**  
Provides additional processing utilities for simulation data, including filtering and transformation functions.

**Features:**  
- Data smoothing and filtering  
- Transformation of trajectory data  
- Auxiliary tools for advanced analysis  

**Usage Example:**

.. code-block:: python

    from target.prcss import smooth_data
    smoothed = smooth_data(raw_signal)

.. automodule:: target.prcss
   :members:
   :undoc-members:
   :show-inheritance:

target.serilalizable module
---------------------------
**Description:**  
Implements serialization and deserialization of data objects for efficient storage and retrieval.

**Features:**  
- Save and load analysis results  
- Support for various serialization formats (e.g., JSON, pickle)  
- Ensures reproducibility and portability of results  

**Usage Example:**

.. code-block:: python

    from target.serilalizable import save_results, load_results
    save_results('output.json', results)
    loaded = load_results('output.json')

.. automodule:: target.serilalizable
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: target
   :members:
   :undoc-members:
   :show-inheritance:
