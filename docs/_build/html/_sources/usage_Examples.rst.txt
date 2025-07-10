Usage Examples
==============

This section provides comprehensive examples for using CPDyAna to analyze molecular dynamics trajectories. Examples cover both basic usage and advanced analysis workflows.

Command-Line Interface Examples
-------------------------------

Mean Square Displacement (MSD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CPDyAna msd \
      --data-dir /path/to/md_data \
      --temperature 800 \
      --diffusing-elements Li \
      --diffusivity-direction-choices X Y Z XYZ \
      --diffusivity-choices Tracer Collective \
      --initial-time 5.0 \
      --final-time 200.0 \
      --initial-slope-time 10.0 \
      --final-slope-time 150.0 \
      --block 500 \
      --step-skip 100 \
      --save-path Li_msd_800K.png

Non-Gaussian Parameter (NGP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CPDyAna ngp \
      --data-dir /path/to/md_data \
      --temperature 800 \
      --diffusing-elements Li \
      --diffusivity-direction-choices XYZ \
      --initial-time 2.0 \
      --final-time 200.0 \
      --save-path Li_ngp_800K.png

Van Hove Correlation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CPDyAna vh \
      --data-dir /path/to/md_data \
      --temperature 800 \
      --diffusing-elements Li \
      --correlation Self Distinct \
      --rmax 15.0 \
      --sigma 0.1 \
      --ngrid 151 \
      --step-skip 100

Radial Distribution Functions (RDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CPDyAna rdf \
      --data-dir /path/to/md_data \
      --central-atom Li Al \
      --pair-atoms Li Al P S O \
      --rmax 12.0 \
      --ngrid 1200 \
      --time-after-start 50.0 \
      --num-frames 100

Ionic Density Mapping
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CPDyAna ionic-density \
      --data-dir /path/to/md_data \
      --element Li \
      --sigma 0.4 \
      --density 0.15 \
      --time-after-start 20.0 \
      --num-frames 500

Velocity Autocorrelation Functions (VAF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CPDyAna vaf \
      --data-dir /path/to/md_data \
      --element Li Na \
      --start 10.0 \
      --nframes 2000 \
      --blocks 5 \
      --t-end-fit-ps 20.0

Vibrational Density of States (VDOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CPDyAna vdos \
      --data-dir /path/to/md_data \
      --elements Li Al P S O \
      --start 50.0 \
      --nframes 5000 \
      --stride 2

Python API Examples
-------------------

Direct Function Usage
~~~~~~~~~~~~~~~~~~~~~

Using CPDyAna functions directly in Python scripts:

.. code-block:: python

   from target.CPDyAna import Job
   from target import plotting as p
   import json

   # Define analysis parameters
   temperatures = [800.0]
   elements = ['Li']
   diffusivity_directions = ['XYZ']
   diffusivity_types = ['Tracer']
   correlations = ['Self']
   
   # File paths (modify for your data)
   pos_files = ['/path/to/data/md.pos']
   cel_files = ['/path/to/data/md.cel']
   evp_files = ['/path/to/data/md.evp']
   ion_files = ['/path/to/data/md.in']
   
   # Run analysis
   input_data, output_data = Job(
       temperature=temperatures,
       diffusing_elements=elements,
       diffusivity_direction_choices=diffusivity_directions,
       diffusivity_choices=diffusivity_types,
       correlation=correlations,
       pos_file=pos_files,
       cel_file=cel_files,
       evp_file=evp_files,
       ion_file=ion_files,
       Conv_factor=0.529177249,  # Bohr to Angstrom
       initial_time=5.0,
       final_time=200.0,
       initial_slope_time=10.0,
       final_slope_time=150.0,
       block=500,
       rmax=10.0,
       step_skip=100,
       sigma=0.1,
       ngrid=101,
       mode="msd"
   )
   
   # Save results
   with open('analysis_results.json', 'w') as f:
       json.dump(output_data, f, indent=2)
   
   # Generate plots
   plot_data = [[800.0, 'Li', 'Tracer', 'XYZ']]
   p.msd_plot(output_data, plot_data, 5.0, 200.0, save_path='custom_msd.png')

SAMOS Module Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Using SAMOS modules for advanced analysis:

.. code-block:: python

   from samos_modules.samos_analysis import util_msd, util_rdf_and_plot
   from samos_modules.samos_io import read_lammps_dump
   
   # Convert LAMMPS trajectory to ASE format
   traj = read_lammps_dump(
       filename='dump.lammpstrj',
       elements=['Li', 'Al', 'P', 'S', 'O'],
       timestep=0.001,  # ps
       save_extxyz=True,
       outfile='converted.extxyz'
   )
   
   # Calculate MSD using SAMOS
   util_msd(
       trajectory_path='converted.extxyz',
       stepsize=2,
       species=['Li'],
       plot=True,
       savefig='samos_msd.png',
       t_start_fit_ps=5,
       t_end_fit_ps=50,
       nblocks=4
   )
   
   # Calculate RDF using SAMOS
   util_rdf_and_plot(
       trajectory_path='converted.extxyz',
       radius=10.0,
       stepsize=1,
       bins=200,
       species_pairs=['Li-O', 'Li-Li', 'Al-O'],
       plot=True,
       savefig='samos_rdf.png'
   )

Batch Processing Workflows
---------------------------

Processing Multiple Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script for batch analysis of multiple MD runs:

.. code-block:: python

   import os
   import glob
   import subprocess
   
   # Directory containing multiple simulation folders
   base_dir = '/path/to/simulations'
   
   # Find all simulation directories
   sim_dirs = glob.glob(os.path.join(base_dir, 'sim_*'))
   
   for sim_dir in sim_dirs:
       print(f"Processing {sim_dir}")
       
       # Check for required files
       required_files = ['*.pos', '*.cel', '*.evp', '*.in']
       if all(glob.glob(os.path.join(sim_dir, pattern)) for pattern in required_files):
           
           # Run MSD analysis
           cmd = [
               'python', 'CPDyAna.py', 'msd',
               '--data-dir', sim_dir,
               '--temperature', '800',
               '--diffusing-elements', 'Li',
               '--save-path', f'{sim_dir}/msd_result.png'
           ]
           
           try:
               subprocess.run(cmd, check=True)
               print(f"  ✓ MSD analysis completed")
           except subprocess.CalledProcessError as e:
               print(f"  ✗ MSD analysis failed: {e}")
       else:
           print(f"  ✗ Missing required files")

Temperature Series Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzing diffusion across temperature range:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from target.CPDyAna import Job
   
   # Temperature range
   temperatures = np.linspace(600, 1200, 7)  # 600K to 1200K
   diffusion_coefficients = []
   
   for T in temperatures:
       # Run analysis for each temperature
       # (assuming files are named with temperature)
       pos_file = [f'md_{T}K.pos']
       cel_file = [f'md_{T}K.cel']
       evp_file = [f'md_{T}K.evp']
       ion_file = [f'md_{T}K.in']
       
       try:
           input_data, output_data = Job(
               temperature=[T],
               diffusing_elements=['Li'],
               # ... other parameters ...
               mode="msd"
           )
           
           # Extract diffusion coefficient
           D = output_data[(T, 'Li')]['msd_data']['Tracer']['XYZ']['diffusion_coeff']
           diffusion_coefficients.append(D)
           
       except Exception as e:
           print(f"Failed for T={T}K: {e}")
           diffusion_coefficients.append(np.nan)
   
   # Plot Arrhenius behavior
   plt.figure(figsize=(8, 6))
   plt.semilogy(1000/temperatures, diffusion_coefficients, 'o-')
   plt.xlabel('1000/T (1/K)')
   plt.ylabel('Diffusion Coefficient (cm²/s)')
   plt.title('Arrhenius Plot for Li Diffusion')
   plt.grid(True)
   plt.savefig('arrhenius_plot.png', dpi=300)

Advanced Analysis Examples
--------------------------

Custom Van Hove Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from target.correrelation_analysis import Van_Hove_self, Van_Hove_distinct
   import numpy as np

   # Load trajectory data (example structure)
   # pos_array: shape (n_frames, n_atoms, 3)
   # dt: time step array

   # Self Van Hove correlation
   dist_interval, reduced_nt, grt_self = Van_Hove_self(
       avg_step=50,
       dt=dt,
       rmax=15.0,
       step_skip=100,
       sigma=0.15,
       ngrid=151,
       moving_ion_pos=pos_array
   )

   # Plot self correlation
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 8))

   for i, t in enumerate(reduced_nt[::5]):  # Plot every 5th time
       ax.plot(dist_interval, grt_self[i], 
               label=f't = {t:.2f} ps', alpha=0.7)

   ax.set_xlabel('Distance (Å)')
   ax.set_ylabel('G_s(r,t)')
   ax.set_title('Self Van Hove Correlation Function')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.savefig('van_hove_self.png', dpi=300)

File Format Tips
----------------

Required File Structure
~~~~~~~~~~~~~~~~~~~~~~

For Quantum ESPRESSO trajectories, ensure your data directory contains:

.. code-block:: text

   data_directory/
   ├── simulation.pos    # Atomic positions
   ├── simulation.cel    # Unit cell parameters
   ├── simulation.evp    # Energy, volume, pressure
   └── simulation.in     # Ion definitions

For multiple temperatures or conditions:

.. code-block:: text

   data_directory/
   ├── md_600K.pos, md_600K.cel, md_600K.evp, md_600K.in
   ├── md_800K.pos, md_800K.cel, md_800K.evp, md_800K.in
   └── md_1000K.pos, md_1000K.cel, md_1000K.evp, md_1000K.in

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

For large trajectories, use these optimization strategies:

.. code-block:: bash

   # Use larger step_skip for faster analysis
   CPDyAna msd \
       --step-skip 1000 \
       --block 1000 \
       # ... other parameters

   # For SAMOS functions, use appropriate stepsize
   util_msd(
       trajectory_path='large_trajectory.extxyz',
       stepsize=10,  # Use every 10th frame
       # ... other parameters
   )