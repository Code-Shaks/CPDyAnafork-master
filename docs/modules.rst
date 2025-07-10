Usage Examples
==============

Basic MSD Analysis
------------------

To perform mean square displacement (MSD) analysis on a LAMMPS trajectory:

.. code-block:: python

   from target.input_reader import read_lammps_trajectory
   from target.calculations import msd_tracer

   # Read the full trajectory into memory
   trajectory, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array = read_lammps_trajectory(
       lammps_file='dump.lammpstrj',
       elements=['Li', 'O'],
       timestep=0.001
   )

   # Calculate MSD for all ions
   msd = msd_tracer(trajectory, n_frames)

Non-Gaussian Parameter (NGP) Analysis
--------------------------------------

.. code-block:: python

   from target.input_reader import read_lammps_trajectory
   from target.calculations import calculate_ngp

   trajectory, n_frames, dt_full, t_full, cell_param_full, thermo_data, volumes, inp_array = read_lammps_trajectory(
       lammps_file='dump.lammpstrj',
       elements=['Li', 'O'],
       timestep=0.001
   )

   # Calculate NGP for tracer diffusion
   ngp = calculate_ngp(['Li'], ['XYZ'], trajectory, trajectory, inp_array, dt_full)

Van Hove Correlation Function
-----------------------------

.. code-block:: python

   from target.correrelation_analysis import Van_Hove_self, Van_Hove_distinct

   # pos_array: shape (n_frames, n_atoms, 3)
   # dt: time step array

   dist_interval, reduced_nt, grt_self = Van_Hove_self(
       avg_step=50,
       dt=dt,
       rmax=15.0,
       step_skip=100,
       sigma=0.15,
       ngrid=151,
       moving_ion_pos=pos_array
   )

Radial Distribution Function (RDF)
----------------------------------

.. code-block:: python

   from target.compute_rdf import compute_rdf

   # Compute RDF for Li-O pairs
   rdf_result = compute_rdf(
       trajectory=trajectory,
       central_atoms=['Li'],
       pair_atoms=['O'],
       rmax=10.0,
       ngrid=200
   )

Velocity Autocorrelation Function (VAF)
---------------------------------------

.. code-block:: python

   from target.compute_vaf import compute_vaf

   vaf_result = compute_vaf(
       trajectory=trajectory,
       element='Li',
       nframes=1000,
       blocks=5
   )

Vibrational Density of States (VDOS)
------------------------------------

.. code-block:: python

   from target.vdos import compute_vdos

   vdos_result = compute_vdos(
       trajectory=trajectory,
       elements=['Li', 'O'],
       start=50.0,
       nframes=5000,
       stride=2
   )

Ionic Density Mapping
---------------------

.. code-block:: python

   from target.probability_density import compute_density

   density_result = compute_density(
       trajectory=trajectory,
       element='Li',
       sigma=0.4,
       density=0.15,
       time_after_start=20.0,
       num_frames=500
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
           subprocess.run(cmd)