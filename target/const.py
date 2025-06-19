"""
Physical Constants and Unit Conversions for CPDyAna
===================================================

This module defines fundamental physical constants and unit conversion factors
used throughout the CPDyAna molecular dynamics analysis package.

Constants are organized by category:
- Length conversions (Bohr, Angstrom, meters)
- Time conversions (atomic units, picoseconds, seconds) 
- Energy conversions (Hartree, eV, Joules)
- Diffusion coefficient conversions
- Physical constants (Boltzmann constant, electron charge, etc.)

All values are given in SI base units unless otherwise specified.

Usage:
    from target.const import BOHR_TO_ANGSTROM, KB_BOLTZMANN
    
Author: CPDyAna Development Team
Version: 01-02-2024
"""
# Constants for CPDyAna Molecular Dynamics Analysis
N_A = 6.022140857e+23 # Avogadro's number
e = 1.6021766208e-19 # Elementary charge
R = 8.3144598 # Universal gas constant
z = 1 # Charge number
k = 1.3806452e-23 # Boltzmann constant
Conv_factor = 0.529177249 # Bohr to Angstrom conversion
sigma = 0.1 # Lennard-Jones parameter
ngrid_lim = 101 # Grid limit for analysis
step_skip = 10000 # Step skip for analysis
