# -*- coding: utf-8 -*-

"""
Utility Module for SAMOS Package
================================

This module provides essential utility functions, classes, and constants
for the SAMOS (Statistical Analysis of MOlecular Simulations) package.
It includes:

- Base classes for array data handling with attribute storage
- Unit conversion constants for molecular dynamics
- Color mapping utilities for visualization
- IO helpers and terminal display utilities

The core `AttributedArray` class serves as a foundation for data storage
in SAMOS, providing a flexible container for numerical arrays with
associated metadata and serialization capabilities.

Constants:
    kB: Boltzmann constant in J/K
    kB_ev: Boltzmann constant in eV/K
    kB_au: Boltzmann constant in atomic units
    amu_kg: Atomic mass unit in kg
    bohr_to_ang: Bohr radius to Angstrom conversion factor

Classes:
    AttributedArray: Array container with metadata and serialization
    InputError: Custom exception for input validation failures

Functions:
    get_color: Get RGB colors for chemical elements
    get_terminal_width: Get current terminal dimensions
    
Author: SAMOS Development Team
Version: 01-02-2024
"""

from json import dumps
import numpy as np
import shutil
import os

from ase.data.colors import jmol_colors, cpk_colors
from ase.data import atomic_numbers

kB = 1.38064852e-23
kB_ev = 8.6173303e-5
kB_au = 3.166810800209422e-06
amu_kg = 1.660539040e-27
bohr_to_ang = 0.52917721092


class AttributedArray(object):
    """
    Flexible array container with attribute storage and serialization.
    
    This class provides a foundation for storing numerical data arrays with
    associated metadata in a serializable format. It forms the base class
    for trajectory and analysis data structures in SAMOS.
    
    The class supports:
    - Named array storage with validation
    - JSON-serializable attribute storage
    - Array rescaling operations
    - File-based serialization and loading
    
    Attributes:
        _arrays (dict): Dictionary of stored arrays
        _attrs (dict): Dictionary of JSON-serializable attributes
        _nstep (int): Number of steps in time-series data
        
    Example:
        >>> # Create container with initial attributes
        >>> data = AttributedArray(temperature=300, timestep=1.0)
        >>> 
        >>> # Add arrays
        >>> positions = np.random.random((100, 50, 3))  # 100 steps, 50 atoms, 3D
        >>> data.set_array('positions', positions, check_nstep=True)
        >>> 
        >>> # Add metadata
        >>> data.set_attr('simulation_type', 'NVT')
        >>> 
        >>> # Save and load
        >>> data.save('simulation_data.tar.gz')
        >>> loaded_data = AttributedArray.load_file('simulation_data.tar.gz')
    """
    _ATTRIBUTE_FILENAME = 'attributes.json'

    def __init__(self, **kwargs):
        self._arrays = {}
        self._attrs = {}

        self._nstep = None
        for key, val in list(kwargs.items()):
            getattr(self, 'set_{}'.format(key))(val)

    def set_array(
        self, name, array, check_existing=False,
        check_nstep=False, check_nat=False,
            wanted_shape_len=None, wanted_shape_1=None,
            wanted_shape_2=None):
        """
        Set a named array with validation options.
        
        This method stores a numpy array with a specified name and performs
        optional validation on its shape and compatibility with existing arrays.
        
        Args:
            name (str): Name to reference the array
            array (numpy.ndarray): Array data to store (or convertible to numpy array)
            check_existing (bool): If True, raise error if array name already exists
            check_nstep (bool): If True, validate first dimension matches other arrays
            check_nat (bool/int): If True or int, validate second dimension matches nat
            wanted_shape_len (int, optional): Validate array has this number of dimensions
            wanted_shape_1 (int, optional): Validate array's second dimension is this size
            wanted_shape_2 (int, optional): Validate array's third dimension is this size
            
        Raises:
            TypeError: If name is not a string or array has wrong dimensions
            ValueError: If array name exists and check_existing=True
            ValueError: If array shape doesn't match existing arrays
            IndexError: If array dimensions don't match wanted_shape parameters
            
        Example:
            >>> # Store positions array with validation
            >>> positions = np.zeros((100, 50, 3))  # 100 steps, 50 atoms, 3D positions
            >>> data.set_array('positions', positions,
            ...                check_nstep=True,
            ...                check_nat=50,
            ...                wanted_shape_len=3,
            ...                wanted_shape_2=3)
        """
        # First, I call np.array to ensure it's a valid array
        array = np.array(array)
        if not isinstance(name, str):
            raise TypeError('Name has to be a string')
        if check_existing:
            if name in list(self._arrays.keys()):
                raise ValueError('Name {} already exists'.formamt(name))
        if wanted_shape_len:
            if len(array.shape) != wanted_shape_len:
                raise TypeError(
                    f"array {name} is of wrong type, has to be of "
                    f"dimension {wanted_shape_len}")
        if wanted_shape_1:
            if array.shape[1] != wanted_shape_1:
                raise IndexError(
                    f"1st dimension of array {name} has to "
                    f"be {wanted_shape_1}")
        if wanted_shape_2:
            if array.shape[2] != wanted_shape_2:
                raise IndexError(
                    f"2nd dimension of array {name} has "
                    f"to be {wanted_shape_2}")
        if check_nstep:
            if self._nstep is None:
                self._nstep = array.shape[0]
            elif self._nstep != array.shape[0]:
                raise ValueError(
                    'Number of steps in array {} ({}) is not '
                    'compliant with number of steps in previous '
                    'arrays ({})'.format(name, array.shape[0],
                                         self._nstep))
        if check_nat and len(array.shape) > 2:
            if not isinstance(check_nat, int):
                raise TypeError(
                    'If check_nat is not False, it has to be an integer')
            if array.shape[1] != check_nat:
                raise ValueError(
                    'Second dimension of array does not '
                    'match the number of atoms')
        self._arrays[name] = array

    def __contains__(self, arrayname):
        return arrayname in self._arrays

    @property
    def nstep(self):
        """
        :returns: The number of trajectory steps
        :raises: ValueError if no unique number of steps can be determined.
        """
        if self._nstep is None:
            raise ValueError('Number of steps has not been set')
        return self._nstep

    def get_array(self, name):
        """
        Retrieve a stored array by name.
        
        Args:
            name (str): Name of the array to retrieve
            
        Returns:
            numpy.ndarray: The requested array
            
        Raises:
            KeyError: If no array with the given name exists
            
        Example:
            >>> # Get stored positions array
            >>> try:
            ...     positions = data.get_array('positions')
            ...     print(f"Got positions with shape {positions.shape}")
            ... except KeyError:
            ...     print("Positions not found")
        """
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(
                'An array with that name ( {} ) has not '
                'been set.'.format(name))

    def get_arraynames(self):
        """
        Get a sorted list of all stored array names.
        
        Returns:
            list: Sorted list of array names
            
        Example:
            >>> # List all available arrays
            >>> array_names = data.get_arraynames()
            >>> print(f"Available arrays: {', '.join(array_names)}")
        """
        return sorted(self._arrays.keys())

    def get_attrs(self):
        """
        Get the complete dictionary of attributes.
        
        Returns:
            dict: Dictionary of all stored attributes
            
        Example:
            >>> # Get all metadata
            >>> metadata = data.get_attrs()
            >>> for key, value in metadata.items():
            ...     print(f"{key}: {value}")
        """
        return self._attrs

    def get_attr(self, key):
        """
        Retrieve a stored attribute by key.
        
        Args:
            key (str): Name of the attribute to retrieve
            
        Returns:
            Any: The stored attribute value
            
        Raises:
            KeyError: If the attribute doesn't exist
            
        Example:
            >>> # Access stored temperature value
            >>> try:
            ...     temperature = data.get_attr('temperature')
            ...     print(f"Temperature: {temperature} K")
            ... except KeyError:
            ...     print("Temperature not set")
        """
        return self._attrs[key]

    def set_attr(self, key, value):
        """
        Set a JSON-serializable attribute.
        
        This method stores metadata as key-value pairs. All values must be
        JSON-serializable for compatibility with file storage.
        
        Args:
            key (str): Attribute name
            value: Attribute value (must be JSON-serializable)
            
        Raises:
            TypeError: If the value cannot be serialized to JSON
            
        Example:
            >>> # Store simulation parameters
            >>> data.set_attr('temperature', 300)
            >>> data.set_attr('ensemble', 'NVT')
            >>> data.set_attr('elements', ['Li', 'P', 'S'])
        """
        # Testing whether this is valid:
        dumps(value)
        self._attrs[key] = value

    def rescale_array(self, arrayname, value):
        """
        Rescale a stored array by a constant factor.
        
        This method multiplies all values in an array by a given factor,
        useful for unit conversions or normalization.
        
        Args:
            arrayname (str): Name of the array to rescale
            value (float): Scaling factor
            
        Raises:
            KeyError: If the array doesn't exist
            
        Example:
            >>> # Convert velocities from Å/fs to Å/ps
            >>> data.rescale_array('velocities', 1000.0)
        """
        self._arrays[arrayname] *= float(value)

    def save(self, filename):
        """
        Save the AttributedArray instance to a compressed tar file.
        
        This method serializes both the numerical arrays and attributes
        to a single compressed file that can later be loaded with load_file().
        
        Args:
            filename (str): Output filename (typically with .tar.gz extension)
            
        Example:
            >>> # Save data to file
            >>> data.save('simulation_results.tar.gz')
        """
        import tarfile
        import tempfile
        from inspect import getmembers, ismethod

        temp_folder = tempfile.mkdtemp()
        for funcname, func in getmembers(self, predicate=ismethod):
            if funcname.startswith('_save_'):
                func(temp_folder)

        with tarfile.open(filename, 'w:gz',
                          format=tarfile.PAX_FORMAT) as tar:
            tar.add(temp_folder, arcname='')

    def _save_arrays(self, folder_name):
        from os.path import join
        for arrayname, array in list(self._arrays.items()):
            np.save(join(folder_name, '{}.npy'.format(arrayname)), array)

    def remove_array(self, arrayname):
        """
    Remove a stored array.
    
    Args:
        arrayname (str): Name of the array to remove
        
    Raises:
        KeyError: If the array doesn't exist
        
    Example:
        >>> # Remove temporary array
        >>> data.remove_array('temp_calculations')
    """
        if arrayname not in self._arrays:
            raise KeyError(f"{arrayname} is not one of arrays")
        del self._arrays[arrayname]

    def _save_attributes(self, folder_name):
        from os.path import join
        import json

        with open(join(folder_name, self._ATTRIBUTE_FILENAME), 'w') as f:
            json.dump(self._attrs, f)

    @classmethod
    def load_file(cls, filename):
        """
        Load an AttributedArray instance from a saved file.
        
        This class method recreates an AttributedArray instance from a file
        previously created with the save() method.
        
        Args:
            filename (str): Path to the saved file (.tar.gz)
            
        Returns:
            AttributedArray: A new instance with loaded data
            
        Raises:
            Exception: If the file cannot be loaded or has invalid format
            
        Example:
            >>> # Load data from file
            >>> loaded_data = AttributedArray.load_file('simulation_results.tar.gz')
            >>> print(f"Loaded {len(loaded_data.get_arraynames())} arrays")
        """
        import tarfile
        import tempfile
        import json
        import os
        from os.path import join
        temp_folder = tempfile.mkdtemp()

        try:
            with tarfile.open(filename, 'r:gz',
                              format=tarfile.PAX_FORMAT) as tar:
                tar.extractall(temp_folder)

            files_in_tar = set(os.listdir(temp_folder))

            with open(join(temp_folder, cls._ATTRIBUTE_FILENAME)) as f:
                attributes = json.load(f)
            files_in_tar.remove(cls._ATTRIBUTE_FILENAME)
            new = cls()
            for k, v in list(attributes.items()):
                new.set_attr(k, v)

            if cls._ATOMS_FILENAME in files_in_tar:
                from ase.io import read
                new.set_atoms(read(join(temp_folder, cls._ATOMS_FILENAME)))
                files_in_tar.remove(cls._ATOMS_FILENAME)

            for array_file in files_in_tar:
                if not array_file.endswith('.npy'):
                    raise Exception(
                        'Unrecognized file in trajectory export: {}'
                        ''.format(array_file))
                new.set_array(array_file.rstrip('.npy'), np.load(
                    join(temp_folder, array_file), mmap_mode='r'))
        except Exception as e:
            shutil.rmtree(temp_folder)
            raise e
        shutil.rmtree(temp_folder)
        return new

CUSTOM_COLORS = {
    'H': (0, 0, 0)
}

def get_color(chemical_symbol, scheme='jmol'):
    """
    Get RGB color tuple for a chemical element.
    
    This function returns the standard color for an element based on
    common visualization schemes like JMOL or CPK.
    
    Args:
        chemical_symbol (str): Chemical symbol (e.g., 'H', 'O', 'Fe')
        scheme (str): Color scheme to use:
            - 'jmol': JMOL visualization colors (default)
            - 'cpk': CPK color convention
            
    Returns:
        tuple: RGB color tuple with values from 0 to 1, or None if not found
        
    Raises:
        ValueError: If the scheme is not recognized
        
    Example:
        >>> # Get oxygen color in JMOL scheme
        >>> o_color = get_color('O')
        >>> print(f"RGB color for oxygen: {o_color}")
        >>> 
        >>> # Get carbon color in CPK scheme
        >>> c_color = get_color('C', scheme='cpk')
    """
    if chemical_symbol in CUSTOM_COLORS:
        return CUSTOM_COLORS[chemical_symbol]
    if chemical_symbol in atomic_numbers:
        if scheme == 'jmol':
            return jmol_colors[atomic_numbers[chemical_symbol]]
        elif scheme == 'cpk':
            return cpk_colors[atomic_numbers[chemical_symbol]]
        else:
            raise ValueError('Unknown scheme {}'.format(scheme))
    else:
        return None
    
class InputError(Exception):
    """
    Exception raised when input validation fails.
    
    This custom exception is used throughout SAMOS to indicate
    that user-provided input doesn't meet validation requirements.
    """
    pass

def get_terminal_width():
    """
    Get the current terminal width and height.
    
    This function attempts to determine the dimensions of the current
    terminal window using various methods, with fallbacks if detection fails.
    
    Returns:
        tuple: (width, height) in characters
        
    Example:
        >>> # Get terminal width for formatting output
        >>> width, height = get_terminal_width()
        >>> print(f"Terminal is {width} columns wide and {height} rows tall")
    """
    env = os.environ

    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            import struct
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
                                                 '1234'))
        except Exception:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except Exception:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        # Use get(key[, default]) instead of a try/catch
        # try:
        #    cr = (env['LINES'], env['COLUMNS'])
        # except:
        #    cr = (25, 80)
    # ~ return int(cr[1]), int(cr[0])
    return int(cr[1]), int(cr[0])

def apply_minimum_image(displacement, cell):
    """
    Apply the minimum image convention (MIC) to a displacement vector.
    Parameters:
    - displacement: np.ndarray of shape (..., 3), displacement vectors
    - cell: np.ndarray of shape (3, 3), simulation cell vectors
    Returns:
    - np.ndarray of same shape as displacement with MIC applied
    """
    inv_cell = np.linalg.inv(cell.T)
    fractional = np.dot(displacement, inv_cell)
    fractional -= np.round(fractional)
    return np.dot(fractional, cell.T)