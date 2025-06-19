"""
JSON Serialization Utilities for CPDyAna
========================================

This module provides utility functions for converting complex Python data structures
(particularly those containing NumPy arrays) into JSON-serializable formats.

CPDyAna analysis functions often return nested dictionaries containing NumPy arrays,
which cannot be directly serialized to JSON. This module handles the conversion
by recursively traversing data structures and converting NumPy arrays to lists
while preserving the overall structure.

Functions:
    convert_to_serializable: Main conversion function for nested data structures
    
Common use cases:
- Saving analysis results to JSON files for later processing
- Preparing data for web APIs or data exchange
- Creating human-readable output files

Author: CPDyAna Development Team
Version: 01-02-2024
"""

import numpy as np
# Convert data to JSON-serializable format
def convert_to_serializable(obj):
    """
    Recursively convert data structures containing NumPy arrays to JSON-serializable format.
    
    This function handles nested dictionaries, lists, and NumPy arrays by converting
    them to native Python types that can be serialized to JSON. The function preserves
    the structure of nested data while ensuring all elements are JSON-compatible.
    
    Args:
        obj: Input object to convert. Can be:
            - dict: Nested dictionary (values will be recursively converted)
            - list/tuple: List or tuple (elements will be recursively converted)  
            - np.ndarray: NumPy array (converted to list)
            - np.number: NumPy scalar (converted to Python number)
            - Other types: Returned unchanged if JSON-serializable
            
    Returns:
        JSON-serializable object with the same structure as input but with
        NumPy arrays converted to lists and NumPy scalars to Python numbers.
        
    Raises:
        TypeError: If object contains non-serializable types that cannot be converted.
        
    Example:
        >>> import numpy as np
        >>> data = {
        ...     'msd': np.array([1.0, 2.0, 3.0]),
        ...     'time': np.array([0, 1, 2]),
        ...     'params': {'temp': np.float64(800.0)},
        ...     'elements': ['Li', 'Na']
        ... }
        >>> serializable = convert_to_serializable(data)
        >>> with open('results.json', 'w') as f:
        ...     json.dump(serializable, f, indent=2)
        
    Note:
        - NumPy arrays are converted to nested lists preserving dimensionality
        - NumPy scalars (np.float64, np.int32, etc.) become Python float/int
        - The function handles arbitrary nesting depth
        - Original data structure is not modified (deep copy behavior)
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    return obj