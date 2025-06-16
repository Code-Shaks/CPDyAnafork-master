import numpy as np

def convert_to_serializable(obj):
    """Convert NumPy arrays,  lists, and nested dictionaries to JSON serializable objects."""
    if isinstance(obj, np.ndarray):  # Check if the object is a NumPy array.
        return obj.tolist()  # Convert the NumPy array to a list.
    elif isinstance(obj, list):  # Check if the object is a list.
        return [convert_to_serializable(item) for item in obj]  # Recursively convert each item in the list.
    elif isinstance(obj, dict):  # Check if the object is a dictionary.
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}  # Recursively convert each key-value pair in the dictionary.
    
    return obj  # Return the object if it is neither a NumPy array, list, nor dictionary.

