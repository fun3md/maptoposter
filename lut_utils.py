"""
lut_utils.py - Utility functions for handling Lookup Tables (LUTs) in the
calibration pipeline for map poster generation.

Provides:
- load_lut(path): Load a CSV LUT file and return a callable interpolation function.
- invert_lut(lut_function): Return an inverted version of the LUT.
- interpolate_lut(lut_function, x): Interpolate the LUT at point(s) x.
"""

import csv
import numpy as np
from scipy.interpolate import interp1d
from typing import Callable, Union

LUTData = Union[np.ndarray, list]

def load_lut(lut_path: str) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    Load a CSV LUT file.

    The CSV is expected to have two columns: input and output values.
    Returns a function that interpolates the LUT for given input values.
    """
    inputs, outputs = [], []
    with open(lut_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or len(row) < 2:
                continue
            inp = float(row[0])
            out = float(row[1])
            inputs.append(inp)
            outputs.append(out)
    # Ensure monotonic inputs for interpolation
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    # Create interpolation function
    interp_func = interp1d(inputs, outputs, kind='linear', fill_value='extrapolate')
    return interp_func

def invert_lut(lut_func: Callable) -> Callable:
    """
    Invert a LUT function.

    Given a LUT that maps input -> output, return a new function that maps
    output -> input using numerical inversion (assuming monotonic mapping).
    """
    # Sample the original LUT densely to build inversion mapping
    # This is a simple approach; for production, consider a more robust method.
    import numpy as np
    # Generate sample points
    sample_points = np.linspace(0, 1, 1000)
    sample_outputs = lut_func(sample_points)
    # Assume outputs are monotonic; invert via interpolation
    inv_func = interp1d(sample_outputs, sample_points, kind='linear', fill_value='extrapolate')
    return inv_func

def interpolate_lut(lut_func: Callable, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Interpolate using an already loaded LUT function.

    This is a thin wrapper around the LUT function to make the API explicit.
    """
    return lut_func(x)

# ----------------------------------------------------------------------
# Basic calibration pipeline skeleton
# ----------------------------------------------------------------------
class CalibrationPipeline:
    """
    Minimal calibration pipeline that loads LUTs and applies them in sequence.
    """
    def __init__(self, lut_paths: list):
        self.lut_functions = [load_lut(p) for p in lut_paths]

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply all loaded LUTs sequentially to the input data."""
        result = data
        for lut in self.lut_functions:
            result = lut(result)
        return result

    def invert_last(self) -> Callable:
        """Return an inverter for the last LUT in the pipeline."""
        return invert_lut(self.lut_functions[-1])