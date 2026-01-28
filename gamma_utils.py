"""Gamma utility functions for gamma correction, undo, and linearization."""

import numpy as np

def gamma_apply(data: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to the given data.

    Parameters
    ----------
    data : np.ndarray
        Input data. Can be any numeric dtype; if integer, values are assumed to be in [0, 255].
    gamma : float
        Gamma value > 0. The correction performed is ``output = input ** (1/gamma)``.

    Returns
    -------
    np.ndarray
        Gamma-corrected array with the same dtype as the input.
    """
    # Convert to float for processing
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / 255.0
    # Apply gamma transformation
    corrected = np.power(data, 1.0 / gamma)
    # Clip to valid range and convert back to original dtype
    corrected = np.clip(corrected, 0.0, 1.0)
    if np.issubdtype(data.dtype, np.integer):
        out = np.rint(corrected * 255).astype(data.dtype)
    else:
        out = corrected.astype(data.dtype)
    return out


def gamma_undo(gamma_corrected_data: np.ndarray, gamma: float) -> np.ndarray:
    """
    Reverse a gamma correction (apply the inverse gamma).

    Parameters
    ----------
    gamma_corrected_data : np.ndarray
        Data that has already been gamma-corrected.
    gamma : float
        The gamma value that was originally used.

    Returns
    -------
    np.ndarray
        Data restored to its linear state, with the same dtype as the input.
    """
    # Convert to float for processing
    if np.issubdtype(gamma_corrected_data.dtype, np.integer):
        data = gamma_corrected_data.astype(np.float64) / 255.0
    else:
        data = gamma_corrected_data.astype(np.float64)
    # Apply inverse gamma
    restored = np.power(data, gamma)
    restored = np.clip(restored, 0.0, 1.0)
    if np.issubdtype(gamma_corrected_data.dtype, np.integer):
        out = np.rint(restored * 255).astype(gamma_corrected_data.dtype)
    else:
        out = restored.astype(gamma_corrected_data.dtype)
    return out


def linearize(raw_data: np.ndarray, lut: np.ndarray = None) -> np.ndarray:
    """
    Convert raw sensor data to linear space.

    If a LUT (lookup table) is supplied, the function interpolates the linear values
    from the LUT.  If no LUT is provided, a default inverse gamma with ``gamma=2.2`` is used.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw sensor values, typically uint8 in the range [0, 255].
    lut : np.ndarray, optional
        1‑D array of linear values for each possible input index.  Must be length 256.

    Returns
    -------
    np.ndarray
        Linearized data, same shape and dtype as ``raw_data``.
    """
    if lut is not None:
        # Ensure LUT is a float array for interpolation
        lut = lut.astype(np.float64)
        # Sample indices for a 256‑entry LUT
        indices = np.linspace(0, len(lut) - 1, 256)
        # Interpolate
        linear = np.interp(raw_data, indices, lut)
        return linear.astype(raw_data.dtype)

    # Default: inverse gamma with gamma = 2.2 (common for cyanotype)
    GAMMA = 2.2
    if np.issubdtype(raw_data.dtype, np.integer):
        data = raw_data.astype(np.float64) / 255.0
    else:
        data = raw_data.astype(np.float64)
    linear = np.power(data, GAMMA)
    linear = np.clip(linear, 0.0, 1.0)
    if np.issubdtype(raw_data.dtype, np.integer):
        out = np.rint(linear * 255).astype(raw_data.dtype)
    else:
        out = linear.astype(raw_data.dtype)
    return out