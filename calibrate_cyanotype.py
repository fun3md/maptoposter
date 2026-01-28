#!/usr/bin/env python3
"""
calibrate_cyanotype.py - Generate test targets, load measurements, fit spline, export inverse LUT.

This script provides utilities for cyanotype calibration:
- generate_test_targets: creates PNG/TIFF test target images.
- load_measurements: loads input-output measurement pairs from CSV.
- fit_spline: fits a smooth spline to measurement data.
- export_inverse_lut: computes and saves the inverse lookup table.

Usage examples:
    python calibrate_cyanotype.py generate --output target.png
    python calibrate_cyanotype.py calibrate --measurements measurements.csv --output_lut inverse_lut.csv --target_path target.png
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from PIL import Image
import os
import json
from stress import apply_stress_v2
import gamma_utils

# ----------------------------------------------------------------------
# Test target generation
# ----------------------------------------------------------------------
def generate_test_target(output_path, steps=256):
    """
    Generate a gradient test target image and save as PNG and TIFF.

    Parameters
    ----------
    output_path : str
        Base path (without extension) for the output files.
    steps : int, optional
        Number of gradient steps (default 256).
    """
    # Create a 2D gradient array (horizontal gradient)
    gradient = np.tile(np.arange(steps, dtype=np.uint8), (steps, 1))
    # Ensure 8-bit
    gradient = np.clip(gradient, 0, 255)
    # Convert to PIL Image
    img = Image.fromarray(gradient)
    # Convert PIL Image to numpy array
    gradient_array = np.array(img)
    # Apply stress enhancement
    stressed_array = apply_stress_v2(gradient_array, samples=25, iterations=10, strength=0.5)
    # Convert back to PIL Image
    stressed_img = Image.fromarray(stressed_array)
    # Save stressed PNG
    png_path = f"{output_path}.png"
    stressed_img.save(png_path)
    # Save stressed TIFF
    tiff_path = f"{output_path}.tiff"
    stressed_img.save(tiff_path)
    print(f"Generated test targets: {png_path} and {tiff_path}")
    return png_path, tiff_path

# ----------------------------------------------------------------------
# Measurement loading
# ----------------------------------------------------------------------
def load_measurements(csv_path):
    """
    Load measurement data from a CSV file.

    The CSV is expected to have at least two columns: input and output.
    Columns names can be anything; the function will use the first two columns.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    inputs : np.ndarray
        Array of input (digital) values.
    outputs : np.ndarray
        Array of output (measured) values.
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns.")
    inputs = df.iloc[:, 0].values.astype(float)
    outputs = df.iloc[:, 1].values.astype(float)
    return inputs, outputs

# ----------------------------------------------------------------------
# Spline fitting
# ----------------------------------------------------------------------
def fit_spline(inputs, outputs):
    """
    Fit a cubic spline to the measurement data.

    Parameters
    ----------
    inputs : np.ndarray
        Input values (x-coordinates).
    outputs : np.ndarray
        Output values (y-coordinates).

    Returns
    -------
    spline : scipy.interpolate.CubicSpline
        Fitted spline function mapping input -> output.
    """
    # Sort by input to avoid issues
    sorted_idx = np.argsort(inputs)
    inputs_sorted = inputs[sorted_idx]
    outputs_sorted = outputs[sorted_idx]
    # Ensure strictly increasing input values for scipy CubicSpline
    # Add a small epsilon to each element based on its index to guarantee uniqueness
    epsilon = 1e-12
    inputs_sorted = inputs_sorted + epsilon * np.arange(len(inputs_sorted))
    # Fit cubic spline
    spline = CubicSpline(inputs_sorted, outputs_sorted, bc_type='natural')
    return spline

# ----------------------------------------------------------------------
# Inverse LUT export
# ----------------------------------------------------------------------
def export_inverse_lut(spline, output_csv, num_samples=1000):
    """
    Export the inverse of a spline as an inverse LUT CSV.

    The inverse LUT maps desired output (target physical value) to the required input
    digital value.

    Parameters
    ----------
    spline : scipy.interpolate.CubicSpline
        The fitted spline function.
    output_csv : str
        Path to the CSV file to write.
    num_samples : int, optional
        Number of points to sample for the inverse LUT (default 1000).
    """
    # Sample densely from min to max of output range
    y_dense = np.linspace(0, 255, num_samples)
    # Use spline to predict corresponding input values
    x_dense = spline(y_dense)
    # Clip to [0,255]
    x_dense = np.clip(x_dense, 0, 255)
    # For inverse LUT, we want to map from desired output (y) to input (x)
    # We'll sample at evenly spaced input steps for final LUT
    lut_input = np.linspace(0, 255, 256)
    # Compute corresponding output using original spline to map input->output
    # But we need inverse mapping: we have y_dense (desired output) -> x_dense (required input).
    # So we can construct a CSV with two columns: DesiredOutput, RequiredInput.
    # Let's write the dense mapping first, then optionally downsample to 256 steps.
    # Write dense mapping
    df_dense = pd.DataFrame({'DesiredOutput': y_dense, 'RequiredInput': x_dense})
    df_dense.to_csv(output_csv, index=False)
    # Also create a 256-step version for typical LUT usage
    lut_indices = np.linspace(0, 255, 256)
    lut_output = np.interp(lut_indices, y_dense, x_dense)
    df_lut = pd.DataFrame({'Input': lut_indices, 'Output': lut_output})
    # Save both? We'll just overwrite with the 256 version for standard LUT format
    df_lut.to_csv(output_csv, index=False)
    print(f"Inverse LUT exported to {output_csv}")
    return

# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cyanotype calibration tool - generate test targets, load measurements, fit spline, export inverse LUT."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for generating test targets
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate test target PNG/TIFF images."
    )
    gen_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Base output path (without extension) for the generated test target."
    )
    gen_parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Number of gradient steps (default: 256)."
    )

    # Subparser for calibration
    calib_parser = subparsers.add_parser(
        "calibrate",
        help="Load measurements, fit spline, export inverse LUT."
    )
    calib_parser.add_argument(
        "--measurements",
        required=True,
        help="Path to CSV file containing measurement pairs (input, output)."
    )
    calib_parser.add_argument(
        "--output_lut",
        required=True,
        help="Path to export the inverse LUT CSV."
    )
    calib_parser.add_argument(
        "--target_path",
        default=None,
        help="Optional path to a test target image to generate before calibration."
    )
    calib_parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Number of steps for inverse LUT sampling (default: 256)."
    )
    calib_parser.add_argument("--dither_type", help="Type of dithering to use (e.g., bayer, random)", default="bayer")
    calib_parser.add_argument("--strength", type=float, help="Strength multiplier for dithering noise", default=1.0)
    calib_parser.add_argument("--min_spread", type=float, help="Spread on Edges/Details (Low noise)", default=1.0)
    calib_parser.add_argument("--max_spread", type=float, help="Spread on Flat Areas (High blending)", default=4.0)
    calib_parser.add_argument("--edge_damping", type=float, help="Edge damping factor for protecting blacks/whites", default=0.5)

    args = parser.parse_args()
 

    if args.command == "generate":
        generate_test_target(args.output, steps=args.steps)
    elif args.command == "calibrate":
        # Store dithering config for pipeline usage
        dither_config = {
            "dither_type": args.dither_type,
            "strength": args.strength,
            "min_spread": args.min_spread,
            "max_spread": args.max_spread,
            "edge_damping": args.edge_damping
        }
        # Write config to JSON file for downstream usage
        with open("pipeline_config.json", "w") as f:
            json.dump(dither_config, f, indent=2)
        # Optionally generate a test target before calibration
        if args.target_path:
            generate_test_target(args.target_path, steps=args.steps)
        # Load measurements
        inputs, outputs = load_measurements(args.measurements)
        # Linearize data using gamma utilities
        inputs_linear = gamma_utils.linearize(inputs).astype(np.float64)
        outputs_linear = gamma_utils.linearize(outputs).astype(np.float64)
        # Fit spline on linearized data
        spline = fit_spline(inputs_linear, outputs_linear)
        # Export inverse LUT
        export_inverse_lut(spline, args.output_lut, num_samples=args.steps)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()