#!/usr/bin/env python3
"""
Main processing pipeline for map poster images.

Steps:
1. Load grayscale image.
2. Apply gamma correction (via gamma_utils).
3. Apply STRESS adaptation (via stress.py).
4. Map using inverse LUT (via lut_utils).
5. Apply dithering (via dither_v2) using config from defaults.yaml (or override).
6. Save result.
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Import utility modules
from gamma_utils import gamma_apply
from stress import apply_stress_v2
from lut_utils import load_lut, invert_lut, interpolate_lut
import dither_v2


def load_config(override_path: str = None) -> dict:
    """
    Load dithering configuration from defaults.yaml, optionally overridden by a user file.
    The config format is simple key: value lines (YAML-like).
    """
    # Start with defaults from dither_v2.load_dither_config (which reads defaults.yaml)
    config = dither_v2.load_dither_config()

    # If an override file is provided, load it and merge (override takes precedence)
    if override_path and os.path.exists(override_path):
        with open(override_path, "r") as f:
            user_config = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to interpret numeric values
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                        else:
                            value = value.lower()
                    user_config[key] = value
        # Apply overrides
        config.update(user_config)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Process an image through gamma correction, STRESS adaptation, "
                    "inverse LUT mapping, and dithering, then save the result."
    )
    parser.add_argument("input_path", help="Path to the input grayscale image")
    parser.add_argument("output_path", help="Path to save the processed image")
    parser.add_argument(
        "--config",
        help="Optional config file to override defaults.yaml for dithering parameters",
        default=None,
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # 1. Load image
    # ----------------------------------------------------------------------
    img = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {args.input_path}", file=sys.stderr)
        sys.exit(1)

    # ----------------------------------------------------------------------
    # 2. Gamma correction
    # ----------------------------------------------------------------------
    # Default gamma value; can be overridden via config if desired
    gamma_value = 2.0  # typical for cyanotype
    # If config includes gamma, use it
    config = load_config(args.config)
    if "gamma" in config:
        gamma_value = config["gamma"]
    img = gamma_apply(img, gamma=gamma_value)

    # ----------------------------------------------------------------------
    # 3. STRESS adaptation
    # ----------------------------------------------------------------------
    # Default stress parameters
    stress_samples = 25
    stress_iterations = 10
    stress_strength = 0.5

    # Allow config overrides if present
    if "samples" in config:
        stress_samples = int(config["samples"])
    if "iterations" in config:
        stress_iterations = int(config["iterations"])
    if "strength" in config:
        stress_strength = float(config["strength"])

    try:
        img = apply_stress_v2(
            img, samples=stress_samples, iterations=stress_iterations, strength=stress_strength
        )
    except ValueError as e:
        print(f"Stress parameter error: {e}", file=sys.stderr)
        sys.exit(1)

    # ----------------------------------------------------------------------
    # 4. Inverse LUT mapping
    # ----------------------------------------------------------------------
    # Default LUT path; can be overridden via environment variable or hardcode
    default_lut_path = "luts/laser_correction_v2.csv"
    lut_path = args.lut if hasattr(args, "lut") else default_lut_path

    # Load LUT and create inverse mapping
    lut_func = load_lut(lut_path)
    inv_lut_func = invert_lut(lut_func)
    # Apply inverse LUT
    img = interpolate_lut(inv_lut_func, img)

    # ----------------------------------------------------------------------
    # 5. Dithering
    # ----------------------------------------------------------------------
    # Load dithering config (may have been overridden)
    dither_config = load_config(args.config)

    # Extract dithering parameters
    dither_type = dither_config.get("dither_type", "bayer")
    strength = dither_config.get("strength", 1.0)
    # For this simple pipeline we only use strength; other parameters could be used
    # for more advanced adaptive dithering but are omitted for brevity.

    # Generate Bayer noise matching the image size
    noise = dither_v2.get_bayer_noise(img.shape[0], img.shape[1])
    # Scale noise by strength
    noisy_img = img.astype(np.float32) + noise * strength
    # Clip and convert back to uint8
    final_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------------
    # 6. Save output
    # ----------------------------------------------------------------------
    success = cv2.imwrite(args.output_path, final_img)
    if not success:
        print(f"Error: Failed to save output image to {args.output_path}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Processed image saved to {args.output_path}")


if __name__ == "__main__":
    main()