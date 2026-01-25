# --- START OF FILE apply_lut_fixed.py ---

import cv2
import pandas as pd
import numpy as np
import argparse
import sys
from PIL import Image

def apply_calibration_and_resize(image_path, lut_path, output_path, size_mm=None, dpi=300):
    # --- 1. Load the LUT CSV ---
    print(f"Loading LUT from: {lut_path}")
    try:
        df = pd.read_csv(lut_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Handle column names from the Notebook generator
    cols = df.columns
    if 'Input_Gray_Level' in cols and 'Output_Laser_Power' in cols:
        col_in = 'Input_Gray_Level'
        col_out = 'Output_Laser_Power'
    elif 'Input' in cols and 'Output' in cols:
        col_in = 'Input'
        col_out = 'Output'
    else:
        print(f"Error: CSV columns not recognized. Found: {cols}")
        print("Expected 'Input_Gray_Level' and 'Output_Laser_Power'")
        sys.exit(1)

    # --- 2. Create the OpenCV Look-Up Table ---
    # The Notebook calculated: Input Image Gray -> Required Laser Power
    # We map x_in -> y_out directly.
    
    x_in = df[col_in].values
    y_out = df[col_out].values

    # We ensure we have a mapping for every pixel 0-255 using interpolation
    # (Just in case the CSV is missing rows, though the notebook generates 256 rows)
    pixel_range = np.arange(256)
    
    # CORRECT LOGIC: Map Pixel Value (x) to Laser Power (y)
    final_lut_values = np.interp(pixel_range, x_in, y_out)
    
    # Format for OpenCV (Must be uint8, shape 256x1)
    opencv_lut = np.clip(final_lut_values, 0, 255).astype(np.uint8).reshape((256, 1))

    # --- 3. Load the Image ---
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not open image file.")
        sys.exit(1)

    # Ensure Grayscale
    if len(img.shape) == 3:
        print("Converting color image to grayscale...")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # --- 4. Resize Logic (Physical Dimensions) ---
    if size_mm is not None and size_mm > 0:
        h, w = gray_img.shape[:2]
        
        # Determine scaling based on longest edge
        if w >= h:
            longest_edge_px = w
            new_w_mm = size_mm
        else:
            longest_edge_px = h
            new_h_mm = size_mm

        # Formula: (MM / 25.4) * DPI = Pixels
        target_longest_px = int((size_mm / 25.4) * dpi)
        
        scale_factor = target_longest_px / longest_edge_px
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        print(f"Target Size: {size_mm}mm longest edge @ {dpi} DPI")
        print(f"Resizing: {w}x{h} -> {new_w}x{new_h} px")

        # Use Lanczos interpolation for high-quality resizing
        gray_img = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        pass

    # --- 5. Apply the Correction ---
    print("Applying correction curve...")
    
    # We DO NOT invert the image manually here.
    # The LUT was calculated such that Input 0 (Black) maps to the correct Power.
    # The LUT contains the physics/inversion logic.
    corrected_img = cv2.LUT(gray_img, opencv_lut)

    # --- 6. Save with Metadata (Using Pillow) ---
    print(f"Saving to: {output_path} with DPI={dpi}")
    
    pil_image = Image.fromarray(corrected_img.astype(np.uint8))
    pil_image.save(output_path, dpi=(int(dpi), int(dpi)))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Calibration LUT and Resize with DPI Metadata.")
    
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("lut", help="Path to the LUT CSV file")
    parser.add_argument("-o", "--output", help="Path for output PNG", default="corrected_output.png")
    
    parser.add_argument("--size_mm", type=float, help="Length of the longest edge in Millimeters")
    parser.add_argument("--dpi", type=float, default=254.0, help="Target DPI (Default 254 = 0.1mm interval)")
    
    args = parser.parse_args()
    
    apply_calibration_and_resize(args.image, args.lut, args.output, args.size_mm, args.dpi)