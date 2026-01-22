import cv2
import pandas as pd
import numpy as np
import argparse
import sys
import os

def apply_calibration_lut(image_path, lut_path, output_path):
    # 1. Load the LUT CSV
    print(f"Loading LUT from: {lut_path}")
    try:
        df = pd.read_csv(lut_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Ensure required columns exist
    if 'Input' not in df.columns or 'Output' not in df.columns:
        print("Error: CSV must contain 'Input' and 'Output' columns.")
        sys.exit(1)

    # 2. Create the Look-Up Table (Array)
    # We use numpy interpolation to ensure we have a value for every integer 0-255.
    # This makes the script robust even if your CSV is missing some rows.
    x_in = df['Input'].values
    y_out = df['Output'].values
    
    # Generate the strict 0-255 index
    target_indices = np.arange(256)
    
    # Interpolate to find exact mappings
    lut_mapped = np.interp(target_indices, x_in, y_out)
    
    # Clip to valid image range and convert to unsigned 8-bit integer
    lut_mapped = np.clip(lut_mapped, 0, 255).astype(np.uint8)
    
    # Reshape for OpenCV (256 rows, 1 column)
    opencv_lut = lut_mapped.reshape((256, 1))

    # 3. Load the Image
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not open image file.")
        sys.exit(1)

    # 4. Check Color/Grayscale
    # If image is color, convert to grayscale first (Standard for Laser prep)
    if len(img.shape) == 3:
        print("Converting color image to grayscale...")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # 5. Apply the LUT
    # cv2.LUT is extremely fast C++ implementation
    print("Applying correction curve...")
    corrected_img = cv2.LUT(gray_img, opencv_lut)

    # 6. Save Output
    print(f"Saving corrected image to: {output_path}")
    cv2.imwrite(output_path, corrected_img)
    print("Done.")

if __name__ == "__main__":
    # Argument Parsing for easy command line use
    parser = argparse.ArgumentParser(description="Apply a Calibration CSV LUT to an Image.")
    
    parser.add_argument("image", help="Path to input image (jpg, png, etc)")
    parser.add_argument("lut", help="Path to the LUT CSV file")
    parser.add_argument("-o", "--output", help="Path for output PNG", default="corrected_output.png")
    
    args = parser.parse_args()
    
    apply_calibration_lut(args.image, args.lut, args.output)