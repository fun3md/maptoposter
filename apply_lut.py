import cv2
import pandas as pd
import numpy as np
import argparse
import sys
import os

def apply_calibration_and_resize(image_path, lut_path, output_path, size_mm=None, dpi=300):
    # --- 1. Load the LUT CSV ---
    print(f"Loading LUT from: {lut_path}")
    try:
        df = pd.read_csv(lut_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if 'Input' not in df.columns or 'Output' not in df.columns:
        print("Error: CSV must contain 'Input' and 'Output' columns.")
        sys.exit(1)

    # --- 2. Create the Look-Up Table ---
    x_in = df['Input'].values
    y_out = df['Output'].values
    target_indices = np.arange(256)
    lut_mapped = np.interp(target_indices, x_in, y_out)
    lut_mapped = np.clip(lut_mapped, 0, 255).astype(np.uint8)
    opencv_lut = lut_mapped.reshape((256, 1))

    # --- 3. Load the Image ---
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not open image file.")
        sys.exit(1)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        print("Converting color image to grayscale...")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # --- 4. Resize Logic (Physical Dimensions) ---
    if size_mm is not None and size_mm > 0:
        h, w = gray_img.shape[:2]
        print(f"Original Dimensions: {w}x{h} px")
        
        # Determine the longest edge
        if w >= h:
            longest_edge_px = w
            aspect_ratio = h / w
            new_w_mm = size_mm
            new_h_mm = size_mm * aspect_ratio
        else:
            longest_edge_px = h
            aspect_ratio = w / h
            new_h_mm = size_mm
            new_w_mm = size_mm * aspect_ratio

        # Calculate target pixels based on DPI
        # Formula: (MM / 25.4) * DPI = Pixels
        target_longest_px = int((size_mm / 25.4) * dpi)
        
        scale_factor = target_longest_px / longest_edge_px
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        print(f"Resizing to {size_mm}mm longest edge at {dpi} DPI.")
        print(f"New Dimensions: {new_w}x{new_h} px")

        # Use Lanczos interpolation (High quality downscaling/upscaling)
        gray_img = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # --- 5. Apply the LUT ---
    print("Applying correction curve...")
    corrected_img = cv2.LUT(gray_img, opencv_lut)

    # --- 6. Save Output ---
    print(f"Saving corrected image to: {output_path}")
    
    # Note: OpenCV saves the pixels correctly, but often defaults metadata to 96 DPI.
    # When importing into LightBurn/LaserSoftware, ensure you import at the same DPI 
    # you specified here, or simply tell the software "Pass-through" dimensions.
    cv2.imwrite(output_path, corrected_img)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Calibration LUT and Resize for Physical Output.")
    
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("lut", help="Path to the LUT CSV file")
    parser.add_argument("-o", "--output", help="Path for output PNG", default="corrected_output.png")
    
    # New Arguments for Sizing
    parser.add_argument("--size_mm", type=float, help="Length of the longest edge in Millimeters")
    parser.add_argument("--dpi", type=float, default=254.0, help="Target DPI (Default 254 = 0.1mm interval)")
    
    args = parser.parse_args()
    
    apply_calibration_and_resize(args.image, args.lut, args.output, args.size_mm, args.dpi)