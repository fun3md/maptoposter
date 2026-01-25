import cv2
import numpy as np
import pandas as pd
import argparse
import sys
from PIL import Image

def load_lut(csv_path):
    """ Loads the CSV and creates an interpolation function. """
    try:
        df = pd.read_csv(csv_path)
        x = df.iloc[:, 0].values.astype(float) # Input (Digital 0-255)
        y = df.iloc[:, 1].values.astype(float) # Output (Corrected 0-255)
        return x, y
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def get_bayer_matrix(size_h, size_w):
    """ Generates a tiled Bayer matrix normalized to 0.0 - 1.0 """
    # 8x8 Bayer Pattern
    bayer_8x8 = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ], dtype=np.float32)
    
    # Normalize to 0.0 - 1.0 range
    bayer_norm = (bayer_8x8 + 0.5) / 64.0
    
    # Tile it to match image size
    rep_h = int(np.ceil(size_h / 8))
    rep_w = int(np.ceil(size_w / 8))
    tiled = np.tile(bayer_norm, (rep_h, rep_w))
    
    return tiled[:size_h, :size_w]

def process_stepped_hybrid(image_path, csv_path, output_path, size_mm, dpi, max_steps):
    # 1. Load Data
    print(f"Loading LUT: {csv_path}")
    lut_x, lut_y = load_lut(csv_path)

    print(f"Loading Image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return

    # 2. Resize
    if size_mm:
        h, w = img.shape
        pixel_scale = (size_mm / 25.4) * dpi
        scale = pixel_scale / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing to {new_w}x{new_h} px ({size_mm}mm @ {dpi} DPI)")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Create Negative
    print("Inverting for Cyanotype Negative...")
    img = 255 - img

    # 4. Map to Correction Curve (0-255 scale)
    # This gives us the ideal corrected values as floats
    print("Applying LUT interpolation...")
    corrected_255 = np.interp(img, lut_x, lut_y)

    # 5. Convert to Hardware Steps (0 - max_steps)
    # If max_steps is 80, 255 becomes 80.0
    print(f"Quantizing to {max_steps} hardware power steps...")
    
    # Calculate scale factor: 255 (image max) -> 80 (hardware max)
    step_scale = max_steps / 255.0
    
    # This is the precise float value of the step we WANT (e.g. 40.5)
    target_steps = corrected_255 * step_scale
    
    # 6. Apply Bayer Dithering on the Steps
    # Split into Base Step (40) and Remainder (0.5)
    base_step = np.floor(target_steps)
    remainder = target_steps - base_step
    
    # Get Dither Pattern
    threshold_map = get_bayer_matrix(img.shape[0], img.shape[1])
    
    # Determine if we bump up to the next step
    # If remainder 0.5 > threshold, step becomes 41, else 40
    step_bump = (remainder > threshold_map).astype(np.float32)
    
    final_steps = base_step + step_bump
    final_steps = np.clip(final_steps, 0, max_steps)

    # 7. Convert BACK to 0-255 Image File
    # The laser software expects 0-255. 
    # We map Step 80 back to Pixel 255.
    # We map Step 1 back to Pixel (255/80) approx 3.
    print("Encoding final image...")
    inverse_scale = 255.0 / max_steps
    final_image_array = final_steps * inverse_scale
    
    # Round to nearest integer for 8-bit save
    final_image_array = np.clip(np.round(final_image_array), 0, 255).astype(np.uint8)

    # 8. Save
    print(f"Saving to {output_path}")
    Image.fromarray(final_image_array).save(output_path, dpi=(int(dpi), int(dpi)))
    print("Done.")
    print("-" * 30)
    print("LASER SETTINGS:")
    print("Mode: Grayscale")
    print("Min Power: 0%")
    print("Max Power: [Set this to the power level that corresponds to your Step 80]")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stepped Hybrid Dither for Limited Power Lasers")
    parser.add_argument("image", help="Input image")
    parser.add_argument("csv", help="Correction CSV")
    parser.add_argument("-o", "--output", default="stepped_print.png")
    parser.add_argument("--size_mm", type=float, default=150)
    parser.add_argument("--dpi", type=float, default=318)
    parser.add_argument("--steps", type=int, default=80, help="Number of usable laser power steps (default 80)")
    
    args = parser.parse_args()
    
    process_stepped_hybrid(args.image, args.csv, args.output, args.size_mm, args.dpi, args.steps)