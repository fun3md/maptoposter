import cv2
import numpy as np
import pandas as pd
import argparse
import sys
import os
from PIL import Image

def load_lut(csv_path):
    if csv_path and os.path.exists(csv_path):
        print(f"Loading LUT from: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            x = df.iloc[:, 0].values.astype(float)
            y = df.iloc[:, 1].values.astype(float)
            is_inverted = y[0] > y[-1]
            if is_inverted:
                print(" -> Detected INVERTED LUT. Disabling auto-inversion.")
            return x, y, is_inverted
        except Exception as e:
            print(f"Error reading CSV: {e}")
            sys.exit(1)
    else:
        print("No CSV provided. Using LINEAR mapping.")
        x = np.array([0.0, 255.0])
        y = np.array([0.0, 255.0])
        return x, y, False

def get_bayer_noise(size_h, size_w):
    """ Generates a normalized Bayer noise map (-0.5 to +0.5) """
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
    
    # Normalize to -0.5 to +0.5 range
    bayer_norm = (bayer_8x8 / 64.0) - 0.5
    
    # Tile it
    rep_h = int(np.ceil(size_h / 8))
    rep_w = int(np.ceil(size_w / 8))
    tiled = np.tile(bayer_norm, (rep_h, rep_w))
    
    return tiled[:size_h, :size_w]

def process_adaptive_hybrid(image_path, csv_path, output_path, size_mm, dpi, max_steps, min_spread, max_spread, gamma):
    # 1. Load Data
    lut_x, lut_y, lut_is_inverted = load_lut(csv_path)

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

    # 3. Handle Inversion
    if not lut_is_inverted:
        print("Inverting image pixels (Standard Negative Generation)...")
        img = 255 - img
    else:
        print("Keeping image pixels as-is (LUT handles Negative Inversion).")

    norm = img / 255.0
    
    # Apply Gamma correction (Power law)
    # Output = Input ^ (1 / Gamma)
    # A gamma of ~2.0 is usually good for porous paper cyanotypes
    compensated = np.power(norm, 1.0 / gamma)
    compensated = np.clip(compensated * 255, 0, 255).astype(np.uint8)

    # 4. Map to Curve
    print("Applying LUT interpolation...")
    corrected_255 = np.interp(compensated, lut_x, lut_y)

    # 5. Convert to Hardware Steps
    print(f"Quantizing to {max_steps} hardware power steps...")
    step_scale = max_steps / 255.0
    ideal_steps = corrected_255 * step_scale
    
    # --- ADAPTIVE DITHERING LOGIC ---
    print(f"Calculating Adaptive Spread (Min: {min_spread}, Max: {max_spread})...")

    # A. Calculate Local Gradient (Variance/Edge Detection)
    # We use Sobel to find how fast pixels change neighbors
    gx = cv2.Sobel(ideal_steps, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(ideal_steps, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(gx, gy)
    
    # B. Create Flatness Mask (0.0 = Busy/Edge, 1.0 = Flat)
    # We use exponential decay. 
    # Sensitivity factor 5.0 means gradients > 5 (out of 255) start reducing spread rapidly.
    flatness_mask = np.exp(-gradient_mag / 10.0)
    
    # C. Calculate Adaptive Spread Map
    # Where Flatness is 1, we use Max Spread. Where Flatness is 0, we use Min Spread.
    current_spread_map = min_spread + (flatness_mask * (max_spread - min_spread))

    # D. Calculate Edge Damping (Black/White Protection)
    # Protects absolute 0 and absolute max from noise
    dist_from_bottom = ideal_steps
    dist_from_top = max_steps - ideal_steps
    edge_distance = np.minimum(dist_from_bottom, dist_from_top)
    
    # We divide by current_spread_map to ensure protection scales with the spread size
    damping_mask = np.clip(edge_distance / (current_spread_map * 0.5 + 0.001), 0.0, 1.0)
    
    # E. Combine Everything
    # Final Noise = Base Bayer * Adaptive Spread * Edge Protection
    base_bayer = get_bayer_noise(img.shape[0], img.shape[1])
    
    effective_noise = base_bayer * current_spread_map * damping_mask
    
    dithered_steps = ideal_steps + effective_noise
    final_steps = np.round(dithered_steps)
    final_steps = np.clip(final_steps, 0, max_steps)
    
    # --- END ADAPTIVE LOGIC ---

    # 7. Convert BACK to 0-255 for Laser Software
    print("Encoding final image...")
    inverse_scale = 255.0 / max_steps
    final_image_array = final_steps * inverse_scale
    final_image_array = np.clip(np.round(final_image_array), 0, 255).astype(np.uint8)

    # 8. Save
    print(f"Saving to {output_path}")
    Image.fromarray(final_image_array).save(output_path, dpi=(int(dpi), int(dpi)))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Hybrid Dither for Cyanotype")
    parser.add_argument("image", help="Input image")
    parser.add_argument("--csv", help="Correction CSV", default=None)
    parser.add_argument("-o", "--output", default="adaptive_print.png")
    parser.add_argument("--size_mm", type=float, default=150)
    parser.add_argument("--dpi", type=float, default=318)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--min_spread", type=float, default=1.0, help="Spread on Edges/Details (Low noise)")
    parser.add_argument("--max_spread", type=float, default=4.0, help="Spread on Flat Areas (High blending)")
    parser.add_argument("-g", "--gamma", type=float, default=1.0, help="add gamma")
    
    args = parser.parse_args()
    
    process_adaptive_hybrid(args.image, args.csv, args.output, args.size_mm, args.dpi, args.steps, args.min_spread, args.max_spread, args.gamma)