import cv2
import numpy as np
import pandas as pd
import argparse
import sys
import os
from PIL import Image

def load_lut(csv_path):
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            x, y = df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)
            # Detect if LUT is already inverted (In: 0 -> Out: 255)
            is_inverted = y[0] > y[-1]
            return x, y, is_inverted
        except Exception as e:
            print(f"Error: {e}"); sys.exit(1)
    return np.array([0.0, 255.0]), np.array([0.0, 255.0]), False

def apply_gamma(img, gamma=1.0):
    if gamma == 1.0: return img
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_stress_v2(img, samples=25, iterations=10, strength=0.5):
    """
    STRESS v2 with stability controls to prevent 'Blow out'.
    strength: 0.0 (original image) to 1.0 (full STRESS effect).
    """
    if strength <= 0: return img
    
    print(f"Applying STRESS Enhancement (Strength: {strength})...")
    img_f = img.astype(np.float32) / 255.0
    h, w = img_f.shape
    
    # Initialize envelopes with slightly wider margins to prevent division by zero
    inv_env = np.zeros_like(img_f) + 0.01
    env = np.ones_like(img_f) - 0.01
    
    for _ in range(iterations):
        for _ in range(samples):
            # Reduced search radius slightly for more local stability
            dy = np.random.normal(0, h * 0.1, (h, w)).astype(np.int32)
            dx = np.random.normal(0, w * 0.1, (h, w)).astype(np.int32)
            
            yy = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
            xx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
            
            sampled = img_f[yy, xx]
            inv_env = np.maximum(inv_env, np.minimum(img_f, sampled))
            env = np.minimum(env, np.maximum(img_f, sampled))
            
    safe_range = np.maximum(env - inv_env, 0.1)
    
    stress_img = (img_f - inv_env) / safe_range
    stress_img = np.clip(stress_img * 255, 0, 255).astype(np.uint8)
    
    # Blend with original image based on strength
    return cv2.addWeighted(img, 1.0 - strength, stress_img, strength, 0)

def get_bayer_noise(size_h, size_w):
    bayer = np.array([[ 0, 32,  8, 40,  2, 34, 10, 42],
                      [48, 16, 56, 24, 50, 18, 58, 26],
                      [12, 44,  4, 36, 14, 46,  6, 38],
                      [60, 28, 52, 20, 62, 30, 54, 22],
                      [ 3, 35, 11, 43,  1, 33,  9, 41],
                      [51, 19, 59, 27, 49, 17, 57, 25],
                      [15, 47,  7, 39, 13, 45,  5, 37],
                      [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32)
    bayer_norm = (bayer / 64.0) - 0.5
    tiled = np.tile(bayer_norm, (int(np.ceil(size_h/8)), int(np.ceil(size_w/8))))
    return tiled[:size_h, :size_w]

def process_stress_hybrid(image_path, csv_path, output_path, size_mm, dpi, max_steps, min_spread, max_spread, gamma, stress_val):
    # 1. Load
    lut_x, lut_y, lut_is_inverted = load_lut(csv_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    # 2. Resize
    pixel_scale = (size_mm / 25.4) * dpi
    scale = pixel_scale / max(img.shape)
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_LANCZOS4)

    # 3. Pre-process Gamma
    img = apply_gamma(img, gamma)

    # 4. Apply STRESS (with safety)
    img = apply_stress_v2(img, strength=stress_val)

    # 5. LUT Mapping
    # NOTE: If your LUT is 0->255 and 255->0, it is a negative LUT.
    # If the file is still 'too white', try adding '--force_invert' logic.
    # if not lut_is_inverted:
    #     img = 255 - img
        
    corrected_255 = np.interp(img, lut_x, lut_y)

    # 6. Adaptive Dither
    step_scale = max_steps / 255.0
    ideal_steps = corrected_255 * step_scale
    
    gx = cv2.Sobel(ideal_steps, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(ideal_steps, cv2.CV_64F, 0, 1, ksize=3)
    flatness = np.exp(-cv2.magnitude(gx, gy) / 10.0)
    
    current_spread = min_spread + (flatness * (max_spread - min_spread))
    edge_dist = np.minimum(ideal_steps, max_steps - ideal_steps)
    damping = np.clip(edge_dist / (current_spread * 0.5 + 0.001), 0.0, 1.0)
    
    noise = get_bayer_noise(img.shape[0], img.shape[1]) * current_spread * damping
    final_steps = np.clip(np.round(ideal_steps + noise), 0, max_steps)
    #final_steps = np.clip(np.round(ideal_steps), 0, max_steps)
    # 7. Final Scale back to 0-255 PNG
    final_img = (final_steps * (255.0 / max_steps)).astype(np.uint8)
    Image.fromarray(final_img).save(output_path, dpi=(int(dpi), int(dpi)))
    print(f"Success. File saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--csv", default=None)
    parser.add_argument("-o", "--output", default="stress_v2_print.png")
    parser.add_argument("--size_mm", type=float, default=150)
    parser.add_argument("--dpi", type=float, default=318)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--stress", type=float, default=0.5, help="Strength of STRESS: 0.0 to 1.0")
    parser.add_argument("--min_spread", type=float, default=1.0)
    parser.add_argument("--max_spread", type=float, default=4.0)
    
    args = parser.parse_args()
    process_stress_hybrid(args.image, args.csv, args.output, args.size_mm, args.dpi, args.steps, args.min_spread, args.max_spread, args.gamma, args.stress)