import cv2
import pandas as pd
import numpy as np
import argparse
import sys

def add_paper_grain(image, intensity=0.05):
    """Adds salt-and-pepper noise to simulate paper fiber texture."""
    h, w, c = image.shape
    noise = np.random.randn(h, w, c) * 255 * intensity
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_gradient_map(grayscale_img, dark_color, light_color):
    """
    Maps grayscale values to a gradient between two RGB colors.
    dark_color: (B, G, R) tuple for black pixels (0)
    light_color: (B, G, R) tuple for white pixels (255)
    """
    # Normalize image to 0.0 - 1.0
    norm = grayscale_img.astype(float) / 255.0
    
    # Expand dims for broadcasting: (H, W, 1)
    norm = np.expand_dims(norm, axis=2)
    
    # Convert colors to arrays
    c_dark = np.array(dark_color)
    c_light = np.array(light_color)
    
    # Linear interpolation: Color = (Dark * (1-val)) + (Light * val)
    # When val is 0 (black), we get Dark. When val is 1 (white), we get Light.
    result = (c_dark * (1 - norm)) + (c_light * norm)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def simulate_print(image_path, lut_path, output_path, simulate_texture=True):
    # 1. Load the Response Curve (LUT)
    print(f"Loading Response Curve from: {lut_path}")
    try:
        df = pd.read_csv(lut_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # We assume the CSV has 'Input' (Digital) and 'Output' (Measured Physical Result)
    # If using the correction LUT, this will show what a corrected image looks like.
    # If using raw measurement data, this shows what an uncorrected image looks like.
    if len(df.columns) < 2:
        print("Error: CSV needs at least two columns (Input, Output).")
        sys.exit(1)
        
    x_in = df.iloc[:, 0].values  # Column 0 (Input Digital)
    y_out = df.iloc[:, 1].values # Column 1 (Measured Response)

    # Create Look-Up Table
    lut_indices = np.arange(256)
    lut_mapped = np.interp(lut_indices, x_in, y_out)
    opencv_lut = np.clip(lut_mapped, 0, 255).astype(np.uint8).reshape((256, 1))

    # 2. Load Image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
        sys.exit(1)
        
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    opencv_lut = np.flipud(opencv_lut)
    # 3. Apply the Density Response (Simulate the Tone Curve)
    # This distorts the image based on how the laser actually burns the paper
    print("Applying density simulation...")
    simulated_density = cv2.LUT(gray, opencv_lut)

    # 4. Apply Cyanotype Color Grading
    # Prussian Blue (approx BGR: 120, 60, 20) to Paper White (BGR: 240, 248, 255)
    print("Applying Cyanotype color map...")
    
    # Define Colors (BGR format for OpenCV)
    # Deep Prussian Blue
    ink_color = (41,20 , 0) 
    # Creamy Watercolor Paper
    paper_color = (235, 245, 255) 
    
    color_img = apply_gradient_map(simulated_density, ink_color, paper_color)

    # 5. Simulate Paper Texture
    if simulate_texture:
        print("Adding paper grain texture...")
        color_img = add_paper_grain(color_img, intensity=0.04)

    # 6. Save
    print(f"Saving simulation to: {output_path}")
    cv2.imwrite(output_path, color_img)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a Cyanotype print based on a Response LUT.")
    
    parser.add_argument("image", help="Input digital image")
    parser.add_argument("lut", help="CSV containing the Response Curve")
    parser.add_argument("-o", "--output", help="Output filename", default="simulated_cyanotype.png")
    parser.add_argument("--no-texture", action="store_true", help="Disable paper grain simulation")
    
    args = parser.parse_args()
    
    simulate_print(args.image, args.lut, args.output, not args.no_texture)