import cv2
import numpy as np
import argparse
from PIL import Image, ImageOps

def apply_dot_gain_compensation(img_array, compensation_factor=1.5):
    """
    Lift midtones to compensate for UV light bleeding (Dot Gain).
    compensation_factor > 1.0 brightens the mids.
    """
    # Normalize to 0-1
    norm = img_array / 255.0
    
    # Apply Gamma correction (Power law)
    # Output = Input ^ (1 / Gamma)
    # A gamma of ~2.0 is usually good for porous paper cyanotypes
    compensated = np.power(norm, 1.0 / compensation_factor)
    
    return np.clip(compensated * 255, 0, 255).astype(np.uint8)

def process_dither(image_path, output_path, size_mm, dpi, gain_compensation):
    print(f"Loading {image_path}...")
    
    # 1. Load using OpenCV for mathematical operations
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return

    # 2. Resize Logic (CRITICAL for Dithering)
    # We must resize to exact print pixels before converting to 1-bit
    if size_mm:
        h, w = img.shape
        pixel_scale = (size_mm / 25.4) * dpi
        
        # Calculate aspect ratio
        if w >= h:
            scale = pixel_scale / w
        else:
            scale = pixel_scale / h
            
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        print(f"Resizing to {new_w}x{new_h} px ({size_mm}mm @ {dpi} DPI)")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Apply Dot Gain Compensation (The "Fix" for crushed mids)
    # This replaces the Inverted LUT. It creates "space" in the mids.
    print(f"Applying Dot Gain Compensation (Gamma: {gain_compensation})...")
    img_compensated = apply_dot_gain_compensation(img, gain_compensation)

    # 4. Convert to PIL for Dithering
    # PIL has better dithering algorithms than OpenCV
    pil_img = Image.fromarray(img_compensated)

    # 5. Dither
    print("Applying Floyd-Steinberg Dithering...")
    # Convert to '1' (1-bit pixels)
    dithered = pil_img.convert('1', dither=Image.Dither.FLOYDSTEINBERG)

    # 6. Save
    print(f"Saving to {output_path}")
    dithered.save(output_path, dpi=(int(dpi), int(dpi)))
    print("Done. Burn this image at a CONSTANT power setting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dithered Bitmap for Cyanotype")
    parser.add_argument("image", help="Input image")
    parser.add_argument("-o", "--output", help="Output filename", default="dithered_print.png")
    parser.add_argument("--size_mm", type=float, default=150, help="Longest edge size in mm")
    parser.add_argument("--dpi", type=float, default=254.0, help="Output DPI (Match your Laser setting)")
    parser.add_argument("--gamma", type=float, default=1.8, help="Compensation Strength (1.0 = None, 1.8 = Recommended for paper)")
    
    args = parser.parse_args()
    
    process_dither(args.image, args.output, args.size_mm, args.dpi, args.gamma)