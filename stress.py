import cv2
import numpy as np
import argparse
import sys
import os

def apply_stress_v2(img, samples=25, iterations=10, strength=0.5):
    """
    Apply STRESS v2 enhancement to an image with stability controls.
    
    Parameters
    ----------
    img : np.ndarray
        Input image as a NumPy array (grayscale).
    samples : int, optional
        Number of sampling iterations per stress pass. Must be positive.
    iterations : int, optional
        Number of stress passes to perform. Must be positive.
    strength : float, optional
        Blend strength between original and stressed image (0.0 to 1.0).
        0.0 returns original image, 1.0 returns fully stressed.
    
    Returns
    -------
    np.ndarray
        Stressed image with same shape and dtype as input.
    
    Raises
    ------
    ValueError
        If samples or iterations are not positive integers, or strength is not in [0.0, 1.0].
    """
    # Parameter validation
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(strength, (int, float)) or not (0.0 <= strength <= 1.0):
        raise ValueError("strength must be a float in the range [0.0, 1.0]")
    
    # Early return for no enhancement
    if strength <= 0:
        return img.copy()
    
    # Ensure input is a numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)
    img_f = img.astype(np.float32) / 255.0
    h, w = img_f.shape
    
    # Initialize envelopes with small margins to prevent division by zero
    inv_env = np.full_like(img_f, 0.01, dtype=np.float32)
    env = np.full_like(img_f, 1.0, dtype=np.float32) - 0.01
    
    # Perform stress iterations
    for _ in range(iterations):
        for _ in range(samples):
            # Generate small random offsets for local sampling
            dy = np.random.normal(0, h * 0.1, (h, w)).astype(np.int32)
            dx = np.random.normal(0, w * 0.1, (h, w)).astype(np.int32)
            
            yy = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
            xx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
            
            sampled = img_f[yy, xx]
            inv_env = np.maximum(inv_env, np.minimum(img_f, sampled))
            env = np.minimum(env, np.maximum(img_f, sampled))
    
    # Compute safe range with clamp to prevent division by zero
    safe_range = np.maximum(env - inv_env, 0.1)
    
    # Apply stress mapping
    stress_img = (img_f - inv_env) / safe_range
    stress_img = np.clip(stress_img * 255, 0, 255).astype(np.uint8)
    
    # Blend with original image
    if strength < 1.0:
        blended = cv2.addWeighted(img, 1.0 - strength, stress_img, strength, 0)
        return blended
    else:
        return stress_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply STRESS v2 enhancement to an image.")
    parser.add_argument("image_path", help="Path to the input grayscale image.")
    parser.add_argument("--samples", type=int, default=25, help="Number of sampling iterations per pass (default: 25).")
    parser.add_argument("--iterations", type=int, default=10, help="Number of stress passes to perform (default: 10).")
    parser.add_argument("--strength", type=float, default=0.5, help="Blend strength (0.0 to 1.0, default: 0.5).")
    parser.add_argument("--output", type=str, default=None, help="Path to output image (default: <input>_stressed.png).")
    args = parser.parse_args()
    
    # Load image in grayscale
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {args.image_path}")
        sys.exit(1)
    
    # Apply stress enhancement
    try:
        result = apply_stress_v2(img, samples=args.samples, iterations=args.iterations, strength=args.strength)
    except ValueError as e:
        print(f"Parameter error: {e}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.image_path)
        output_path = f"{base}_stressed.png"
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"Stressed image saved to {output_path}")