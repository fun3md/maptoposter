# Summary and Detailed Explanation of `jupyter/dither_image.ipynb`

## 1. Overview
`jupyter/dither_image.ipynb` is a Jupyter notebook that implements a full pipeline for **laser‑printed poster production**.  
The pipeline covers:

1. **Image loading & physical sizing** – reads a high‑resolution source image, rescales it to a target longest edge (mm) while preserving aspect ratio.
2. **Perceptual grayscale conversion** – converts RGB to grayscale using Rec. 601 luma coefficients.
3. **Blue‑noise dithering** – applies a custom blue‑noise texture to simulate the stochastic nature of laser dot placement.
4. **Cyanotype‑style correction** – applies gamma pre‑compensation, LUT‑based density compensation, and spatial deconvolution (sharpening) to counteract ink spread and paper texture.
5. **Final output** – writes the processed image (`post_dither.png`) and optionally visualizes intermediate steps.

The notebook is tightly coupled with auxiliary files located in the repository root and sub‑folders (e.g., `bluenoise/`, `jupyter/` LUTs, kernels).

---

## 2. Key Configuration Parameters (top of the notebook)

| Variable | Description | Typical Value |
|----------|-------------|---------------|
| `input_img_path` | Path to the “clean” source image (high‑resolution) | `../posters/IMG_2075-Enhanced-NR.png` |
| `target_image_dpi` | DPI of the target printed poster | `260` |
| `TARGET_LONGEST_EDGE_MM` | Desired length of the longest side of the printed poster (mm) | `270` |
| `runout_mm` | Extra white border around the final image (mm) | `2.0` |
| `percentile` | Percentile clipping for contrast scaling | `99` |
| `pre_adj_strength` | Strength of the final sharpening adjustment (0‑1.5) | `0.5` |
| `pre_gamma` | Gamma applied during pre‑compensation (usually 1.0) | `1.0` |
| `lut_file` | CSV file containing the lookup table for cyanotype correction | `uv_laser_correction_inverted.csv` |
| `BLUE_NOISE_PATH` | Path to the 128×128 blue‑noise PNG used for dithering | `../bluenoise/128_128/LDR_LLL1_0.png` |
| `kernel_file` | Numpy kernel file for deconvolution (simulates ink spread) | `averaged_deconvolution_kernel.npy` |
| `kernel_source_dpi` | Source DPI of the kernel (usually 1200) | `1200` |

These parameters are used throughout the notebook to control scaling, contrast, and the intensity of various image‑processing steps.

---

## 3. Core Functions

### 3.1 `apply_dot_gain(image_path, kernel_path, source_dpi=1200, target_dpi=300)`
* **Purpose**: Simulates dot gain by convolving a scaled kernel with the image.
* **Steps**:
  1. Load the target image.
  2. Resize it to match the source DPI.
  3. Load and optionally resize the kernel.
  4. Apply `cv2.filter2D` convolution.
  5. (Optional) Apply non‑linear gamma adjustment.
* **Returns**: Original image, simulated image, and the kernel used.

### 3.2 `apply_cyanotype_correction(image_path, kernel_path, lut_csv_path, target_gamma=2.2, strength=1.0)`
* **Purpose**: Implements the full cyanotype‑style correction pipeline.
* **Sub‑steps**:
  1. **Gamma Pre‑Compensation** – raises image values to `1/target_gamma`.
  2. **LUT Compensation** – loads a CSV LUT, inverts it if needed, and interpolates to create an inverse mapping.
  3. **Spatial Deconvolution (Sharpening)** – rescales the kernel to the target DPI, normalizes it, and applies a high‑pass filter.
  4. **Final Output** – combines the corrected image with a paper texture and applies a final clipping operation.
* **Visualization**: Plots original, combined correction curve, and the final printed result.
* **Return**: The final corrected image (`final_img`).

### 3.3 Helper Functions
| Function | Role |
|----------|------|
| `load_and_normalize(path)` | Loads an image (PIL or NumPy), converts to RGB, normalizes to `[0,1]`. |
| `rgb_to_grayscale_perceptual(img_rgb)` | Converts RGB to grayscale using Rec. 601 coefficients. |
| `tile_noise(noise_texture, target_shape)` | Repeats a small noise texture to cover the whole image. |
| `resize_to_physical_dim(pil_img, longest_edge_mm, dpi)` | Resizes an image to a physical size (mm) while preserving aspect ratio. |
| `scale_to_percentile_global(img_rgb, percentile)` | Clips extreme pixel values based on a percentile to improve contrast. |
| `generate_paper_texture(shape, scale=0.5, intensity=0.05)` | Procedurally creates a watercolor‑paper texture. |
| `apply_print_simulation_alt(...)` | Alternative pipeline that directly maps digital values to ink density using an inverted LUT and adds texture. |

All helpers are defined near the bottom of the notebook and are used by the main execution block.

---

## 4. Execution Flow

1. **Parameter Setup** – The notebook defines all configuration values (paths, DPI, target size, etc.).
2. **Image Loading** – `load_and_normalize` reads `input_img_path` and optionally scales it to the target physical dimension.
3. **Grayscale Conversion** – The RGB image is converted perceptually to grayscale.
4. **Noise Loading & Tiling** – The blue‑noise texture is loaded, converted to grayscale, and tiled to match the image size.
5. **Power‑Level Mapping** – Using a predefined `power_levels` array, each pixel’s grayscale value is mapped to a discrete laser‑power step via `np.searchsorted`.
6. **Normalization & Dithering** – The pixel’s position within its step interval is normalized, compared against the tiled noise, and thresholded to produce a dithered binary output (`dithered_output`).
7. **Post‑Processing** – A run‑out border is added, and the result is saved as `post_dither.png`.
8. **Cyanotype Correction** – `apply_cyanotype_correction` is called with the original image, kernel, LUT, and user‑defined gamma/strength. This produces a refined image (`corrected_pre_dither.png`).
9. **Visualization** – Multiple Matplotlib sub‑plots display:
   - Original grayscale input.
   - Blue‑noise dithered result.
   - Histogram of the input.
   - Optional intermediate steps from the cyanotype correction.
10. **Final Save** – The dithered image is saved as `post_dither.png`.

The notebook ends with a few commented‑out alternative visualizations and a generic exception handler that prompts the user to verify file paths.

---

## 5. Dependencies & File References

| Type | Path (relative to workspace) | Purpose |
|------|------------------------------|---------|
| Image file | `../posters/IMG_2075-Enhanced-NR.png` | Source high‑resolution poster image. |
| Blue‑noise texture | `../bluenoise/128_128/LDR_LLL1_0.png` | Stochastic dithering pattern. |
| Kernel (deconvolution) | `averaged_deconvolution_kernel.npy` | Simulates ink spread; used for sharpening. |
| LUT CSV | `uv_laser_correction_inverted.csv` | Maps digital values to paper density (inverse LUT). |
| Output images | `corrected_pre_dither.png`, `post_dither.png`, `simulated_paper_conv.png` | Intermediate and final results. |

All these files must be present in the repository; otherwise the notebook raises a `FileNotFoundError`.

---

## 6. How the Pipeline Relates to the Overall Project

- **Laser‑Printing Workflow**: The notebook bridges the gap between a digital image and a physically printed laser‑etched poster. It accounts for *dot gain*, *paper texture*, and *density compensation*—critical factors for high‑quality laser prints.
- **Integration with Other Notebooks**: It re‑uses kernels and LUTs from other Jupyter notebooks (`jupyter/laser_settings.ipynb`, `jupyter/dither_experiment.ipynb`) and feeds its output into downstream scripts (`create_map_poster.py`, `simulate_cyanotype.py`).
- **Configuration Centralization**: Parameters are defined at the top, making it easy to adjust the pipeline for different paper sizes, DPI targets, or LUTs without digging into the code.

---

## 7. Potential Extensions & Tweaks

| Idea | Description |
|------|-------------|
| **Dynamic Power‑Level Bounds** | Replace the static `power_levels = np.array([0.0, 0, 0, 1.0], dtype=np.float32)` with a more granular set of levels for finer control. |
| **Adaptive Noise Scaling** | Scale the blue‑noise amplitude based on local image contrast to avoid over‑dithering in flat areas. |
| **GPU Acceleration** | Convert the convolution and resizing steps to use CuPy or Numba for large images. |
| **Batch Processing** | Wrap the pipeline into a function that can process a folder of images automatically. |
| **Parameter Sweep UI** | Expose sliders for `pre_adj_strength`, `pre_gamma`, and `runout_mm` using `ipywidgets` for interactive experimentation. |

---

## 8. Quick Reference Cheat‑Sheet

```python
# Top‑level config
input_img_path          = '../posters/IMG_2075-Enhanced-NR.png'
target_image_dpi        = 260
TARGET_LONGEST_EDGE_MM  = 270
runout_mm               = 2.0
percentile              = 99
pre_adj_strength        = 0.5
pre_gamma               = 1.0
lut_file                = 'uv_laser_correction_inverted.csv'
BLUE_NOISE_PATH         = '../bluenoise/128_128/LDR_LLL1_0.png'
kernel_file             = 'averaged_deconvolution_kernel.npy'
kernel_source_dpi       = 1200
```

```python
# Main call
corrected = apply_cyanotype_correction(
    image_path=input_img_path,
    kernel_path=kernel_file,
    lut_csv_path=lut_file,
    target_gamma=pre_gamma,
    strength=pre_adj_strength
)
cv2.imwrite('corrected_pre_dither.png', corrected)
```

---

**End of Summary**  