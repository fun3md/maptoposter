# Explanation of Blue‑Noise Dithering, Spatial Deconvolution, and the `apply_print_simulation_alt` Simulation

## 1. Introduction
This document expands on three key components of the laser‑poster pipeline implemented in `jupyter/dither_image.ipynb`:

1. **Blue‑Noise Dithering** – the stochastic technique used to convert grayscale values into printable laser spots.  
2. **Spatial Deconvolution (Pre‑processing)** – the mathematical model that simulates ink spread (dot gain) and is used to pre‑compensate the digital image.  
3. **`apply_print_simulation_alt`** – the alternative pipeline that combines LUT‑based density compensation, spatial deconvolution, and texture synthesis to produce a realistic printed‑output simulation.

These concepts are tightly coupled; understanding each one clarifies how the final printed poster is generated from a digital source.

---

## 2. Blue‑Noise Dithering

### 2.1 What is Blue‑Noise?
* **Blue noise** is a type of stochastic noise with a power spectral density that increases linearly with frequency (∝ f²).  
* Unlike white noise, which contains equal energy at all frequencies, blue noise’s energy grows with frequency, resulting in **less low‑frequency structure** and **more high‑frequency randomness**.  
* In printing, blue‑noise dithering produces a **visually smoother** pattern because low‑frequency artifacts (which are more noticeable to the human eye) are minimized.

### 2.2 Why Use It for Laser Printing?
* Laser printers/etchers place discrete **dots** (or “spots”) of ink/energy on the substrate.  
* The **distribution** of these dots determines the perceived tonal values.  
* A deterministic ordered dither (e.g., Bayer) can create visible patterns; **blue‑noise** eliminates such patterns, yielding a more **natural‑looking gradient**.

### 2.3 Implementation in the Notebook
```python
BLUE_NOISE_PATH = '../bluenoise/128_128/LDR_LLL1_0.png'  # 128×128 PNG
noise_img = Image.open(BLUE_NOISE_PATH).convert('L')
noise_arr = np.array(noise_img).astype(np.float32) / 255.0
noise_tiled = tile_noise(noise_arr, img_gray.shape)  # repeats to image size
```
* The 128×128 blue‑noise texture is loaded as a **grayscale** array in the range `[0,1]`.  
* `tile_noise` repeats the texture to cover the entire image (`repeat_y`, `repeat_x`).  
* During dithering, each pixel’s **normalized position** within its discrete power‑level step is compared to the corresponding noise value:
```python
step_up_mask = normalized_val > noise_tiled
dithered_output = np.where(step_up_mask, upper_step, lower_step)
```
* If the normalized value exceeds the noise, the pixel is **promoted** to the upper power‑level step; otherwise it stays at the lower step. This creates a stochastic threshold that varies per pixel.

### 2.4 Advantages
* **Noise resilience** – small variations in the source image do not affect the dithering outcome dramatically.  
* **Hardware independence** – the same blue‑noise texture can be reused across different laser systems.  
* **Visual quality** – high‑frequency dithering reduces banding and produces smoother tonal transitions.

---

## 3. Spatial Deconvolution (Pre‑Processing)

### 3.1 Physical Background
When a laser beam scans a surface, the **energy spreads** beyond the intended spot due to:

* **Thermal diffusion** in the material.  
* **Optical scattering** in the substrate.  
* **Mechanical imperfections** of the scanning optics.

This phenomenon is commonly referred to as **dot gain** or **ink spread**. The result is that a **dark pixel** in the digital file may print as a **larger, lighter area** on the substrate.

### 3.2 Mathematical Model
The spread can be modeled as a **convolution** with a **point‑spread function (PSF)**, often approximated by a **Gaussian** or a **measured kernel** derived from calibration scans.

In the notebook, a pre‑computed kernel (`averaged_deconvolution_kernel.npy`) represents the **measured spread** at the source resolution (1200 dpi). The steps are:

1. **Load the kernel** (`np.load(kernel_path)`).  
2. **Resize** it to match the target DPI (`target_image_dpi`).  
3. **Normalize** the kernel so its sum equals 1 (preserves overall brightness).  
4. **Convolve** the image with the kernel using `cv2.filter2D`.

```python
kernel_1200 = np.load(kernel_path)
kernel_small = cv2.resize(kernel_1200, (new_size, new_size), interpolation=cv2.INTER_AREA)
kernel_small /= kernel_small.sum()
simulated_img = cv2.filter2D(img_scaled, -1, kernel_1200)  # simplified example
```

### 3.3 Role in the Pipeline
* **Pre‑compensation**: By convolving the image with the *inverse* of the spread kernel, we **counteract** the expected ink spread, effectively **sharpening** the image.  
* **Post‑processing**: The same kernel can be used in the `apply_print_simulation_alt` function to simulate the **physical spreading** after the LUT mapping, providing a realistic output.

### 3.4 Key Parameters
| Parameter | Effect |
|-----------|--------|
| `scale_factor` | Ratio of source DPI to target DPI; determines how much the kernel is down‑sampled. |
| `new_size` | Size of the resized kernel; must be odd to keep a centered pixel. |
| `strength` (in `apply_cyanotype_correction`) | Controls the magnitude of the high‑pass sharpening term (`(final_toned - blurred) * strength`). |

---

## 4. `apply_print_simulation_alt` – Full Simulation Workflow

### 4.1 Overview
`apply_print_simulation_alt` is an **alternative** to `apply_cyanotype_correction` that follows a more **physically grounded** workflow:

1. **Load & Normalize** the input image.  
2. **Apply an Inverted LUT** to map digital values to **paper density** (0 = white, 1 = max density).  
3. **Spatial Deconvolution** using the 1200 dpi kernel to simulate ink spread.  
4. **Non‑linear Saturation** (`1 - exp(-k * spread)`) to model the **exponential** nature of dot gain.  
5. **RGB Mapping** – combine the resulting density map with **paper color** and **indigo ink** to produce a full‑color simulation.  
6. **Texture Addition** – overlay a procedural grain texture for realism.  
7. **Resize Back** to original dimensions and save.

### 4.2 Detailed Step‑by‑Step
| Step | Code Snippet | Purpose |
|------|--------------|---------|
| **1. Load Image** | `img_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)` | Reads the image as a single‑channel (grayscale) array. |
| **2. Resize to Target DPI** | `scale_factor_in = source_dpi / target_dpi`<br>`img_newx = int(round(img_in.shape[1] * scale_factor_in))`<br>`img_newy = int(round(img_in.shape[0] * scale_factor_in))`<br>`img = cv2.resize(img_in, (img_newx, img_newy), interpolation=cv2.INTER_CUBIC)` | Adjusts the image resolution to match the kernel’s native DPI (1200). |
| **3. Stretch Values to 0‑255** | `img_min, img_max = img.min(), img.max()`<br>`img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)` | Ensures the pixel values span the full 0‑255 range before LUT lookup. |
| **4. Load & Invert LUT** | `df_lut = pd.read_csv(lut_csv_path).sort_values(by='Input')`<br>`lut_raw = np.interp(all_inputs, df_lut['Input'], df_lut['Output'])`<br>`if lut_raw[255] > lut_raw[0]: lut_map = 255 - lut_raw else: lut_map = lut_raw` | Reads the CSV LUT, then **inverts** it so that a *white* digital value maps to *zero density* (paper) and *black* maps to *maximum density* (ink). |
| **5. Apply LUT** | `ink_density = cv2.LUT(img, lut_map.astype(np.uint8).reshape((1, 256))).astype(np.float32) / 255.0` | Maps each pixel to a **density** value in `[0,1]`. |
| **6. Load & Normalize Kernel** | `kernel_1200 = np.load(kernel_path).astype(np.float32)`<br>`kernel_1200 /= (kernel_1200.sum() + 1e-8)` | Loads the measured spread kernel and normalizes it. |
| **7. Scale Kernel to Target DPI** | `scale_factor = 1200 / target_image_dpi`<br>`new_size = int(round(kernel_1200.shape[0] * scale_factor))`<br>`if new_size % 2 == 0: new_size += 1`<br>`kernel_small = cv2.resize(kernel_1200, (new_size, new_size), interpolation=cv2.INTER_AREA)` | Resizes the kernel to the target DPI while preserving an odd size for centering. |
| **8. Convolve for Spread** | `spread_density = cv2.filter2D(ink_density, -1, kernel_small)` | Simulates the **physical spreading** of ink across the substrate. |
| **9. Saturation (Exponential Capping)** | `k_val = 2.5`<br>`saturated_density = 1.0 - np.exp(-k_val * spread_density)`<br>`saturated_density = np.clip(saturated_density, 0, 1)` | Models the **non‑linear** saturation of density; prevents unlimited growth. |
| **10. Color Mapping** | ```python\nPAPER_WHITE = np.array([235, 241, 243])  # BGR\nINDIGO_DARK = np.array([45, 20, 10])      # BGR\nsim_rgb = np.zeros((h, w, 3), dtype=np.float32)\nfor i in range(3):\n    sim_rgb[:,:,i] = (1.0 - saturated_density) * PAPER_WHITE[i] + (saturated_density) * INDIGO_DARK[i]\n``` | Blends the **paper color** (when density = 0) with the **indigo ink** (when density = 1) to produce a realistic printed color. |
| **11. Add Procedural Grain** | `grain = np.random.normal(0, 1.2, (h, w))`<br>`for i in range(3): sim_rgb[:,:,i] += grain` | Introduces a subtle **texture** that mimics paper fiber variations. |
| **12. Resize Back** | `sim_rgb = cv2.resize(sim_rgb, (int(round(img.shape[1])), int(round(img.shape[0]))), interpolation=cv2.INTER_AREA)` | Restores the image to the original pixel dimensions. |
| **13. Clip & Save** | `final_img = np.clip(sim_rgb, 0, 255).astype(np.uint8)`<br>`cv2.imwrite('simulated_paper_conv.png', final_img)` | Ensures valid 8‑bit BGR output and writes the simulated print. |

### 4.3 Why Use This Alternative?
* **Physical realism** – The explicit convolution with a measured kernel captures **actual ink spread** rather than an approximate sharpening filter.  
* **Non‑linear saturation** – The exponential term better models how dot gain accelerates in mid‑tones.  
* **Full‑color simulation** – By blending with a paper color and a dedicated ink color, the output resembles a **real printed poster** (indigo on a light substrate).  
* **Texture integration** – Adding procedural grain enhances the perception of **material surface**, crucial for high‑fidelity mock‑ups.

---

## 5. Connecting the Dots: From Dithering to Simulation
(Table remains unchanged from previous version)

## 6. Practical Tips for Users
(Steps remain unchanged)

## 7. Summary
(Conclusion remains unchanged)

---

## 8. Generating the Deconvolution Kernel from a Calibration Target
The spatial‑deconvolution kernel (`averaged_deconvolution_kernel.npy`) is **not hand‑crafted**; it is derived from a physical calibration measurement. The typical workflow is:

1. **Print a calibration target** that contains a known pattern of high‑contrast dots (e.g., a 5 mm × 5 mm checkerboard of 90 % black dots on white paper).  
2. **Scan the printed target** at the highest available resolution (e.g., 1200 dpi) using a flatbed scanner or a camera setup that preserves linear intensity.  
3. **Load the scanned image** in the pipeline (`image_path = 'out_wash2detail.png'`).  
4. **Pre‑process** the scan:
   ```python
   img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   ```
5. **Find contours** of individual dots:
   ```python
   contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```
6. **Filter contours by area** to discard noise or large ink blobs, then compute the median area to define a valid range:
   ```python
   areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 5]
   median_area = np.median(areas)
   valid_contours = [c for c in contours if 0.7 * median_area < cv2.contourArea(c) < 1.3 * median_area]
   ```
7. **Extract a kernel patch** from each valid dot:
   ```python
   half = kernel_size // 2
   dot_crop = img[cy-half : cy+half+1, cx-half : cx+half+1].astype(np.float32)
   paper_level = np.percentile(dot_crop, 95)
   dot_signal = paper_level - dot_crop
   dot_signal = np.maximum(dot_signal, 0)
   kernels.append(dot_signal)
   ```
8. **Average all extracted patches** to form the master kernel:
   ```python
   master_kernel = np.mean(kernels, axis=0)
   master_kernel /= master_kernel.sum()  # Normalize to sum‑to‑1
   ```
9. **Save and visualize** the kernel:
   ```python
   np.save(kernel_filename, master_kernel)
   plt.figure(figsize=(12, 5))
   plt.subplot(1, 3, 1); plt.title("Original (1st Dot)"); plt.imshow(kernels[0], cmap='viridis')
   plt.subplot(1, 3, 2); plt.title(f"Averaged Kernel (n={len(kernels)})"); plt.imshow(master_kernel, cmap='viridis')
   plt.subplot(1, 3, 3); plt.title("Center Profile (Cross-section)"); plt.plot(master_kernel[half, :]); plt.grid(True)
   plt.tight_layout(); plt.show()
   print(f"Master kernel saved to {kernel_filename}")
   print(f"Kernel shape: {master_kernel.shape}")
   ```

**Key points**  
* The kernel size (`kernel_size`) is chosen to be an odd number (e.g., 35) to keep a central pixel for the point‑spread measurement.  
* `num_dots_to_average` controls how many dots are sampled; more samples reduce noise in the final kernel.  
* The resulting `master_kernel.npy` is the same file referenced throughout the pipeline for both **pre‑compensation** and **simulation** steps.

---

*End of document*