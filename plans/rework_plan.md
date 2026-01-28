# Rework Plan: Cyanotype Tone Calibration & Pre‑Compensation Pipeline

## Vision
Provide a robust, modular pipeline that:
- Measures UV + cyanotype response.
- Inverts the curve for 8‑bit input.
- Applies STRESS‑type local adaptation.
- Adds calibrated dithering.

## Current Implementation Overview
- **dither_v2.py**: Adaptive hybrid dithering, LUT mapping, edge‑damping, spread control.
- **cyano_stress.py**: Adds STRESS enhancement, gamma handling, similar adaptive logic.
- **jupyter/print_density_calibration.ipynb**: Interactive calibration UI (sliders) but not fully scripted.
- **luts/interpolated_output.csv**: Sample LUT (input → output) used for mapping.
- **apply_lut.py**: Helper to load and apply LUTs (likely used elsewhere).

## Identified Gaps
1. Calibration workflow is interactive only; no automated script to generate and fit curves.
2. LUT inversion logic is scattered; no dedicated inverse‑LUT generator.
3. STRESS implementation lacks explicit safety bounds and parameter validation.
4. Dithering parameters are hard‑coded; no configurability across hardware.
5. No clear separation of concerns (measurement, curve fitting, LUT creation, image processing).
6. Limited testing / validation against real prints.
7. Documentation is fragmented.

## Rework Objectives
- Decouple measurement, curve fitting, LUT generation, and image processing into independent modules.
- Provide a scripted calibration pipeline (generate targets → measure → fit → export LUT).
- Implement automatic inverse‑LUT creation for 8‑bit mapping.
- Refactor STRESS adaptation with configurable strength, sample count, and safety checks.
- Make dithering parameters (type, strength, spread) CLI‑configurable.
- Add comprehensive unit/integration tests.
- Update documentation and example notebooks.

## Step‑by‑Step Plan
- [ ] **Modularize LUT handling**
  - Create `lut_utils.py` with `load_lut`, `invert_lut`, `interpolate_lut`.
- - **Automate calibration**
  - Script `calibrate_cyanotype.py` to generate test targets (PNG/TIFF), load measurements, fit spline, export `inverse_lut.csv`.
- - **Integrate gamma & linearization**
  - Add `gamma_utils.py` for gamma apply/undo, ensure linear working space.
- - **Refactor STRESS adaptation**
  - Move `apply_stress_v2` into `stress.py` with explicit parameter validation.
  - Add safety clamp for envelope range to avoid division‑by‑zero.
- - **Parameterize dithering**
  - Introduce config file (`defaults.yaml`) for `min_spread`, `max_spread`, `strength`, `dither_type`.
  - Allow runtime override via CLI flags.
- - **Update main pipeline**
  - `process_image.py` orchestrates: load image → gamma → STRESS → LUT mapping → dither → output.
- - **Testing & Validation**
  - Write unit tests for each module.
  - Create synthetic test images to verify inversion accuracy.
  - Print test strips and compare measured density vs. target.
- - **Documentation**
  - Update README with workflow diagram.
  - Add example notebooks for calibration and printing.
- - **Release & Versioning**
  - Tag new version (e.g., v2.0).
  - Publish to internal package index or repo.

## Suggested Parameter Values (starting point)
- **Test target**: 0‑255 ramp + 0‑100 % steps, 300 dpi, 150 mm square.
- **Exposure**: min = just‑above “no mark”, max = just‑below reversal.
- **LUT interpolation**: cubic spline, monotonic constrained.
- **STRESS**: samples = 25, iterations = 10, strength = 0.4‑0.6.
- **Dither**: Bayer 8×8, spread 1‑4, edge damping enabled.

## Risks & Mitigations
- **Over‑exposure causing reversal** → enforce hard limits in exposure script.
- **LUT inversion errors** → validate monotonicity after inversion.
- **Parameter drift across materials** → store material‑specific config files.