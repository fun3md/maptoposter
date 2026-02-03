1. Hardware & Physical Setup
Machine: Ortur/Twotrees TTS-55 Pro (5.5W Diode).
Focus Strategy: -2mm Defocus (moving the focal point below the surface).
Resulting Spot Size: Expanded to ~0.18mm (from the native 0.08mm). This creates a "fat" beam ideal for deep, smooth engraving but requires lower resolution to prevent over-charring.
2. Motion & Pulse Settings (The "Stability" Fix)
To eliminate interference lines (banding) at high speeds, the PWM frequency must be high enough to ensure pulse overlap.
Feedrate: 4000 mm/min (66.6 mm/sec).
Optimized PWM Frequency: 3000 Hz.
Logic: At 3000 Hz, the laser pulses every 0.022mm. With a 0.18mm spot, each point on the wood is hit by ~8 consecutive pulses, ensuring a perfectly smooth line with no "dotted" appearance.
3. Resolution & DPI (The "Detail" Fix)
Recommended Resolution: 180 DPI (0.141mm Line Interval).
Why not 300 DPI? With a 0.18mm spot, 300 DPI causes massive overlap (burning the same spot twice). This results in "plowing" trenches and losing all image detail. 180 DPI allows the "fat" beam to create a smooth, flat-bottomed engrave.
4. Calibrated Power Range (S-Value 0–1000)
Based on the provided correction CSVs, the laser has a specific "Ignition" (starts burning) and "Saturation" (max black) point. Because we lowered the DPI to 180, we must increase the power by 1.67x to maintain darkness.
Min Power (S-Value): 70
Note: While the laser technically ignites at 26–43, 70 is recommended for electrical stability to ensure the beam doesn't flicker during light grayscale areas.
Max Power (S-Value): 382
Note: Any power higher than this at 4000mm/min will likely result in excessive charring/soot rather than a darker black.
5. Key Notes for Context Transfer
The Scalability Rule: If you increase speed (e.g., to 6000 mm/min), you must increase PWM Frequency (to 4000 Hz) and increase Max Power proportionally to maintain the same burn depth.
The DPI/Power Relationship: Lowering DPI (moving lines further apart) requires higher power per line to achieve the same visual density.
Software Implementation:
In LightBurn, set "Line Interval" to 0.141mm.
Set the Layer "Min Power" to 7% and "Max Power" to 38.2%.
Ensure $30=1000 in GRBL settings to match this math.
Final "Gold" Settings Table
Parameter	Value
Speed	4000 mm/min
PWM Frequency	3000 Hz
DPI	180 (0.141 mm interval)
S-Min	70
S-Max	382
Z-Offset	-2.0 mm