# SVG to G-code Converter for Laser Cutting

This tool converts SVG files to G-code for laser cutting machines. It extracts paths from SVG files, considers only the viewable area, maps SVG colors to laser power settings, and optimizes path order to minimize travel distances.

## Features

- Converts SVG paths to G-code for laser cutting
- Maps SVG colors to laser power settings (darker colors = higher power)
- Advanced path optimization to minimize cutting time and machine wear
- Multiple optimization levels for different needs (fast, balanced, thorough)
- Configurable laser power range (min/max)
- Configurable feedrate for cutting operations
- Configurable speed for repositioning moves
- Handles lines and curves (approximating curves with line segments)
- Respects the SVG viewBox for proper scaling

## Installation

1. Ensure you have Python 3.6+ installed
2. Install required dependencies:

```bash
pip install -r requirements_laser.txt
```

## Usage

Basic usage:

```bash
python svg_to_gcode.py your_file.svg
```

This will create a G-code file with the same name as your SVG file but with a `.nc` extension.

### Command Line Options

```
usage: svg_to_gcode.py [-h] [--output OUTPUT] [--min-power MIN_POWER]
                       [--max-power MAX_POWER] [--feedrate FEEDRATE]
                       [--reposition REPOSITION] [--no-optimize]
                       [--optimize-level {fast,balanced,thorough}]
                       svg_file

Convert SVG to G-code for laser cutting

positional arguments:
  svg_file              Input SVG file

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output G-code file (default: input file with .nc extension)
  --min-power MIN_POWER
                        Minimum laser power (default: 0)
  --max-power MAX_POWER
                        Maximum laser power (default: 1000)
  --feedrate FEEDRATE   Feedrate for cutting (default: 1000)
  --reposition REPOSITION
                        Speed for repositioning moves (default: 3000)
  --no-optimize         Disable path optimization (default: optimization enabled)
  --optimize-level {fast,balanced,thorough}
                        Optimization level: fast (nearest-neighbor only), balanced (limited 2-opt),
                        or thorough (full 2-opt) (default: balanced)
```

### Examples

Convert with custom power settings:

```bash
python svg_to_gcode.py design.svg --min-power 100 --max-power 800
```

Specify output file and feedrate:

```bash
python svg_to_gcode.py design.svg -o output.nc --feedrate 1500
```

Use thorough path optimization for complex designs:

```bash
python svg_to_gcode.py complex_design.svg --optimize-level thorough
```

Use fast optimization for simple designs or quick previews:

```bash
python svg_to_gcode.py simple_design.svg --optimize-level fast
```

Disable path optimization completely:

```bash
python svg_to_gcode.py design.svg --no-optimize
```

## How It Works

1. The script parses the SVG file to extract paths and their attributes
2. It optimizes the path order to minimize travel distances:
   - Filters out paths with power <= min_power
   - Generates an initial solution using the nearest-neighbor algorithm
   - Improves the solution using the 2-opt algorithm (for balanced and thorough modes)
   - Reports optimization statistics (initial path length, optimized path length, improvement percentage)
3. For each path in the optimized order, it:
   - Extracts the stroke color and maps it to a laser power setting
   - Converts the path segments to G-code commands
   - Handles curves by approximating them with line segments
4. The G-code uses:
   - `G0` for rapid positioning moves (laser off)
   - `G1` for cutting moves (laser on with specified power)
   - `S` parameter to control laser power (0-1000)
   - `M4` to turn the laser on
   - `M5` to turn the laser off

## G-code Format

The generated G-code follows this format:

```
G90 (use absolute coordinates)
G0 X0 Y0 S0
G1 M4 F1000
... (cutting operations) ...
M5 S0 F1000
G0 X0 Y0 Z0 (move back to origin)
```

## Color to Power Mapping

The script maps SVG colors to laser power using the following approach:

1. Convert the color to grayscale using standard luminance formula
2. Invert the grayscale value (darker colors get higher power)
3. Scale to the specified min-max power range

This means:
- Black (`#000000`) will use maximum laser power
- White (`#FFFFFF`) will use minimum laser power
- Grayscale values in between will be mapped proportionally

## Path Optimization

The script includes advanced path optimization to minimize travel distances, which reduces cutting time and machine wear. Three optimization levels are available:

### Fast (Nearest-Neighbor)
- Uses a greedy nearest-neighbor algorithm
- For each step, finds the path with the closest starting point to the current position
- Quick to compute but may produce suboptimal paths
- Best for simple designs or quick previews

### Balanced (Default)
- Uses nearest-neighbor to generate an initial solution
- Applies 2-opt improvement with a reasonable iteration limit
- Good balance between computation time and path quality
- Suitable for most designs

### Thorough
- Uses nearest-neighbor to generate an initial solution
- Applies full 2-opt improvement until convergence (no iteration limit)
- Produces the highest quality paths but may take longer to compute
- Best for complex designs where cutting time is critical

The 2-opt algorithm works by repeatedly finding pairs of path segments that, when swapped, reduce the total travel distance. This can significantly improve path quality compared to the nearest-neighbor algorithm alone.

## Limitations

- The script primarily handles path elements in SVG
- Text elements should be converted to paths in your SVG editor before processing
- Complex SVG features like filters, masks, or patterns are not supported