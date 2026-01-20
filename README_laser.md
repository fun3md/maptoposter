# SVG to G-code Converter for Laser Cutting

This tool converts SVG files to G-code for laser cutting machines. It extracts paths from SVG files, considers only the viewable area, and maps SVG colors to laser power settings.

## Features

- Converts SVG paths to G-code for laser cutting
- Maps SVG colors to laser power settings (darker colors = higher power)
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
                       [--reposition REPOSITION]
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

## How It Works

1. The script parses the SVG file to extract paths and their attributes
2. For each path, it:
   - Extracts the stroke color and maps it to a laser power setting
   - Converts the path segments to G-code commands
   - Handles curves by approximating them with line segments
3. The G-code uses:
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

## Limitations

- The script primarily handles path elements in SVG
- Text elements should be converted to paths in your SVG editor before processing
- Complex SVG features like filters, masks, or patterns are not supported