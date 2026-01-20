#!/usr/bin/env python3
"""
SVG to G-code Converter for Laser Cutting

This script converts SVG files to G-code for laser cutting machines.
It extracts paths from the SVG, considers only the viewable area,
and maps SVG colors to laser power settings.
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from svgpathtools import svg2paths, Path, Line, CubicBezier, QuadraticBezier, Arc
import svgpathtools
from xml.dom import minidom

# Register the SVG namespace
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

def hex_to_rgb(hex_color):
    """Convert hex color to RGB values (0-255)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_grayscale(rgb):
    """Convert RGB to grayscale using standard luminance formula"""
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def color_to_power(color, min_power, max_power):
    """Map color to laser power
    
    Darker colors (lower grayscale values) get higher power
    """
    if color.startswith('#'):
        rgb = hex_to_rgb(color)
        gray = rgb_to_grayscale(rgb)
    else:
        # Handle named colors or other formats - default to medium power
        gray = 128
    
    # Normalize grayscale to 0-1 and invert (darker = higher power)
    power_factor = 1 - (gray / 255)
    
    # Scale to min-max power range
    power = min_power + power_factor * (max_power - min_power)
    return int(power)

def get_svg_viewbox(svg_file):
    """Extract viewBox from SVG file"""
    doc = minidom.parse(svg_file)
    svg_elem = doc.getElementsByTagName('svg')[0]
    
    if svg_elem.hasAttribute('viewBox'):
        viewbox = svg_elem.getAttribute('viewBox').split()
        return [float(x) for x in viewbox]
    
    # If no viewBox, try to get width and height
    width = float(svg_elem.getAttribute('width').replace('px', '')) if svg_elem.hasAttribute('width') else 100
    height = float(svg_elem.getAttribute('height').replace('px', '')) if svg_elem.hasAttribute('height') else 100
    
    return [0, 0, width, height]

def get_svg_dimensions(svg_file):
    """Get SVG dimensions from file"""
    doc = minidom.parse(svg_file)
    svg_elem = doc.getElementsByTagName('svg')[0]
    
    # Get viewBox if available
    viewbox = None
    if svg_elem.hasAttribute('viewBox'):
        viewbox = [float(x) for x in svg_elem.getAttribute('viewBox').split()]
    
    # Get width and height
    width = None
    height = None
    
    if svg_elem.hasAttribute('width'):
        width_str = svg_elem.getAttribute('width')
        width = float(re.sub(r'[^0-9.]', '', width_str))
    
    if svg_elem.hasAttribute('height'):
        height_str = svg_elem.getAttribute('height')
        height = float(re.sub(r'[^0-9.]', '', height_str))
    
    # Use viewBox dimensions if width/height not specified
    if viewbox and (width is None or height is None):
        if width is None:
            width = viewbox[2]
        if height is None:
            height = viewbox[3]
    
    return width, height, viewbox

def svg_to_gcode(svg_file, output_file, min_power=0, max_power=1000, feedrate=1000, reposition_speed=3000):
    """Convert SVG file to G-code for laser cutting"""
    
    # Get SVG dimensions and viewBox
    width, height, viewbox = get_svg_dimensions(svg_file)
    
    if viewbox:
        min_x, min_y, vb_width, vb_height = viewbox
    else:
        min_x, min_y = 0, 0
        vb_width, vb_height = width, height
    
    # Parse SVG paths
    paths, attributes = svg2paths(svg_file)
    
    # Open output file
    with open(output_file, 'w') as f:
        # Write G-code header
        f.write("G90 (use absolute coordinates)\n")
        f.write(f"G0 X0 Y0 S0\n")
        f.write(f"G1 M4 F{feedrate}\n")
        
        # Process each path
        for i, (path, attr) in enumerate(zip(paths, attributes)):
            # Get path color and calculate laser power
            stroke = attr.get('stroke', '#000000')
            power = color_to_power(stroke, min_power, max_power)
            
            # Skip paths with zero power
            if power <= min_power:
                continue
            
            # Get first point of path
            start = path.start
            start_x = start.real
            start_y = start.imag
            
            # Move to start position with laser off
            f.write(f"G0 X{start_x:.4f} Y{start_y:.4f} S0\n")
            f.write(f"M4\n")
            
            # Process each segment in the path
            for segment in path:
                if isinstance(segment, Line):
                    end_x = segment.end.real
                    end_y = segment.end.imag
                    f.write(f"G1 X{end_x:.4f} Y{end_y:.4f} S{power}\n")
                
                elif isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                    # Approximate curves with line segments
                    points = segment.point(np.linspace(0, 1, 10))
                    for point in points[1:]:  # Skip first point as it's the current position
                        x, y = point.real, point.imag
                        f.write(f"G1 X{x:.4f} Y{y:.4f} S{power}\n")
            
            # Turn off laser at end of path
            f.write(f"M5\n")
        
        # Write G-code footer
        f.write(f"M5 S{min_power} F{feedrate}\n")
        f.write("G0 X0 Y0 Z0 (move back to origin)\n")

def main():
    parser = argparse.ArgumentParser(description='Convert SVG to G-code for laser cutting')
    parser.add_argument('svg_file', help='Input SVG file')
    parser.add_argument('--output', '-o', help='Output G-code file (default: input file with .nc extension)')
    parser.add_argument('--min-power', type=int, default=0, help='Minimum laser power (default: 0)')
    parser.add_argument('--max-power', type=int, default=1000, help='Maximum laser power (default: 1000)')
    parser.add_argument('--feedrate', type=int, default=1000, help='Feedrate for cutting (default: 1000)')
    parser.add_argument('--reposition', type=int, default=3000, help='Speed for repositioning moves (default: 3000)')
    
    args = parser.parse_args()
    
    # Set default output filename if not specified
    if not args.output:
        base_name = os.path.splitext(args.svg_file)[0]
        args.output = f"{base_name}.nc"
    
    svg_to_gcode(
        args.svg_file, 
        args.output, 
        min_power=args.min_power,
        max_power=args.max_power,
        feedrate=args.feedrate,
        reposition_speed=args.reposition
    )
    
    print(f"Converted {args.svg_file} to {args.output}")
    print(f"Settings: min_power={args.min_power}, max_power={args.max_power}, feedrate={args.feedrate}")

if __name__ == "__main__":
    main()