#!/usr/bin/env python3
"""
SVG to G-code Converter for Laser Cutting

This script converts SVG files to G-code for laser cutting machines.
It extracts paths from the SVG, considers only the viewable area,
maps SVG colors to laser power settings, and optimizes path order
to minimize travel distances.

Path Optimization Features:
- Fast: Uses a greedy nearest-neighbor algorithm to find the next closest path
- Balanced: Uses nearest-neighbor followed by limited 2-opt improvement (default)
- Thorough: Uses nearest-neighbor followed by full 2-opt improvement until convergence

The 2-opt algorithm significantly improves path optimization by repeatedly
swapping path segments when it reduces the total travel distance, resulting
in shorter cutting times and less machine wear.
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from svgpathtools import svg2paths, Path, Line, CubicBezier, QuadraticBezier, Arc
import svgpathtools
from xml.dom import minidom
import math

# Register the SVG namespace
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

def hex_to_rgb(hex_color):
    """Convert hex color to RGB values (0-255)"""
    # Handle empty strings or invalid inputs
    if not hex_color or not isinstance(hex_color, str):
        return (0, 0, 0)  # Default to black
        
    hex_color = hex_color.lstrip('#')
    
    # Check if we have a valid hex color after stripping #
    if len(hex_color) != 6:
        return (0, 0, 0)  # Default to black for invalid hex
        
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        # If conversion fails, return black
        return (0, 0, 0)

def rgb_to_grayscale(rgb):
    """Convert RGB to grayscale using standard luminance formula"""
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def color_to_power(color, min_power, max_power):
    """Map color to laser power
    
    Darker colors (lower grayscale values) get higher power
    """
    # Handle None or empty string
    if not color:
        # Default to medium power
        return int(min_power + (max_power - min_power) / 2)
        
    try:
        if isinstance(color, str) and color.startswith('#'):
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
    except Exception:
        # If any error occurs, return medium power as a fallback
        return int(min_power + (max_power - min_power) / 2)

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

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def optimize_path_order(paths, attributes, min_power, optimization_level='balanced'):
    """Optimize the order of paths to minimize travel distance
    
    Uses a nearest-neighbor algorithm followed by 2-opt improvement
    
    Args:
        paths: List of SVG paths
        attributes: List of path attributes
        min_power: Minimum laser power threshold
        optimization_level: 'fast', 'balanced', or 'thorough'
    
    Returns:
        List of (path, attribute) tuples in optimized order
    """
    # Step 1: Filter paths and prepare path data
    path_data = prepare_path_data(paths, attributes, min_power)
    
    if not path_data:
        return []
    
    # Step 2: Generate initial solution using nearest neighbor
    print(f"Optimizing path order for {len(path_data)} paths...")
    
    # Get initial solution using nearest neighbor
    optimized_indices = nearest_neighbor(path_data)
    
    # Calculate initial tour distance
    initial_distance = calculate_tour_distance(path_data, optimized_indices)
    print(f"Initial path length (nearest neighbor): {initial_distance:.2f} units")
    
    # Step 3: Improve solution using 2-opt if not using 'fast' optimization
    if optimization_level != 'fast':
        # Set iteration limits based on optimization level
        if optimization_level == 'balanced':
            max_iterations = min(1000, len(path_data) * 10)  # Reasonable limit for balanced mode
        else:  # thorough
            max_iterations = None  # No limit for thorough mode
        
        # Apply 2-opt improvement
        optimized_indices = two_opt(path_data, optimized_indices, max_iterations)
        
        # Calculate improved distance
        final_distance = calculate_tour_distance(path_data, optimized_indices)
        improvement = (initial_distance - final_distance) / initial_distance * 100
        print(f"Optimized path length (2-opt): {final_distance:.2f} units")
        print(f"Improvement: {improvement:.2f}%")
    
    # Step 4: Convert indices back to path-attribute pairs
    optimized_paths = [
        (path_data[i]['path'], path_data[i]['attr'])
        for i in optimized_indices
    ]
    
    return optimized_paths

def prepare_path_data(paths, attributes, min_power):
    """Filter out paths with power <= min_power and prepare path data"""
    path_data = []
    for i, (path, attr) in enumerate(zip(paths, attributes)):
        # Get stroke color with fallback to black
        stroke = attr.get('stroke', '#000000') if attr else '#000000'
        
        # Skip paths with no stroke or 'none' stroke
        if not stroke or stroke.lower() == 'none':
            continue
            
        power = color_to_power(stroke, min_power, 1000)  # Use any max_power, we just need to check min
        
        if power > min_power:
            start_point = (path.start.real, path.start.imag)
            end_point = (path.end.real, path.end.imag)
            path_data.append({
                'index': i,
                'path': path,
                'attr': attr,
                'start': start_point,
                'end': end_point
            })
    
    return path_data

def nearest_neighbor(path_data):
    """Generate initial solution using nearest neighbor algorithm"""
    n = len(path_data)
    if n == 0:
        return []
    
    # Start from origin
    current_point = (0, 0)
    unvisited = list(range(n))
    tour = []
    
    while unvisited:
        # Find the closest unvisited path
        closest_idx = -1
        min_distance = float('inf')
        
        for i in unvisited:
            # Calculate distance from current point to path start
            distance = calculate_distance(current_point, path_data[i]['start'])
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # Add the closest path to the tour
        tour.append(closest_idx)
        unvisited.remove(closest_idx)
        
        # Update current point to the end of the path we just processed
        current_point = path_data[closest_idx]['end']
    
    return tour

def calculate_tour_distance(path_data, tour):
    """Calculate the total distance of a tour"""
    if not tour:
        return 0
    
    total_distance = 0
    
    # Distance from origin to first path
    total_distance += calculate_distance((0, 0), path_data[tour[0]]['start'])
    
    # Distance between consecutive paths
    for i in range(len(tour) - 1):
        current_path = path_data[tour[i]]
        next_path = path_data[tour[i + 1]]
        total_distance += calculate_distance(current_path['end'], next_path['start'])
    
    return total_distance

def two_opt(path_data, tour, max_iterations=None):
    """Improve tour using 2-opt algorithm
    
    Args:
        path_data: List of path data dictionaries
        tour: Initial tour as a list of indices
        max_iterations: Maximum number of iterations (None for unlimited)
    
    Returns:
        Improved tour
    """
    n = len(tour)
    if n <= 2:
        return tour  # No improvement possible for 0, 1 or 2 paths
    
    # Calculate the total distance of the initial tour
    best_distance = calculate_tour_distance(path_data, tour)
    improvement = True
    iteration = 0
    
    # Continue until no improvement is found or max iterations reached
    while improvement and (max_iterations is None or iteration < max_iterations):
        improvement = False
        iteration += 1
        
        # Progress reporting for long-running optimizations
        if iteration % 100 == 0:
            print(f"2-opt iteration {iteration}, current distance: {best_distance:.2f}")
        
        # Try all possible 2-opt swaps
        for i in range(n - 1):
            # Use first improvement strategy for efficiency
            for j in range(i + 2, n):
                # Create new tour with 2-opt swap: reverse the segment between i and j
                new_tour = tour.copy()
                new_tour[i+1:j+1] = reversed(tour[i+1:j+1])
                
                # Calculate the distance of the new tour
                new_distance = calculate_tour_distance(path_data, new_tour)
                
                # If the new tour is better, accept it
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improvement = True
                    break  # Break inner loop to restart with the new tour
            
            if improvement:
                break  # Break outer loop to restart with the new tour
    
    if max_iterations is not None and iteration >= max_iterations:
        print(f"Reached maximum iterations ({max_iterations})")
    else:
        print(f"2-opt converged after {iteration} iterations")
    
    return tour

def svg_to_gcode(svg_file, output_file, min_power=0, max_power=1000, feedrate=1000,
                reposition_speed=3000, optimize=True, optimization_level='balanced'):
    """Convert SVG file to G-code for laser cutting
    
    Args:
        svg_file: Input SVG file path
        output_file: Output G-code file path
        min_power: Minimum laser power (default: 0)
        max_power: Maximum laser power (default: 1000)
        feedrate: Feedrate for cutting (default: 1000)
        reposition_speed: Speed for repositioning moves (default: 3000)
        optimize: Whether to optimize path order (default: True)
        optimization_level: 'fast', 'balanced', or 'thorough' (default: 'balanced')
    """
    import datetime
    import os
    
    # Get SVG dimensions and viewBox
    print(f"Reading SVG dimensions from {svg_file}...")
    width, height, viewbox = get_svg_dimensions(svg_file)
    
    if viewbox:
        min_x, min_y, vb_width, vb_height = viewbox
    else:
        min_x, min_y = 0, 0
        vb_width, vb_height = width, height
    
    print(f"SVG dimensions: {width}x{height}, ViewBox: {viewbox}")
    
    # Parse SVG paths
    print(f"Parsing SVG paths...")
    paths, attributes = svg2paths(svg_file)
    print(f"Found {len(paths)} paths in SVG file")
    
    # Initialize statistics
    stats = {
        'total_paths': len(paths),
        'processed_paths': 0,
        'skipped_paths': 0,
        'total_travel_distance': 0,
        'total_cutting_distance': 0,
        'start_time': datetime.datetime.now(),
    }
    
    # Optimize path order if requested
    if optimize:
        print(f"Optimizing path order using '{optimization_level}' strategy...")
        path_attr_pairs = optimize_path_order(paths, attributes, min_power, optimization_level)
    else:
        print("Path optimization disabled, using original path order")
        path_attr_pairs = list(zip(paths, attributes))
    
    # Count paths that will be processed (power > min_power)
    valid_paths = 0
    for _, attr in path_attr_pairs:
        stroke = attr.get('stroke', '#000000') if attr else '#000000'
        if stroke and stroke.lower() != 'none':
            power = color_to_power(stroke, min_power, max_power)
            if power > min_power:
                valid_paths += 1
    
    stats['valid_paths'] = valid_paths
    print(f"Processing {valid_paths} valid paths (with power > {min_power})")
    
    # Open output file
    print(f"Writing G-code to {output_file}...")
    with open(output_file, 'w') as f:
        # Write G-code header with information
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        svg_filename = os.path.basename(svg_file)
        
        f.write("; G-code generated from SVG file\n")
        f.write(f"; Source: {svg_filename}\n")
        f.write(f"; Date: {timestamp}\n")
        f.write(f"; Settings: min_power={min_power}, max_power={max_power}, feedrate={feedrate}, reposition_speed={reposition_speed}\n")
        f.write(f"; Optimization: {'enabled (' + optimization_level + ')' if optimize else 'disabled'}\n")
        f.write(f"; SVG dimensions: {width}x{height}, ViewBox: {viewbox}\n")
        f.write(f"; Total paths: {stats['total_paths']}, Valid paths: {stats['valid_paths']}\n")
        f.write("\n")
        f.write("G90 (use absolute coordinates)\n")
        f.write(f"G0 X0 Y0 S0\n")
        f.write(f"G1 M4 F{feedrate}\n")
        
        # Process each path in the optimized order
        path_count = 0
        previous_end_x = 0
        previous_end_y = 0
        
        for path, attr in path_attr_pairs:
            # Get path color and calculate laser power
            stroke = attr.get('stroke', '#000000') if attr else '#000000'
            
            # Skip paths with no stroke or 'none' stroke
            if not stroke or stroke.lower() == 'none':
                stats['skipped_paths'] += 1
                continue
                
            power = color_to_power(stroke, min_power, max_power)
            
            # Skip paths with zero power
            if power <= min_power:
                stats['skipped_paths'] += 1
                continue
            
            # Get first point of path
            start = path.start
            start_x = start.real
            start_y = start.imag
            
            # Update path counter
            path_count += 1
            stats['processed_paths'] += 1
            
            # Add informational comment about the current path
            f.write(f"\n; Path {path_count}/{stats['valid_paths']} - Power: {power} ({int(power/max_power*100)}%)\n")
            
            # Calculate travel distance to this path
            if path_count > 1:
                # Calculate distance from last position to current start
                last_pos = (previous_end_x, previous_end_y)
                current_pos = (start_x, start_y)
                travel_distance = calculate_distance(last_pos, current_pos)
                stats['total_travel_distance'] += travel_distance
                f.write(f"; Travel distance: {travel_distance:.2f} units\n")
            
            # Move to start position with laser off
            f.write(f"G0 X{start_x:.4f} Y{start_y:.4f} S0 F{reposition_speed}\n")
            f.write(f"G1 F{feedrate}\n")
            f.write(f"M4\n")
            
            # Initialize path cutting distance
            path_cutting_distance = 0
            
            # Process each segment in the path
            for segment in path:
                if isinstance(segment, Line):
                    end_x = segment.end.real
                    end_y = segment.end.imag
                    
                    # Calculate segment length
                    segment_length = calculate_distance((start_x, start_y), (end_x, end_y))
                    path_cutting_distance += segment_length
                    
                    f.write(f"G1 X{end_x:.4f} Y{end_y:.4f} S{power}\n")
                    
                    # Update start position for next segment
                    start_x, start_y = end_x, end_y
                
                elif isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                    # Approximate curves with line segments
                    points = segment.point(np.linspace(0, 1, 10))
                    prev_x, prev_y = start_x, start_y
                    
                    for point in points[1:]:  # Skip first point as it's the current position
                        x, y = point.real, point.imag
                        
                        # Calculate segment length
                        segment_length = calculate_distance((prev_x, prev_y), (x, y))
                        path_cutting_distance += segment_length
                        
                        f.write(f"G1 X{x:.4f} Y{y:.4f} S{power}\n")
                        
                        # Update previous position
                        prev_x, prev_y = x, y
                    
                    # Update start position for next segment
                    start_x, start_y = prev_x, prev_y
            
            # Add cutting distance information
            stats['total_cutting_distance'] += path_cutting_distance
            f.write(f"; Cutting distance: {path_cutting_distance:.2f} units\n")
            
            # Turn off laser at end of path
            f.write(f"M5\n")
            
            # Store the end position of this path for calculating travel to next path
            previous_end_x, previous_end_y = start_x, start_y
        
        # Write G-code footer
        f.write(f"M5 S{min_power} F{feedrate}\n")
        f.write(f"G0 X0 Y0 Z0 F{reposition_speed} (move back to origin)\n")

def main():
    parser = argparse.ArgumentParser(description='Convert SVG to G-code for laser cutting')
    parser.add_argument('svg_file', help='Input SVG file')
    parser.add_argument('--output', '-o', help='Output G-code file (default: input file with .nc extension)')
    parser.add_argument('--min-power', type=int, default=0, help='Minimum laser power (default: 0)')
    parser.add_argument('--max-power', type=int, default=1000, help='Maximum laser power (default: 1000)')
    parser.add_argument('--feedrate', type=int, default=1000, help='Feedrate for cutting (default: 1000)')
    parser.add_argument('--reposition', type=int, default=3000, help='Speed for repositioning moves (default: 3000)')
    parser.add_argument('--no-optimize', action='store_true', help='Disable path optimization (default: optimization enabled)')
    parser.add_argument('--optimize-level', choices=['fast', 'balanced', 'thorough'], default='balanced',
                        help='Optimization level: fast (nearest-neighbor only), balanced (limited 2-opt), '
                             'or thorough (full 2-opt) (default: balanced)')
    
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
        reposition_speed=args.reposition,
        optimize=not args.no_optimize,
        optimization_level=args.optimize_level
    )
    
    print(f"Converted {args.svg_file} to {args.output}")
    print(f"Settings: min_power={args.min_power}, max_power={args.max_power}, feedrate={args.feedrate}")
    
    if args.no_optimize:
        print("Path optimization: disabled")
    else:
        print(f"Path optimization: enabled (level: {args.optimize_level})")

if __name__ == "__main__":
    main()