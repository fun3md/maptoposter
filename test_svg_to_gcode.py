#!/usr/bin/env python3
"""
Test script for svg_to_gcode.py

This script creates a simple SVG file with various shapes and colors,
then converts it to G-code using the svg_to_gcode.py script.
It also compares the results with and without path optimization.
"""

import os
import sys
import subprocess
import math
from xml.dom import minidom

def create_test_svg(filename="test.svg", width=100, height=100):
    """Create a test SVG file with various shapes and colors"""
    
    doc = minidom.getDOMImplementation().createDocument(None, "svg", None)
    svg = doc.documentElement
    
    # Set SVG attributes
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("width", str(width))
    svg.setAttribute("height", str(height))
    svg.setAttribute("viewBox", f"0 0 {width} {height}")
    
    # Create a rectangle
    rect = doc.createElement("rect")
    rect.setAttribute("x", "10")
    rect.setAttribute("y", "10")
    rect.setAttribute("width", "80")
    rect.setAttribute("height", "80")
    rect.setAttribute("fill", "none")
    rect.setAttribute("stroke", "#000000")
    rect.setAttribute("stroke-width", "1")
    svg.appendChild(rect)
    
    # Create a circle
    circle = doc.createElement("circle")
    circle.setAttribute("cx", "50")
    circle.setAttribute("cy", "50")
    circle.setAttribute("r", "30")
    circle.setAttribute("fill", "none")
    circle.setAttribute("stroke", "#666666")
    circle.setAttribute("stroke-width", "1")
    svg.appendChild(circle)
    
    # Create a line
    line = doc.createElement("line")
    line.setAttribute("x1", "10")
    line.setAttribute("y1", "10")
    line.setAttribute("x2", "90")
    line.setAttribute("y2", "90")
    line.setAttribute("stroke", "#999999")
    line.setAttribute("stroke-width", "1")
    svg.appendChild(line)
    
    # Create a path (triangle)
    path = doc.createElement("path")
    path.setAttribute("d", "M 10,90 L 50,10 L 90,90 Z")
    path.setAttribute("fill", "none")
    path.setAttribute("stroke", "#333333")
    path.setAttribute("stroke-width", "1")
    svg.appendChild(path)
    
    # Write to file
    with open(filename, "w") as f:
        f.write(doc.toprettyxml())
    
    print(f"Created test SVG file: {filename}")
    return filename

def calculate_travel_distance(gcode_file):
    """Calculate the total travel distance in the G-code file"""
    total_distance = 0
    prev_x, prev_y = 0, 0
    
    with open(gcode_file, "r") as f:
        for line in f:
            # Look for G0 (rapid move) commands
            if line.strip().startswith("G0 "):
                # Extract X and Y coordinates
                parts = line.strip().split()
                x, y = None, None
                
                for part in parts:
                    if part.startswith("X"):
                        x = float(part[1:])
                    elif part.startswith("Y"):
                        y = float(part[1:])
                
                if x is not None and y is not None:
                    # Calculate distance from previous position
                    distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                    total_distance += distance
                    
                    # Update previous position
                    prev_x, prev_y = x, y
    
    return total_distance

def main():
    # Create test SVG
    svg_file = create_test_svg()
    
    # Test both with and without optimization
    results = []
    
    for optimize in [True, False]:
        # Set output filename based on optimization setting
        output_suffix = "_optimized" if optimize else "_unoptimized"
        output_file = os.path.splitext(svg_file)[0] + output_suffix + ".nc"
        
        # Build command
        cmd = [
            sys.executable, 
            "svg_to_gcode.py", 
            svg_file, 
            "--output", output_file,
            "--min-power", "100", 
            "--max-power", "800"
        ]
        
        if not optimize:
            cmd.append("--no-optimize")
        
        # Run command
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            # Check if output file was created
            if os.path.exists(output_file):
                print(f"Successfully created G-code file: {output_file}")
                
                # Calculate travel distance
                travel_distance = calculate_travel_distance(output_file)
                
                # Print first few lines of G-code
                with open(output_file, "r") as f:
                    lines = f.readlines()
                    print(f"\nFirst 10 lines of G-code ({output_file}):")
                    for i, line in enumerate(lines[:10]):
                        print(f"{i+1:3d} | {line.strip()}")
                    print("...")
                    print(f"Total lines: {len(lines)}")
                    print(f"Total travel distance: {travel_distance:.2f} units")
                
                # Store results for comparison
                results.append({
                    "file": output_file,
                    "optimized": optimize,
                    "lines": len(lines),
                    "travel_distance": travel_distance
                })
            else:
                print(f"Error: G-code file not created: {output_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running svg_to_gcode.py: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
    
    # Compare results
    if len(results) == 2:
        opt = next(r for r in results if r["optimized"])
        unopt = next(r for r in results if not r["optimized"])
        
        distance_reduction = unopt["travel_distance"] - opt["travel_distance"]
        distance_percent = (distance_reduction / unopt["travel_distance"]) * 100 if unopt["travel_distance"] > 0 else 0
        
        print("\n=== Optimization Results ===")
        print(f"Unoptimized travel distance: {unopt['travel_distance']:.2f} units")
        print(f"Optimized travel distance: {opt['travel_distance']:.2f} units")
        print(f"Distance reduction: {distance_reduction:.2f} units ({distance_percent:.1f}%)")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()