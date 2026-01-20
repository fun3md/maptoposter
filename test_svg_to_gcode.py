#!/usr/bin/env python3
"""
Test script for svg_to_gcode.py

This script creates a simple SVG file with various shapes and colors,
then converts it to G-code using the svg_to_gcode.py script.
"""

import os
import sys
import subprocess
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

def main():
    # Create test SVG
    svg_file = create_test_svg()
    
    # Convert to G-code
    try:
        cmd = [sys.executable, "svg_to_gcode.py", svg_file, "--min-power", "100", "--max-power", "800"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Check if output file was created
        gcode_file = os.path.splitext(svg_file)[0] + ".nc"
        if os.path.exists(gcode_file):
            print(f"Successfully created G-code file: {gcode_file}")
            
            # Print first few lines of G-code
            with open(gcode_file, "r") as f:
                lines = f.readlines()
                print("\nFirst 10 lines of G-code:")
                for i, line in enumerate(lines[:10]):
                    print(f"{i+1:3d} | {line.strip()}")
                print("...")
                print(f"Total lines: {len(lines)}")
        else:
            print(f"Error: G-code file not created: {gcode_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running svg_to_gcode.py: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()