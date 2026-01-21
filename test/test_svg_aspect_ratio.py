#!/usr/bin/env python3
"""
Test script to analyze SVG aspect ratio calculation in create_map_poster.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def test_aspect_ratio_calculation():
    """Test the aspect ratio calculation logic from create_map_poster.py"""
    
    print("Testing SVG aspect ratio calculation...")
    
    # Simulate the current logic from create_map_poster.py lines 322-371
    # Create a square temporary figure (this is the problem!)
    temp_fig, temp_ax = plt.subplots(figsize=(10, 10), facecolor='white')
    temp_ax.set_facecolor('white')
    temp_ax.set_position([0, 0, 1, 1])
    
    # Simulate plotting some map data bounds (simulating urban area)
    # Let's say the natural map data has a landscape aspect ratio
    map_data_bounds = {
        'x_min': -5000, 'x_max': 5000,  # 10km wide
        'y_min': -3000, 'y_max': 3000   # 6km tall
    }
    
    # Set the bounds to simulate what would happen with real map data
    temp_ax.set_xlim(map_data_bounds['x_min'], map_data_bounds['x_max'])
    temp_ax.set_ylim(map_data_bounds['y_min'], map_data_bounds['y_max'])
    
    # Get the data bounds (this is what the current code does)
    xlim = temp_ax.get_xlim()
    ylim = temp_ax.get_ylim()
    data_width = xlim[1] - xlim[0]
    data_height = ylim[1] - ylim[0]
    aspect_ratio = data_width / data_height
    
    print(f"Data bounds: xlim={xlim}, ylim={ylim}")
    print(f"Data width: {data_width}, height: {data_height}")
    print(f"Calculated aspect ratio: {aspect_ratio:.3f}")
    
    # Simulate the size calculation (from lines 357-366)
    svg_size_mm = 300  # Example size
    size_inches = svg_size_mm / 25.4  # Convert to inches
    
    print(f"Target size: {svg_size_mm}mm = {size_inches:.2f} inches")
    
    # Current logic from the code
    if aspect_ratio >= 1:
        new_width = size_inches
        new_height = size_inches / aspect_ratio
    else:
        new_height = size_inches
        new_width = size_inches * aspect_ratio
    
    print(f"Calculated figure size: {new_width:.2f} x {new_height:.2f} inches")
    print(f"Final aspect ratio: {new_width/new_height:.3f}")
    
    plt.close(temp_fig)
    
    return aspect_ratio, new_width, new_height

def test_improved_aspect_ratio():
    """Test an improved aspect ratio calculation"""
    
    print("\n" + "="*50)
    print("Testing IMPROVED aspect ratio calculation...")
    
    # The issue: we should use a non-square temporary figure
    # Let's use a landscape temporary figure instead
    temp_fig, temp_ax = plt.subplots(figsize=(12, 8), facecolor='white')  # Landscape
    temp_ax.set_facecolor('white')
    temp_ax.set_position([0, 0, 1, 1])
    
    # Same map data bounds
    map_data_bounds = {
        'x_min': -5000, 'x_max': 5000,  # 10km wide
        'y_min': -3000, 'y_max': 3000   # 6km tall
    }
    
    temp_ax.set_xlim(map_data_bounds['x_min'], map_data_bounds['x_max'])
    temp_ax.set_ylim(map_data_bounds['y_min'], map_data_bounds['y_max'])
    
    # Get the data bounds
    xlim = temp_ax.get_xlim()
    ylim = temp_ax.get_ylim()
    data_width = xlim[1] - xlim[0]
    data_height = ylim[1] - ylim[0]
    aspect_ratio = data_width / data_height
    
    print(f"Data bounds: xlim={xlim}, ylim={ylim}")
    print(f"Data width: {data_width}, height: {data_height}")
    print(f"Calculated aspect ratio: {aspect_ratio:.3f}")
    
    # Same size calculation
    svg_size_mm = 300
    size_inches = svg_size_mm / 25.4
    
    print(f"Target size: {svg_size_mm}mm = {size_inches:.2f} inches")
    
    # Same logic but with better initial bounds
    if aspect_ratio >= 1:
        new_width = size_inches
        new_height = size_inches / aspect_ratio
    else:
        new_height = size_inches
        new_width = size_inches * aspect_ratio
    
    print(f"Calculated figure size: {new_width:.2f} x {new_height:.2f} inches")
    print(f"Final aspect ratio: {new_width/new_height:.3f}")
    
    plt.close(temp_fig)
    
    return aspect_ratio, new_width, new_height

def analyze_berlin_svg():
    """Analyze the actual Berlin SVG to see what happened"""
    
    print("\n" + "="*50)
    print("Analyzing Berlin SVG dimensions...")
    
    # From the SVG file we read earlier
    width_pt = 772.554331
    height_pt = 769.302491
    
    # Convert points to inches (1 inch = 72 points)
    width_inches = width_pt / 72.0
    height_inches = height_pt / 72.0
    
    print(f"SVG dimensions: {width_pt:.1f}pt x {height_pt:.1f}pt")
    print(f"In inches: {width_inches:.2f}\" x {height_inches:.2f}\"")
    print(f"Aspect ratio: {width_pt/height_pt:.4f}")
    
    return width_pt/height_pt

if __name__ == "__main__":
    print("SVG Aspect Ratio Analysis")
    print("="*50)
    
    # Test current logic
    current_ratio, current_width, current_height = test_aspect_ratio_calculation()
    
    # Test improved logic  
    improved_ratio, improved_width, improved_height = test_improved_aspect_ratio()
    
    # Analyze actual Berlin SVG
    berlin_ratio = analyze_berlin_svg()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Current logic aspect ratio: {current_ratio:.3f}")
    print(f"Improved logic aspect ratio: {improved_ratio:.3f}")
    print(f"Berlin SVG actual ratio: {berlin_ratio:.4f}")
    print(f"Classic poster ratio (e.g., A-series): {4/3:.3f} (4:3) or {3/2:.3f} (3:2)")