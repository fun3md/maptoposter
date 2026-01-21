#!/usr/bin/env python3
"""
Test the fixed SVG aspect ratio calculation
"""

import matplotlib.pyplot as plt
import numpy as np

def test_fixed_aspect_ratio():
    """Test the FIXED aspect ratio calculation logic"""
    
    print("Testing FIXED SVG aspect ratio calculation...")
    
    # Simulate the FIXED logic: use landscape temporary figure instead of square
    temp_fig, temp_ax = plt.subplots(figsize=(12, 8), facecolor='white')  # Landscape
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
    
    # Simulate the size calculation
    svg_size_mm = 300  # Example size
    size_inches = svg_size_mm / 25.4  # Convert to inches
    
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

def test_classic_poster_ratios():
    """Show what classic poster ratios look like"""
    
    print("\n" + "="*50)
    print("Classic Poster Ratios for Reference:")
    
    ratios = {
        "A-series (ISO)": 1.414,  # √2:1
        "Movie Poster": 1.85,     # 1.85:1
        "Photo Print 4x6": 1.5,   # 3:2
        "Photo Print 5x7": 1.4,   # 7:5
        "Classic 4x3": 1.333,     # 4:3
        "Modern 16x9": 1.778,     # 16:9
        "Square (current bug)": 1.0
    }
    
    svg_size_mm = 300
    size_inches = svg_size_mm / 25.4
    
    for name, ratio in ratios.items():
        if ratio >= 1:
            width = size_inches
            height = size_inches / ratio
        else:
            height = size_inches
            width = size_inches * ratio
            
        print(f"{name:20}: {ratio:.3f} → {width:.2f}\" × {height:.2f}\"")

if __name__ == "__main__":
    print("Fixed SVG Aspect Ratio Test")
    print("="*50)
    
    # Test the fixed logic
    fixed_ratio, fixed_width, fixed_height = test_fixed_aspect_ratio()
    
    # Show classic ratios for comparison
    test_classic_poster_ratios()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Fixed logic aspect ratio: {fixed_ratio:.3f}")
    print(f"This should now preserve the natural map proportions!")
    print(f"Expected result: Landscape posters for most cities")