#!/usr/bin/env python3
"""
Test script to verify the SVG aspect ratio fix.
This script tests the aspect ratio calculation logic.
"""

import osmnx as ox
import matplotlib.pyplot as plt

def test_aspect_ratio_calculation():
    """Test the aspect ratio calculation with a real city."""
    
    # Test with Berlin coordinates
    berlin_point = (52.5200, 13.4050)
    distance = 15000  # 15km radius
    
    print("Testing aspect ratio calculation...")
    print(f"City: Berlin")
    print(f"Coordinates: {berlin_point}")
    print(f"Distance: {distance}m")
    
    # Download graph
    print("\nDownloading street network...")
    G = ox.graph_from_point(berlin_point, dist=distance, dist_type='bbox', network_type='all')
    
    # Project to UTM
    print("Projecting to UTM...")
    G = ox.project_graph(G)
    
    # Calculate aspect ratio the same way as in the fixed code
    nodes = list(G.nodes())
    if nodes:
        lons = [G.nodes[node]['x'] for node in nodes]
        lats = [G.nodes[node]['y'] for node in nodes]
        
        data_width = max(lons) - min(lons)
        data_height = max(lats) - min(lats)
        aspect_ratio = data_width / data_height
        
        print(f"\nResults:")
        print(f"Geographic bounds: {data_width:.0f}m × {data_height:.0f}m")
        print(f"Natural aspect ratio: {aspect_ratio:.3f}")
        
        # Calculate poster dimensions for a 300mm longest edge
        size_mm = 300
        size_inches = size_mm / 25.4
        
        if aspect_ratio >= 1:
            new_width = size_inches
            new_height = size_inches / aspect_ratio
        else:
            new_height = size_inches
            new_width = size_inches * aspect_ratio
            
        print(f"Poster size (300mm longest edge): {new_width:.2f}\" × {new_height:.2f}\"")
        print(f"Expected SVG dimensions: {new_width*72:.0f}pt × {new_height*72:.0f}pt")
        
        # Check if this looks reasonable
        if 1.2 <= aspect_ratio <= 2.0:
            print("✅ Aspect ratio looks good for a city poster!")
        elif aspect_ratio < 1.2:
            print("⚠️  Aspect ratio is quite square - might be a compact city")
        else:
            print("⚠️  Aspect ratio is very wide - might be a sprawling city")
            
    else:
        print("❌ No nodes found in graph!")

if __name__ == "__main__":
    test_aspect_ratio_calculation()