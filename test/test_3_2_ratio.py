#!/usr/bin/env python3
"""
Test script to verify that SVG output uses 3:2 aspect ratio
"""

import subprocess
import sys
import os

def test_3_2_ratio():
    """Test that SVG output uses 3:2 aspect ratio"""
    
    # Run the create_map_poster script with SVG output
    cmd = [
        sys.executable, "create_map_poster.py",
        "--city", "Berlin",
        "--country", "Germany",
        "--format", "svg",
        "--svg-size", "200",  # 200mm longest edge
        "--distance", "5000"  # Small distance for quick test
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ Command executed successfully")
            
            # Check the output for our debug messages
            if "[DEBUG] Forced 3:2 aspect ratio" in result.stdout:
                print("✓ 3:2 aspect ratio enforced")
            else:
                print("✗ 3:2 aspect ratio not enforced")
                
            if "aspect ratio: 1.500" in result.stdout:
                print("✓ Correct aspect ratio value (1.5 = 3:2)")
            else:
                print("✗ Aspect ratio value incorrect")
                
            # Find the generated SVG file
            posters_dir = "posters"
            if os.path.exists(posters_dir):
                svg_files = [f for f in os.listdir(posters_dir) if f.endswith('.svg')]
                if svg_files:
                    latest_svg = max(svg_files, key=lambda x: os.path.getctime(os.path.join(posters_dir, x)))
                    svg_path = os.path.join(posters_dir, latest_svg)
                    print(f"✓ Generated SVG: {svg_path}")
                    
                    # Check SVG content for viewbox
                    with open(svg_path, 'r') as f:
                        svg_content = f.read()
                        if 'viewBox=' in svg_content:
                            print("✓ SVG has viewBox attribute")
                        else:
                            print("✗ SVG missing viewBox attribute")
                else:
                    print("✗ No SVG files found in posters directory")
            else:
                print("✗ Posters directory not found")
                
        else:
            print(f"✗ Command failed with return code {result.returncode}")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("✗ Command timed out")
    except Exception as e:
        print(f"✗ Error running command: {e}")

if __name__ == "__main__":
    test_3_2_ratio()