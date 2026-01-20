#!/usr/bin/env python3
"""
Test script for the hex_to_rgb and color_to_power functions
to verify the fixes for handling empty strings and invalid inputs.
"""

from svg_to_gcode import hex_to_rgb, color_to_power

def test_hex_to_rgb():
    """Test the hex_to_rgb function with various inputs"""
    test_cases = [
        ('#000000', (0, 0, 0)),      # Black
        ('#FFFFFF', (255, 255, 255)), # White
        ('#FF0000', (255, 0, 0)),    # Red
        ('#00FF00', (0, 255, 0)),    # Green
        ('#0000FF', (0, 0, 255)),    # Blue
        ('000000', (0, 0, 0)),       # No hash
        ('', (0, 0, 0)),             # Empty string
        (None, (0, 0, 0)),           # None
        ('invalid', (0, 0, 0)),      # Invalid hex
        ('FF', (0, 0, 0)),           # Too short
    ]
    
    for input_value, expected_output in test_cases:
        result = hex_to_rgb(input_value)
        print(f"hex_to_rgb('{input_value}') = {result} (Expected: {expected_output})")
        assert result == expected_output, f"Failed: {input_value} -> {result} != {expected_output}"

def test_color_to_power():
    """Test the color_to_power function with various inputs"""
    min_power = 100
    max_power = 800
    test_cases = [
        ('#000000', 800),  # Black -> max power
        ('#FFFFFF', 100),  # White -> min power
        ('#808080', 450),  # Gray -> medium power
        ('', 450),         # Empty string -> medium power
        (None, 450),       # None -> medium power
        ('invalid', 450),  # Invalid -> medium power
    ]
    
    for input_value, expected_output in test_cases:
        result = color_to_power(input_value, min_power, max_power)
        print(f"color_to_power('{input_value}', {min_power}, {max_power}) = {result} (Expected: {expected_output})")
        # Allow small differences due to rounding
        assert abs(result - expected_output) <= 1, f"Failed: {input_value} -> {result} != {expected_output}"

if __name__ == "__main__":
    print("Testing hex_to_rgb function...")
    test_hex_to_rgb()
    print("\nTesting color_to_power function...")
    test_color_to_power()
    print("\nAll tests passed!")