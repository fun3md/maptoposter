#!/usr/bin/env python3

def color_to_power_fast(hex_color, min_power):
    """Simplified color parser without heavy error handling overhead for loop"""
    if not isinstance(hex_color, str) or not hex_color.startswith('#'):
        return 128 # Default
    try:
        # Manual hex parse is faster than int(x, 16) calls in a loop
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return int(min_power + (1 - gray/255.0) * (1000 - min_power))
    except:
        return min_power

# Test the function
colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC']
min_power = 0

print('Testing color_to_power_fast function:')
for color in colors:
    power = color_to_power_fast(color, min_power)
    print(f'{color} -> power: {power} (>{min_power}: {power > min_power})')