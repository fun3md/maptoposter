#!/usr/bin/env python3
"""Test CUDA with proper PATH"""

import os
import sys

# Add CUDA bin directory to PATH
cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin'
current_path = os.environ.get('PATH', '')
if cuda_bin not in current_path:
    os.environ['PATH'] = cuda_bin + ';' + current_path
    print(f"Added to PATH: {cuda_bin}")

# Also set CUDA_HOME
os.environ['CUDA_HOME'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'

print(f"PATH contains CUDA: {'CUDA' in os.environ.get('PATH', '')}")
print()

# Now test CUDA
from numba import cuda
print(f"CUDA is_available(): {cuda.is_available()}")

if cuda.is_available():
    try:
        device = cuda.get_current_device()
        print(f"Device: {device.name}")
        print("CUDA is working!")
    except Exception as e:
        print(f"Error getting device: {e}")
else:
    print("CUDA still not available")
    # Check if cudart.dll can be found
    import ctypes
    try:
        ctypes.WinDLL('cudart.dll')
        print("cudart.dll found!")
    except OSError as e:
        print(f"cudart.dll not found: {e}")