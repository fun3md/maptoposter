#!/usr/bin/env python3
"""Test CUDA with proper PATH"""

import os
import sys

# Add CUDA runtime bin directory to PATH
cuda_runtime_bin = r'F:\projects\python3119\Lib\site-packages\nvidia\cuda_runtime\bin'
current_path = os.environ.get('PATH', '')
if cuda_runtime_bin not in current_path:
    os.environ['PATH'] = cuda_runtime_bin + ';' + current_path
    print(f"Added to PATH: {cuda_runtime_bin}")

print(f"PATH contains CUDA runtime: {'cuda_runtime' in os.environ.get('PATH', '')}")
print()

# Now test CUDA
from numba import cuda
print(f"CUDA is_available(): {cuda.is_available()}")

if cuda.is_available():
    try:
        device = cuda.get_current_device()
        print(f"Device: {device.name}")
        print("\n✓ CUDA is working!")
    except Exception as e:
        print(f"Error getting device: {e}")
else:
    print("CUDA still not available")
    # Try to get more info
    try:
        info = cuda.get_current_device()
        print(f"Device info: {info}")
    except Exception as e:
        print(f"Error: {e}")