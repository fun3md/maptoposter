#!/usr/bin/env python3
"""Test CUDA with environment variables set"""

import os
import subprocess
import sys

# Set CUDA environment variables
os.environ['NUMBA_CUDA_DRIVER'] = 'C:\\Windows\\System32\\nvcuda.dll'
os.environ['CUDA_HOME'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0'
os.environ['CUDA_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0'

# Now test CUDA
print("Testing CUDA with environment variables...")
print(f"NUMBA_CUDA_DRIVER: {os.environ.get('NUMBA_CUDA_DRIVER')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
print()

from numba import cuda
print(f"CUDA is_available(): {cuda.is_available()}")

if cuda.is_available():
    try:
        device = cuda.get_current_device()
        print(f"Device: {device.name}")
    except Exception as e:
        print(f"Error getting device: {e}")
else:
    print("CUDA still not available")
    # Try to get more details
    try:
        # Check if driver can be loaded
        from numba.cuda.cudadrv import driver
        print(f"Driver loaded: {driver.is_initialized()}")
    except Exception as e:
        print(f"Driver error: {e}")