#!/usr/bin/env python3
"""CUDA diagnostics script"""

import sys

print("Python version:", sys.version)
print()

# Check numba
try:
    import numba
    print(f"numba version: {numba.__version__}")
except ImportError as e:
    print(f"numba not installed: {e}")
    sys.exit(1)

# Check CUDA availability
from numba import cuda
print(f"CUDA is_available(): {cuda.is_available()}")

# Try to get more details
try:
    print("\nTrying to list devices...")
    devices = cuda.list_devices()
    print(f"Number of CUDA devices: {len(devices)}")
    for i, dev in enumerate(devices):
        print(f"  Device {i}: {dev}")
except Exception as e:
    print(f"Error listing devices: {e}")

# Check if driver is accessible
try:
    print("\nTrying to get current device...")
    device = cuda.get_current_device()
    print(f"Current device: {device.name}")
except Exception as e:
    print(f"Error getting current device: {e}")

# Check environment
print("\nEnvironment variables related to CUDA:")
for key, value in sorted(os.environ.items()):
    if 'CUDA' in key.upper():
        print(f"  {key}: {value}")

# Check if nvidia-smi is available
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("\nnvidia-smi output:")
        print(result.stdout[:500])
    else:
        print(f"\nnvidia-smi error: {result.stderr}")
except FileNotFoundError:
    print("\nnvidia-smi not found")
except Exception as e:
    print(f"\nnvidia-smi error: {e}")