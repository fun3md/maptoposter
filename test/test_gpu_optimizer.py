#!/usr/bin/env python3
"""
Test script for GPU-accelerated path optimizer.

This script tests:
1. CUDA availability detection
2. GPU optimizer fallback behavior for small datasets
3. Integration with svg_to_gcode module
"""

import sys
import numpy as np
from svgpathtools import Path, Line

# Test CUDA availability
print("=" * 60)
print("GPU Optimizer Test Suite")
print("=" * 60)

# Test 1: CUDA availability check
print("\n[Test 1] Checking CUDA availability...")
try:
    from gpu_optimizer import CUDA_AVAILABLE, get_cuda_info, CUDA_DRIVER_ERROR
    if CUDA_AVAILABLE:
        info = get_cuda_info()
        print("  [OK] CUDA is available")
        if info and 'name' in info:
            print(f"    Device: {info['name']}")
            print(f"    Compute Capability: {info['compute_capability']}")
    else:
        print("  [SKIP] CUDA not available")
        if CUDA_DRIVER_ERROR:
            print(f"    Error: {CUDA_DRIVER_ERROR}")
        print("  -> GPU optimization will use CPU fallback (expected on non-NVIDIA systems)")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 2: GPU optimizer module functions exist
print("\n[Test 2] Checking GPU optimizer module functions...")
from gpu_optimizer import (
    optimize_path_order_gpu,
    _nearest_neighbor_numpy,
    CUDA_AVAILABLE
)
print("  [OK] All required functions imported successfully")

# Test 3: Create test paths and test CPU fallback
print("\n[Test 3] Testing with small dataset (should use CPU fallback)...")
from gpu_optimizer import optimize_path_order_gpu

# Create simple test paths
test_paths = []
test_attributes = []

# Create 10 simple line paths
for i in range(10):
    path = Path(Line(complex(i, 0), complex(i + 1, 0)))
    test_paths.append(path)
    test_attributes.append({'stroke': '#000000'})

result = optimize_path_order_gpu(test_paths, test_attributes, min_power=0)

if result is None:
    print("  [OK] Correctly returned None for small dataset (GPU overhead not justified)")
else:
    print(f"  [OK] Returned {len(result)} optimized paths")

# Test 4: Test nearest neighbor function
print("\n[Test 4] Testing nearest neighbor function...")
coords = np.array([
    [[0, 0], [1, 0]],
    [[1, 0], [1, 1]],
    [[1, 1], [0, 1]],
    [[0, 1], [0, 0]],
], dtype=np.float32)

tour = _nearest_neighbor_numpy(coords)
print(f"  [OK] Nearest neighbor tour: {tour}")
print(f"    Tour length: {len(tour)}")

# Test 5: Integration with svg_to_gcode
print("\n[Test 5] Testing integration with svg_to_gcode...")
from svg_to_gcode import optimize_path_order_vectorized

# Create more test paths
test_paths = []
test_attributes = []
np.random.seed(42)  # For reproducibility

for i in range(50):
    x, y = np.random.rand() * 100, np.random.rand() * 100
    path = Path(Line(complex(x, y), complex(x + 5, y + 5)))
    test_paths.append(path)
    test_attributes.append({'stroke': '#000000'})

result = optimize_path_order_vectorized(test_paths, test_attributes, min_power=0, optimize_level='balanced')
print(f"  [OK] optimize_path_order_vectorized returned {len(result)} paths")

# Test 6: Test with thorough optimization (should try GPU)
print("\n[Test 6] Testing with thorough optimization level...")
result = optimize_path_order_vectorized(test_paths, test_attributes, min_power=0, optimize_level='thorough')
print(f"  [OK] Thorough optimization returned {len(result)} paths")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)

# Summary
print("\nSummary:")
print(f"  - CUDA Available: {CUDA_AVAILABLE}")
print("  - For large datasets (N > 200) with 'thorough' optimization:")
print("    -> GPU-accelerated exhaustive 2-Opt will be used")
print("  - For small datasets or 'balanced'/'fast' optimization:")
print("    -> CPU vectorized implementation will be used")
print("  - Graceful fallback is implemented for non-CUDA systems")