#!/usr/bin/env python3
"""
Test script to verify nearest neighbor optimizations

This script creates test data and runs the optimized nearest neighbor
algorithms to verify they work correctly and demonstrate performance improvements.
"""

import sys
import time
import numpy as np
from svg_to_gcode import nearest_neighbor, nearest_neighbor_basic, nearest_neighbor_parallel, nearest_neighbor_optimized

def create_test_data(num_paths=1000):
    """Create test path data for benchmarking"""
    import random
    
    # Generate random start and end points
    np.random.seed(42)  # For reproducible results
    starts = np.random.rand(num_paths, 2) * 1000  # Random points in 1000x1000 area
    ends = np.random.rand(num_paths, 2) * 1000
    
    path_data = []
    for i in range(num_paths):
        path_data.append({
            'index': i,
            'path': None,  # Not needed for this test
            'attr': {},
            'start': (starts[i][0], starts[i][1]),
            'end': (ends[i][0], ends[i][1])
        })
    
    return path_data

def benchmark_algorithm(algorithm_func, path_data, name, **kwargs):
    """Benchmark a nearest neighbor algorithm"""
    print(f"\nTesting {name}...")
    start_time = time.time()
    
    try:
        result = algorithm_func(path_data, show_progress=True, **kwargs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        num_paths = len(path_data)
        paths_per_second = num_paths / elapsed if elapsed > 0 else 0
        
        print(f"[OK] {name} completed successfully")
        print(f"  - Processed {num_paths} paths")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - Speed: {paths_per_second:.0f} paths/second")
        print(f"  - Tour length: {len(result)} paths")
        
        return {
            'name': name,
            'time': elapsed,
            'paths_per_second': paths_per_second,
            'result_length': len(result),
            'success': True
        }
    except Exception as e:
        print(f"[FAIL] {name} failed: {e}")
        return {
            'name': name,
            'time': 0,
            'paths_per_second': 0,
            'result_length': 0,
            'success': False,
            'error': str(e)
        }

def main():
    """Run benchmarks for different path counts and algorithms"""
    print("Nearest Neighbor Optimization Test")
    print("=" * 50)
    
    # Test different path counts
    test_sizes = [100, 500, 1000, 2000, 5000]
    
    for num_paths in test_sizes:
        print(f"\n{'='*20} Testing with {num_paths} paths {'='*20}")
        
        # Create test data
        print(f"Creating test data with {num_paths} paths...")
        path_data = create_test_data(num_paths)
        
        # Determine which algorithms to test based on path count
        algorithms_to_test = []
        
        if num_paths <= 1000:
            algorithms_to_test.append((nearest_neighbor_basic, "Basic NN (<1000)", {}))
            if num_paths > 500:
                algorithms_to_test.append((nearest_neighbor_parallel, "Parallel NN (500-1000)", {'num_processes': 4}))
        else:
            algorithms_to_test.append((nearest_neighbor_optimized, f"Optimized NN (>10000, {num_paths} paths)", {'num_processes': 4}))
        
        # Test the unified nearest_neighbor function
        algorithms_to_test.append((nearest_neighbor, f"Unified NN ({num_paths} paths)", {'num_processes': 4}))
        
        # Run benchmarks
        results = []
        for algorithm, name, kwargs in algorithms_to_test:
            result = benchmark_algorithm(algorithm, path_data, name, **kwargs)
            results.append(result)
        
        # Print summary
        print(f"\nSummary for {num_paths} paths:")
        successful_results = [r for r in results if r['success']]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['paths_per_second'])
            print(f"  Best performance: {best_result['name']} ({best_result['paths_per_second']:.0f} paths/sec)")
            
            # Compare with basic if available
            basic_result = next((r for r in successful_results if 'Basic' in r['name']), None)
            if basic_result:
                improvement = (best_result['paths_per_second'] / basic_result['paths_per_second'] - 1) * 100
                print(f"  Improvement over basic: {improvement:.1f}%")
        else:
            print("  No successful results")
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print("\nKey optimizations implemented:")
    print("1. Spatial partitioning for large datasets (>5000 paths)")
    print("2. Parallel distance calculations for medium datasets (1000-5000 paths)")
    print("3. Memory optimization with garbage collection")
    print("4. Better progress reporting with performance metrics")
    print("5. Adaptive algorithm selection based on path count")

if __name__ == "__main__":
    main()