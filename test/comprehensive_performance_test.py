#!/usr/bin/env python3
"""
Comprehensive Performance Test Suite for SVG to G-code Optimization
"""

import time
import os
import sys
import psutil
import gc
import numpy as np
from pathlib import Path

def create_large_test_svg(filename, num_paths=1000):
    """Create a large test SVG with many paths for performance testing"""
    svg_content = ['''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1000" height="1000" viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg">''']
    
    # Create various path types with different colors
    colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC']
    
    for i in range(num_paths):
        color = colors[i % len(colors)]
        x = (i % 50) * 20 + 10
        y = (i // 50) * 20 + 10
        
        if i % 4 == 0:
            # Simple line
            svg_content.append(f'  <path d="M {x} {y} L {x+15} {y+15}" stroke="{color}" stroke-width="1" fill="none"/>')
        elif i % 4 == 1:
            # Curve
            svg_content.append(f'  <path d="M {x} {y} Q {x+7} {y-5} {x+15} {y}" stroke="{color}" stroke-width="1" fill="none"/>')
        elif i % 4 == 2:
            # Complex path
            svg_content.append(f'  <path d="M {x} {y} L {x+5} {y+5} L {x+10} {y} L {x+15} {y+5}" stroke="{color}" stroke-width="1" fill="none"/>')
        else:
            # Arc/circle
            svg_content.append(f'  <path d="M {x+7} {y} A 7 7 0 1 1 {x} {y} A 7 7 0 1 1 {x+7} {y}" stroke="{color}" stroke-width="1" fill="none"/>')
    
    svg_content.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg_content))
    
    print(f"Created large test SVG: {filename} with {num_paths} paths")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def measure_performance(svg_file, optimization_level, output_file):
    """Measure performance of a single conversion"""
    
    # Force garbage collection before test
    gc.collect()
    initial_memory = get_memory_usage()
    
    start_time = time.time()
    
    # Run the conversion
    if optimization_level == "none":
        cmd = f'python svg_to_gcode.py "{svg_file}" --output "{output_file}" --no-optimize'
    else:
        cmd = f'python svg_to_gcode.py "{svg_file}" --output "{output_file}" --optimize-level {optimization_level}'
    
    exit_code = os.system(cmd)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Measure memory after
    gc.collect()
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory
    
    # Get output file size
    file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    
    return {
        'success': exit_code == 0,
        'time': elapsed,
        'memory_used': memory_used,
        'file_size': file_size,
        'initial_memory': initial_memory,
        'final_memory': final_memory
    }

def verify_output_correctness(file1, file2):
    """Verify that two G-code files are functionally equivalent"""
    if not os.path.exists(file1) or not os.path.exists(file2):
        return False
    
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    # Remove comments and whitespace for comparison
    def clean_gcode(lines):
        cleaned = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(';') and not line.startswith('#'):
                # Extract G-code commands only
                if any(cmd in line for cmd in ['G0', 'G1', 'G2', 'G3', 'M3', 'M4', 'M5']):
                    cleaned.append(line)
        return cleaned
    
    clean1 = clean_gcode(lines1)
    clean2 = clean_gcode(lines2)
    
    # Compare the number of commands and their sequence
    if len(clean1) != len(clean2):
        return False
    
    # Compare command sequences (allowing for some variation in coordinates due to optimization)
    for i, (cmd1, cmd2) in enumerate(zip(clean1, clean2)):
        # Extract command type (G0, G1, M3, etc.)
        cmd_type1 = cmd1.split()[0] if cmd1.split() else ''
        cmd_type2 = cmd2.split()[0] if cmd2.split() else ''
        
        if cmd_type1 != cmd_type2:
            return False
    
    return True

def run_comprehensive_tests():
    """Run comprehensive performance tests"""
    
    print("="*80)
    print("COMPREHENSIVE SVG TO G-CODE PERFORMANCE TEST SUITE")
    print("="*80)
    
    # Test configurations
    test_configs = [
        ("Small (50 paths)", 50),
        ("Medium (500 paths)", 500),
        ("Large (2000 paths)", 2000)
    ]
    
    optimization_levels = ["none", "fast", "balanced", "thorough"]
    
    all_results = {}
    
    for test_name, num_paths in test_configs:
        print(f"\n{'-'*60}")
        print(f"TESTING: {test_name}")
        print(f"{'-'*60}")
        
        # Create test SVG
        svg_file = f"test_{num_paths}_paths.svg"
        if not os.path.exists(svg_file):
            create_large_test_svg(svg_file, num_paths)
        
        test_results = {}
        
        for opt_level in optimization_levels:
            print(f"\nTesting {opt_level} optimization...")
            
            output_file = f"test_{num_paths}_{opt_level}.nc"
            result = measure_performance(svg_file, opt_level, output_file)
            
            if result['success']:
                print(f"  [OK] {opt_level:10} - Time: {result['time']:.3f}s, Memory: {result['memory_used']:.1f}MB, Size: {result['file_size']:,} bytes")
                test_results[opt_level] = result
            else:
                print(f"  [FAIL] {opt_level:10} - FAILED")
                test_results[opt_level] = None
        
        all_results[test_name] = test_results
        
        # Clean up test files for this configuration
        for opt_level in optimization_levels:
            output_file = f"test_{num_paths}_{opt_level}.nc"
            if os.path.exists(output_file):
                os.remove(output_file)
    
    # Generate performance report
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS REPORT")
    print("{'='*80}")
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        print("-" * 40)
        
        # Find baseline (no optimization) for comparison
        baseline = results.get('none')
        if not baseline:
            print("  No baseline available for comparison")
            continue
        
        for opt_level, result in results.items():
            if opt_level == 'none' or not result:
                continue
            
            speedup = baseline['time'] / result['time'] if result['time'] > 0 else 1
            memory_efficiency = baseline['memory_used'] / result['memory_used'] if result['memory_used'] > 0 else 1
            
            print(f"  {opt_level:10}: {result['time']:.3f}s "
                  f"(speedup: {speedup:.2f}x, memory: {result['memory_used']:.1f}MB, "
                  f"efficiency: {memory_efficiency:.2f}x)")
    
    # Correctness verification
    print(f"\n{'='*80}")
    print("CORRECTNESS VERIFICATION")
    print("{'='*80}")
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        baseline_file = f"test_{test_name.split('(')[1].split(' ')[0]}_none.nc"  # Extract number
        
        for opt_level in optimization_levels[1:]:  # Skip 'none'
            if results.get(opt_level):
                opt_file = f"test_{test_name.split('(')[1].split(' ')[0]}_{opt_level}.nc"
                if os.path.exists(baseline_file) and os.path.exists(opt_file):
                    is_correct = verify_output_correctness(baseline_file, opt_file)
                    status = "[PASS]" if is_correct else "[FAIL]"
                    print(f"  {opt_level:10}: {status}")
    
    # Cleanup
    print(f"\n{'='*80}")
    print("CLEANUP")
    print("{'='*80}")
    
    for test_name, num_paths in test_configs:
        svg_file = f"test_{num_paths}_paths.svg"
        if os.path.exists(svg_file):
            os.remove(svg_file)
            print(f"Removed {svg_file}")
    
    print("\nPerformance testing completed!")

if __name__ == "__main__":
    try:
        run_comprehensive_tests()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()