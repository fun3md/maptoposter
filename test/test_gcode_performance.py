#!/usr/bin/env python3
"""
Performance test script for optimized G-code generation
"""

import time
import os
import sys

def test_performance():
    """Test the performance of the optimized G-code generator"""
    
    # Check if we have a test SVG file
    test_svg = "test.svg"
    if not os.path.exists(test_svg):
        print(f"Creating test SVG file: {test_svg}")
        create_test_svg(test_svg)
    
    print("="*60)
    print("G-CODE GENERATION PERFORMANCE TEST")
    print("="*60)
    
    # Test different optimization levels
    test_cases = [
        ("Fast optimization", "fast"),
        ("Balanced optimization", "balanced"), 
        ("No optimization", "none")
    ]
    
    results = []
    
    for name, opt_level in test_cases:
        print(f"\nTesting {name}...")
        
        output_file = f"test_{opt_level}.nc"
        
        start_time = time.time()
        
        if opt_level == "none":
            # Run without optimization
            cmd = f'python svg_to_gcode.py "{test_svg}" --output "{output_file}" --no-optimize'
        else:
            # Run with optimization
            cmd = f'python svg_to_gcode.py "{test_svg}" --output "{output_file}" --optimize-level {opt_level}'
        
        print(f"Running: {cmd}")
        
        # Execute the command
        exit_code = os.system(cmd)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if exit_code == 0:
            print(f"[OK] {name} completed in {elapsed:.2f} seconds")
            results.append((name, elapsed, True))
            
            # Check output file size
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"  Output file size: {file_size:,} bytes")
        else:
            print(f"[FAIL] {name} failed")
            results.append((name, elapsed, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [(name, time) for name, time, success in results if success]
    
    if successful_results:
        fastest_time = min(time for _, time in successful_results)
        slowest_time = max(time for _, time in successful_results)
        
        for name, elapsed, success in results:
            if success:
                speedup = slowest_time / elapsed if elapsed > 0 else 1
                print(f"{name:25} {elapsed:8.2f}s (speedup: {speedup:.1f}x)")
            else:
                print(f"{name:25} FAILED")
        
        print(f"\nPerformance improvement: {slowest_time/fastest_time:.1f}x faster with optimization")
    else:
        print("No successful tests to compare")
    
    # Cleanup
    print(f"\nCleaning up test files...")
    for name, opt_level in test_cases:
        output_file = f"test_{opt_level}.nc"
        if os.path.exists(output_file):
            os.remove(output_file)
    
    if os.path.exists(test_svg) and test_svg != "test.svg":  # Don't remove original test.svg
        os.remove(test_svg)

def create_test_svg(filename):
    """Create a simple test SVG with various path types"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="400" height="400" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
    <rect width="400" height="400" fill="white"/>
    
    <!-- Simple lines -->
    <path d="M 50 50 L 350 50" stroke="#000000" stroke-width="2" fill="none"/>
    <path d="M 50 100 L 350 100" stroke="#333333" stroke-width="2" fill="none"/>
    <path d="M 50 150 L 350 150" stroke="#666666" stroke-width="2" fill="none"/>
    
    <!-- Curves -->
    <path d="M 50 200 Q 200 100 350 200" stroke="#000000" stroke-width="2" fill="none"/>
    <path d="M 50 250 Q 200 350 350 250" stroke="#333333" stroke-width="2" fill="none"/>
    
    <!-- Complex paths -->
    <path d="M 100 300 L 150 320 L 200 300 L 250 320 L 300 300" stroke="#000000" stroke-width="2" fill="none"/>
    
    <!-- Circles (using arcs) -->
    <path d="M 200 350 A 25 25 0 1 1 150 350 A 25 25 0 1 1 200 350" stroke="#666666" stroke-width="2" fill="none"/>
</svg>'''
    
    with open(filename, 'w') as f:
        f.write(svg_content)
    
    print(f"Created test SVG: {filename}")

if __name__ == "__main__":
    test_performance()