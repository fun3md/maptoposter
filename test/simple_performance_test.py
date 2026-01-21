#!/usr/bin/env python3
"""
Simple Performance Test for SVG to G-code Optimization
"""

import time
import os
import gc
import psutil
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def create_test_svg(filename, num_paths=100):
    """Create a test SVG with specified number of paths"""
    svg_content = ['''<?xml version="1.0" encoding="UTF-8"?>
<svg width="500" height="500" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">''']
    
    colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC']
    
    for i in range(num_paths):
        color = colors[i % len(colors)]
        x = (i % 20) * 25 + 10
        y = (i // 20) * 25 + 10
        
        if i % 3 == 0:
            svg_content.append(f'  <path d="M {x} {y} L {x+20} {y+20}" stroke="{color}" stroke-width="1" fill="none"/>')
        elif i % 3 == 1:
            svg_content.append(f'  <path d="M {x} {y} Q {x+10} {y-10} {x+20} {y}" stroke="{color}" stroke-width="1" fill="none"/>')
        else:
            svg_content.append(f'  <path d="M {x} {y} L {x+10} {y+10} L {x+20} {y}" stroke="{color}" stroke-width="1" fill="none"/>')
    
    svg_content.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg_content))

def run_performance_test():
    """Run performance test with different optimization levels"""
    
    print("="*70)
    print("SVG TO G-CODE PERFORMANCE TEST")
    print("="*70)
    
    # Test with different path counts
    test_sizes = [50, 200, 500]
    
    for num_paths in test_sizes:
        print(f"\nTesting with {num_paths} paths:")
        print("-" * 40)
        
        # Create test SVG
        svg_file = f"perf_test_{num_paths}.svg"
        create_test_svg(svg_file, num_paths)
        
        # Test each optimization level
        results = {}
        
        for opt_level in ['none', 'fast', 'balanced']:
            output_file = f"perf_test_{num_paths}_{opt_level}.nc"
            
            # Measure performance
            gc.collect()
            initial_memory = get_memory_usage()
            
            start_time = time.time()
            
            # Run conversion
            if opt_level == 'none':
                cmd = f'python svg_to_gcode.py "{svg_file}" --output "{output_file}" --no-optimize'
            else:
                cmd = f'python svg_to_gcode.py "{svg_file}" --output "{output_file}" --optimize-level {opt_level}'
            
            exit_code = os.system(cmd)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            gc.collect()
            final_memory = get_memory_usage()
            memory_used = final_memory - initial_memory
            
            # Get file size
            file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
            
            if exit_code == 0:
                results[opt_level] = {
                    'time': elapsed,
                    'memory': memory_used,
                    'size': file_size
                }
                print(f"  {opt_level:10}: {elapsed:.3f}s, {memory_used:.1f}MB, {file_size:,} bytes")
            else:
                print(f"  {opt_level:10}: FAILED")
                results[opt_level] = None
        
        # Calculate speedup
        if results.get('none') and results.get('fast'):
            speedup = results['none']['time'] / results['fast']['time']
            print(f"  Speedup (fast vs none): {speedup:.2f}x")
        
        # Cleanup
        for opt_level in ['none', 'fast', 'balanced']:
            output_file = f"perf_test_{num_paths}_{opt_level}.nc"
            if os.path.exists(output_file):
                os.remove(output_file)
        
        if os.path.exists(svg_file):
            os.remove(svg_file)
    
    print(f"\n{'='*70}")
    print("PERFORMANCE TEST COMPLETED")
    print("="*70)

if __name__ == "__main__":
    run_performance_test()