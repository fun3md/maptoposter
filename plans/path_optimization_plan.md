# Path Optimization Plan for SVG to G-code Conversion

## Current Implementation Analysis

The current implementation in `svg_to_gcode.py` uses a greedy nearest-neighbor algorithm:

1. It starts from the origin point (0, 0)
2. For each step, it finds the path with the closest starting point to the current position
3. It adds this path to the optimized list and removes it from the candidates
4. It updates the current position to the end point of the selected path
5. It repeats until all paths are processed

**Limitations of the current approach:**
- Can produce suboptimal solutions, sometimes significantly worse than the optimal solution
- No mechanism to escape local optima
- Path quality degrades as the number of paths increases

## Recommended Solution: Nearest Neighbor + 2-Opt

After evaluating multiple algorithms, the **Nearest Neighbor + 2-Opt hybrid approach** offers the best balance of implementation simplicity, performance, and solution quality.

### How 2-Opt Works

The 2-Opt algorithm improves an existing tour by:
1. Starting with an initial tour (from the nearest-neighbor algorithm)
2. Repeatedly finding pairs of edges that, when swapped, reduce the total path length
3. Continuing until no more improvements can be made

![2-Opt Swap Illustration](https://upload.wikimedia.org/wikipedia/commons/b/b7/2-opt_illustration.png)

### Implementation Structure

```python
def optimize_path_order(paths, attributes, min_power):
    """Optimize the order of paths to minimize travel distance
    
    Uses a nearest-neighbor algorithm followed by 2-opt improvement
    """
    # Step 1: Filter paths and prepare path data
    path_data = prepare_path_data(paths, attributes, min_power)
    
    # Step 2: Generate initial solution using nearest neighbor
    optimized_indices = nearest_neighbor(path_data)
    
    # Step 3: Improve solution using 2-opt
    optimized_indices = two_opt(path_data, optimized_indices)
    
    # Step 4: Convert indices back to path-attribute pairs
    optimized_paths = [
        (path_data[i]['path'], path_data[i]['attr']) 
        for i in optimized_indices
    ]
    
    return optimized_paths
```

## Performance Comparison

| Algorithm | Time Complexity | Solution Quality | Implementation Difficulty |
|-----------|-----------------|------------------|--------------------------|
| Nearest Neighbor (current) | O(n²) | Poor to Fair | Easy |
| Nearest Neighbor + 2-Opt | O(n²) + O(i·n²) | Good to Very Good | Moderate |
| Simulated Annealing | Varies | Very Good | Moderate |
| Christofides | O(n³) | Good (1.5x optimal) | Hard |
| Lin-Kernighan | O(n²·log(n)) | Excellent | Very Hard |
| Held-Karp | O(n²·2ⁿ) | Optimal | Moderate |

## Implementation Recommendations

1. **Implement the core 2-Opt algorithm** as outlined in the detailed implementation approach
2. **Add optimization level controls** to balance computation time vs. path quality:
   - `--optimize=fast`: Nearest-neighbor only (current implementation)
   - `--optimize=balanced`: Nearest-neighbor + limited 2-Opt (default)
   - `--optimize=thorough`: Nearest-neighbor + full 2-Opt
3. **Include progress reporting** for better user experience
4. **Add documentation** explaining the optimization benefits and options

## Expected Benefits

- **Small SVGs (<100 paths)**: Minimal additional computation time, potentially saving seconds to minutes in cutting time
- **Medium SVGs (100-1000 paths)**: Moderate additional computation time, potentially saving minutes to hours in cutting time
- **Large SVGs (>1000 paths)**: More significant computation time, but with options to control the optimization level

## Alternative Approaches

If the 2-Opt implementation doesn't provide sufficient improvement, consider:

1. **Simulated Annealing**: Better at escaping local optima, but requires parameter tuning
2. **Using existing TSP libraries** like Python-TSP or NetworkX
3. **Lin-Kernighan Heuristic**: More complex but produces near-optimal solutions

## Implementation Timeline

1. Implement the core 2-Opt algorithm
2. Add optimization level controls
3. Test with various SVG sizes and complexities
4. Add progress reporting and documentation
5. Consider visualization options to demonstrate the optimization benefits

## Conclusion

The Nearest Neighbor + 2-Opt hybrid approach offers a significant improvement over the current greedy algorithm while maintaining reasonable implementation complexity and computation time. The modular design allows for future enhancements if needed.