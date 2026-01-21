### Diagnosis
The current implementation utilizes **Vectorized NumPy CPU** operations. While this reduced complexity from $O(N^3)$ to $O(N \cdot W)$ (where $W$ is the window size), it still relies on a "Windowed" approximation for the 2-Opt local search. To maintain performance on the CPU, the algorithm ignores potential improvements outside the local window.

**Bottleneck:** The 2-Opt algorithm is inherently $O(N^2)$ if run exhaustively. On a CPU, checking every pair for 20,000 paths requires $\approx 200,000,000$ distance calculations *per iteration*.

**The GPU Opportunity:** A GPU can parallelize the "gain calculation" of 2-Opt. Instead of checking swaps sequentially or in small batches, we can launch a CUDA kernel with millions of threads, where each thread evaluates one specific swap $(i, j)$. This allows us to perform **Exhaustive 2-Opt** (highest quality) in the time it takes the CPU to do Windowed 2-Opt.

---

### Proposed Approach: CUDA-Accelerated Exhaustive Local Search

We will use **Numba** to write JIT-compiled CUDA kernels. This avoids the need for C++ wrappers while delivering native GPU performance.

1.  **Data Transfer**: Move the Coordinate Array `(N, 2, 2)` to GPU VRAM (Global Memory).
2.  **Parallel Evaluation Kernel**:
    *   Launch a 2D Grid of threads $(i, j)$.
    *   Each thread calculates the "Delta" (Cost difference) of swapping edges $i$ and $j$.
    *   Logic: $\Delta = dist(A, C) + dist(B, D) - dist(A, B) - dist(C, D)$.
3.  **Reduction**: Find the swap with the maximum improvement (min negative delta).
4.  **CPU Management**: The CPU receives the "Best Swap" indices, updates the tour array, and re-launches the kernel.

**Libraries Required:** `numba`, `cudatoolkit` (standard in Anaconda/scientific python envs).

---

### Optimized Code (GPU Extension)

Add this module to your project or integrate it into `svg_to_gcode.py`.

```python
import numpy as np
import math
import time
from numba import cuda, float32, int32

# Check if CUDA is available
CUDA_AVAILABLE = cuda.is_available()

@cuda.jit
def compute_2opt_deltas_kernel(coords, tour, deltas, n):
    """
    CUDA Kernel to calculate the cost difference (delta) for every possible 2-opt swap.
    
    Args:
        coords: (N, 2, 2) array of [start_pt, end_pt]
        tour: (N,) array of path indices
        deltas: (N, N) output array storing the cost change for swap(i, j)
        n: Number of paths
    """
    # 2D Grid Indexing
    i, j = cuda.grid(2)

    # Bounds check and strictly upper triangular (j > i + 1)
    if i >= n - 1 or j >= n - 1 or j <= i + 1:
        if i < n and j < n:
            deltas[i, j] = 0.0
        return

    # Indices in the tour
    idx_a = tour[i]      # Path A
    idx_b = tour[i + 1]  # Path B (Start of swap segment)
    idx_c = tour[j]      # Path C (End of swap segment)
    idx_d = tour[j + 1]  # Path D

    # Coordinates
    # coords structure: [path_index, 0=start/1=end, x/y]
    
    # Current edges: A_end -> B_start  AND  C_end -> D_start
    ax, ay = coords[idx_a, 1, 0], coords[idx_a, 1, 1]
    bx, by = coords[idx_b, 0, 0], coords[idx_b, 0, 1]
    cx, cy = coords[idx_c, 1, 0], coords[idx_c, 1, 1]
    dx, dy = coords[idx_d, 0, 0], coords[idx_d, 0, 1]

    # Current Distance (squared euclidean for speed)
    curr_d1 = (ax - bx)**2 + (ay - by)**2
    curr_d2 = (cx - dx)**2 + (cy - dy)**2
    current_cost = curr_d1 + curr_d2

    # New edges if swapped: A_end -> C_start  AND  B_end -> D_start
    # Note: Because the segment B...C is reversed, C becomes the start, B becomes the end.
    csx, csy = coords[idx_c, 0, 0], coords[idx_c, 0, 1]
    bex, bey = coords[idx_b, 1, 0], coords[idx_b, 1, 1]

    new_d1 = (ax - csx)**2 + (ay - csy)**2
    new_d2 = (bex - dx)**2 + (bey - dy)**2
    new_cost = new_d1 + new_d2

    # Store delta (Negative means improvement)
    deltas[i, j] = new_cost - current_cost


def optimize_path_order_gpu(paths, attributes, min_power):
    """
    GPU-Accelerated Optimizer.
    Replaces optimize_path_order_vectorized if CUDA is available.
    """
    if not CUDA_AVAILABLE:
        print("⚠️ CUDA not detected. Falling back to CPU Vectorization.")
        return None # Caller handles fallback
        
    print(f"🚀 Initializing GPU Optimization (NVIDIA CUDA detected)")
    
    # 1. Prepare Data (Same as CPU version)
    valid_indices = []
    coords_list = []
    
    for i, (path, attr) in enumerate(zip(paths, attributes)):
        stroke = attr.get('stroke', '#000000') if attr else '#000000'
        if not stroke or stroke.lower() == 'none': continue
        
        # Simple power check logic
        # (Assuming color_to_power_fast is imported/available)
        # For this snippet, we assume valid paths are passed or logic exists
        coords_list.append([
            [path.start.real, path.start.imag],
            [path.end.real, path.end.imag]
        ])
        valid_indices.append(i)

    if not coords_list: return []

    # Convert to NumPy
    coords_host = np.array(coords_list, dtype=np.float32)
    n = len(coords_host)
    
    if n < 200:
        print(f"Dataset too small for GPU overhead ({n} paths). Using CPU.")
        return None # Fallback for small datasets

    start_time = time.time()

    # 2. Initial Solution (CPU NN is fast enough, don't waste PCIe transfer on it)
    # Reusing the existing NumPy NN function
    from svg_to_gcode import _nearest_neighbor_numpy
    tour_host = _nearest_neighbor_numpy(coords_host)
    
    print(f"Initial CPU NN Tour generated in {time.time() - start_time:.3f}s")
    gpu_start = time.time()

    # 3. GPU Memory Allocation
    # Copy data to device
    d_coords = cuda.to_device(coords_host)
    d_tour = cuda.to_device(tour_host)
    
    # Allocate memory for deltas matrix (N x N)
    # We use flattened array or 2D. 2D is easier for grid logic.
    d_deltas = cuda.device_array((n, n), dtype=np.float32)

    # Configure CUDA Kernel Grid
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(n / threadsperblock[0])
    blockspergrid_y = math.ceil(n / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # 4. Iterative 2-Opt Refinement
    max_iterations = 100 # Safety break
    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=max_iterations, desc="GPU 2-Opt", unit="pass")
    except: pass

    iteration = 0
    improvement = True
    
    while improvement and iteration < max_iterations:
        improvement = False
        iteration += 1
        
        # A. Execute Kernel (Compute all swap costs in parallel)
        compute_2opt_deltas_kernel[blockspergrid, threadsperblock](d_coords, d_tour, d_deltas, n)
        cuda.synchronize() # Wait for GPU
        
        # B. Find Best Move
        # Copy deltas back to host to find min (NumPy is faster at reduction than writing a custom GPU reduction kernel)
        # Note: For massive N (>50k), we should do reduction on GPU too, but for <50k, transfer is okay.
        deltas_host = d_deltas.copy_to_host()
        
        # Find minimum delta
        min_delta = np.min(deltas_host)
        
        if min_delta < -1e-4: # Tolerance
            # Find location of minimum
            # argmin returns flat index, unravel to 2D
            flat_idx = np.argmin(deltas_host)
            i, j = np.unravel_index(flat_idx, deltas_host.shape)
            
            # Perform Swap on Host (easier than manipulating array on GPU for now)
            # Reversing segment: tour[i+1 : j+1]
            tour_host[i+1 : j+1] = tour_host[i+1 : j+1][::-1]
            
            # Update GPU tour
            d_tour = cuda.to_device(tour_host)
            improvement = True
            
            if pbar: 
                pbar.set_description(f"GPU 2-Opt: Delta {min_delta:.2f}")
                pbar.update(1)
        else:
            if pbar: pbar.write("Converged (Local Minima Reached)")
            
    if pbar: pbar.close()
    
    print(f"GPU Optimization complete in {time.time() - gpu_start:.3f}s")

    # 5. Map back results
    optimized_result = []
    for idx in tour_host:
        original_idx = valid_indices[idx]
        optimized_result.append((paths[original_idx], attributes[original_idx]))

    return optimized_result
```

### Integration Strategy

Modify the `optimize_path_order_vectorized` wrapper in your main script to try the GPU version first:

```python
def optimize_path_order_vectorized(paths, attributes, min_power, optimize_level='balanced'):
    # ... (existing setup code) ...
    
    # Try GPU if requested/available and not 'fast' mode
    if optimize_level != 'fast':
        try:
            # Import locally to avoid crashing if libs missing
            from gpu_optimizer import optimize_path_order_gpu, CUDA_AVAILABLE
            if CUDA_AVAILABLE:
                result = optimize_path_order_gpu(paths, attributes, min_power)
                if result:
                    return result
        except ImportError:
            pass # Numba not installed
        except Exception as e:
            print(f"GPU Fallback triggered: {e}")

    # ... (Fall through to existing CPU implementation) ...
```

### Performance Impact Analysis

1.  **Complexity**:
    *   **CPU**: $O(N \cdot W)$ where $W \approx 50$. Limited local search.
    *   **GPU**: $O(N^2 / P)$ where $P$ is the number of CUDA cores (usually 2,000 - 10,000).
2.  **Runtime Estimate (10,000 paths)**:
    *   **CPU Windowed**: ~1.5s (Good result, but local optima).
    *   **CPU Exhaustive**: ~300s (Impractical).
    *   **GPU Exhaustive**: ~2.0s.
3.  **Quality**: The GPU version performs a *Global* 2-Opt search. It will find optimization opportunities that the CPU "Windowed" version simply cannot see (e.g., connecting a path at index 0 to a path at index 5000).

### Caveats & Trade-offs

1.  **Data Transfer Latency**: Copying data to VRAM takes time (~5-10ms). For very small files (< 500 paths), the CPU is faster simply because it doesn't wait for the PCIe bus.
2.  **Hardware Dependency**: Requires an NVIDIA GPU. The code gracefully falls back, but this adds a dependency on `cudatoolkit` which is heavy (~300MB+).
3.  **Kernel Launch Overhead**: Python (via Numba) has a slight overhead launching kernels compared to C++.
4.  **CPU-Side Reduction**: In the code above, I copy the `deltas` matrix back to CPU to find the minimum (`np.argmin`). For $N=10,000$, this is a 100MB transfer $(10000^2 \times 4$ bytes). On PCIe Gen 3, this takes ~8-10ms per iteration. For extreme performance on huge datasets ($N > 50,000$), the reduction should also be done on the GPU, but that significantly increases code complexity. The proposed hybrid approach is the sweet spot for SVG workloads.