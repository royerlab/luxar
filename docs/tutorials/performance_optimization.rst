Tutorial 4: Performance Optimization
======================================

Learn how to handle billion-point datasets efficiently and achieve smooth 60 FPS interaction.

Performance Fundamentals
-------------------------

**Luxar's Performance Model**:

Performance = f(Spatial Locality, Cache Hit Rate, Chunk Granularity, GPU Throughput)

Understanding the Bottlenecks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Network I/O**: Fetching chunks from HTTP (~100ms each)

   * **Solution**: Intelligent prefetching, aggressive caching

2. **Decompression**: Blosc decompression (~5ms per chunk)

   * **Solution**: Parallel decompression, L1 cache for decoded data

3. **GPU Rendering**: Drawing millions of points (~60ms for 10M points)

   * **Solution**: Spatial culling, LOD, instancing

4. **Memory Allocation**: Creating TypedArrays (~10ms for large buffers)

   * **Solution**: Object pooling, reuse buffers

**The Critical Path**:

.. code-block:: text

   User navigates → Query chunks (1ms)
                  → L1 check (1μs)
                  → L2 check (1ms) ← Usually hits here (80-95%)
                  → HTTP fetch (100ms) ← Only first time
                  → Decode (5ms)
                  → GPU upload (2ms)
                  → Render (10ms)

   Target: <16ms total (60 FPS)
   Achieved: 10-20ms with cache hits, 120ms on cache miss

Optimization 1: Chunk Size Selection
-------------------------------------

**The Tradeoff**:

* **Small chunks** (e.g., 256 points):

  * Pros: Granular loading, less waste
  * Cons: Many HTTP requests, overhead per chunk

* **Large chunks** (e.g., 8192 points):

  * Pros: Fewer requests, less overhead
  * Cons: Coarse granularity, load unnecessary points

**Recommended Sizes**:

.. code-block:: python

   # For static 3D data (no time dimension)
   compiler = LuxarZarrCompiler(
       'scene.zarr',
       chunk_size=2048  # Good balance: 2K points per chunk
   )

   # For time-series (4D with time animation)
   compiler = LuxarZarrCompiler(
       'timeseries.zarr',
       chunk_size=512  # Smaller chunks for finer time granularity
   )

   # For massive static datasets (>100M points)
   compiler = LuxarZarrCompiler(
       'huge.zarr',
       chunk_size=4096  # Larger chunks reduce metadata overhead
   )

**Calculation**:

Target chunk size in bytes: 32KB - 1MB

.. code-block:: python

   # 3D positions (3 × float32 = 12 bytes per point)
   # + RGB colors (3 × uint8 = 3 bytes per point)
   # = 15 bytes per point

   # Target 64KB chunks:
   chunk_size = 64 * 1024 / 15 ≈ 4369 points

   # Round to power of 2: 4096 points

Optimization 2: Spatial Ordering Strategy
------------------------------------------

**Morton vs Hilbert**:

**Morton (Z-Order)**:

* **Pros**: Faster to compute, simpler bit-interleaving
* **Cons**: Occasional "jumps" at quadrant boundaries
* **Best for**: General-purpose, 3D data

**Hilbert**:

* **Pros**: Better locality (no jumps), slightly better compression
* **Cons**: More complex algorithm, slightly slower
* **Best for**: Maximum compression, smooth navigation

.. code-block:: python

   # For interactive visualization (speed priority)
   compiler = LuxarZarrCompiler('scene.zarr', ordering_method="morton")

   # For archival storage (compression priority)
   compiler = LuxarZarrCompiler('archive.zarr', ordering_method="hilbert")

**Measured Impact**:

* Compression: Hilbert ~10% better than Morton
* Write speed: Morton ~20% faster than Hilbert
* Query speed: Equivalent

Optimization 3: Encoding Mode Selection
----------------------------------------

**AUTO Mode** (Recommended):

Analyzes data and chooses optimal encoding:

.. code-block:: python

   from luxar.encoding import EncodingMode

   compiler = LuxarZarrCompiler('scene.zarr', encoding_mode=EncodingMode.AUTO)

   # Positions: Analyzes range, uses uint16 if |max-min| < 65536
   # Colors: uint8 for SDR, float32 for HDR
   # Radii: Broadcasts if uniform, uint16 if varying

**PRECISION Mode** (Lossless):

Keeps float32 for everything:

.. code-block:: python

   compiler = LuxarZarrCompiler('scene.zarr', encoding_mode=EncodingMode.PRECISION)

   # Positions: float32 (no quantization)
   # Colors: float32 (full HDR range)
   # Radii: float32

   # Use when: Precision critical, storage not constrained

**MEMORY Mode** (Aggressive):

Maximum compression:

.. code-block:: python

   compiler = LuxarZarrCompiler('scene.zarr', encoding_mode=EncodingMode.MEMORY)

   # Positions: uint16 or uint8 (aggressive quantization)
   # Colors: uint8 (SDR only)
   # Radii: uint8 or broadcast

   # Use when: Network-limited, compression critical

**Compression Ratios**:

.. code-block:: text

   1M points, 3D RGB:
   - PRECISION: 24MB (no quantization) + 12MB blosc = 36MB
   - AUTO: 12MB (uint16+uint8) + 3MB blosc = 15MB (2.4× smaller)
   - MEMORY: 6MB (uint8) + 1.5MB blosc = 7.5MB (4.8× smaller)

Optimization 4: Prefetching Configuration
------------------------------------------

**Concept**: Predict which chunks user will load next, fetch in background

**Strategy**: Load adjacent chunks (±1 in each dimension)

.. code-block:: typescript

   // In viewer, cache automatically prefetches:
   // Current chunk: [5, 3, 2]
   // Prefetches: [4,3,2], [6,3,2], [5,2,2], [5,4,2], [5,3,1], [5,3,3]

**Configuration** (viewer URL parameters):

.. code-block:: text

   ?src=data.zarr&prefetch=aggressive  # Prefetch all adjacent
   ?src=data.zarr&prefetch=conservative  # Prefetch only likely direction
   ?src=data.zarr&prefetch=off  # No prefetching (debugging)

**Impact**:

* Aggressive: 90-95% cache hit rate, 3× bandwidth usage
* Conservative: 80-85% cache hit rate, 1.5× bandwidth
* Off: 20-30% cache hit rate (only repeat views cached)

Optimization 5: Viewer Configuration
-------------------------------------

**Point Rendering Budget**:

Limit rendered points for consistent frame rate:

.. code-block:: text

   URL: ?src=data.zarr&maxPoints=1000000

   # Viewer will load up to 1M points
   # If more are visible, shows furthest points first
   # Ensures 60 FPS even with billions of points

**LOD (Level of Detail)**:

Reduce point detail at distance:

.. code-block:: text

   # Points far from camera rendered smaller
   # Automatically handled by world-space sizing
   # Can configure minimum pixel size:

   ?src=data.zarr&minPixelSize=0.5

**Frustum Culling**:

Only render points in view:

.. code-block:: text

   # Automatically enabled
   # Points outside camera view not rendered
   # Typically excludes 90%+ of points

Billion-Point Dataset Strategy
-------------------------------

**Scenario**: 1 billion points, 5D (x, y, z, time=100, channel=4)

**Naive Approach**:

.. code-block:: python

   # DON'T DO THIS:
   positions = np.random.randn(1_000_000_000, 5).astype(np.float32)
   # Requires 20GB RAM! Will crash on most machines

**Luxar Approach** (Progressive):

.. code-block:: python

   with LuxarZarrCompiler(
       'billion_points.zarr',
       chunk_size=2048,           # 2K points per chunk = 488K chunks
       encoding_mode=EncodingMode.MEMORY,  # Aggressive compression
       ordering_method="hilbert",  # Best compression
   ) as compiler:

       scene = compiler.create_scene(dimensions=dims_5d)

       # Process in batches (only 10M in memory at once)
       batch_size = 10_000_000

       for batch_idx in range(100):  # 100 batches × 10M = 1B points
           # Generate or load this batch only
           positions_batch = generate_batch(batch_idx, batch_size)

           # Write immediately (no accumulation!)
           scene.add_points(
               f"batch_{batch_idx:03d}",
               positions_batch,
               colors=compute_colors(batch_idx),
               radii=1.0  # Scalar broadcasts
           )

           # Memory usage: ~400MB per batch (10M points × 40 bytes)
           # Total write time: ~1000 seconds (1M points/sec)

   # Result: 1B points written using only 400MB RAM!

**File Size**:

.. code-block:: text

   Uncompressed: 1B × 5 × 4 bytes = 20GB
   With MEMORY encoding: 1B × 5 × 1 byte = 5GB (quantized)
   With blosc compression: 5GB → 1-2GB (3-5× compression)
   Final: ~1.5GB for 1 billion 5D points (13× compression)

**Viewing Performance**:

.. code-block:: text

   Typical view: 100K points visible (0.01% of total)
   Chunks loaded: ~50 chunks (0.01% of 488K total chunks)
   Load time: First view ~2s, navigating ~50ms (prefetch)
   Render: 60 FPS with 100K points

Monitoring Performance
----------------------

**Built-in Metrics**:

The viewer provides real-time performance monitoring:

.. code-block:: text

   Press ` (backtick) to toggle performance overlay:

   FPS: 60
   Points: 145,234 / 1,000,000,000 (0.01%)
   Chunks: 71 loaded, 12 visible
   Cache: L1: 45 hits, L2: 26 hits, L3: 12 fetches
   Memory: 142 MB / 2048 MB (cache limit)
   GPU: Vertex 3ms, Fragment 7ms, Total 10ms

**Interpreting Metrics**:

* **FPS < 60**: Too many points or GPU bottleneck

  * Solution: Reduce maxPoints, enable LOD

* **Many L3 fetches**: Poor cache hit rate

  * Solution: Enable prefetching, increase cache size

* **GPU time > 16ms**: Rendering bottleneck

  * Solution: Reduce point count, simplify shaders

Profiling Tools
---------------

**Python (Write Performance)**:

.. code-block:: python

   import time

   start = time.time()
   with LuxarZarrCompiler('scene.zarr') as compiler:
       scene = compiler.create_scene()
       scene.add_points("data", positions, colors)
   elapsed = time.time() - start

   rate = len(positions) / elapsed
   print(f"Write rate: {rate/1e6:.2f}M points/sec")

**TypeScript (Query Performance)**:

.. code-block:: typescript

   // Browser console (when ?debug in URL):
   const state = window.__luxarDebug.getState();
   console.log('Query time:', state.lastQueryTime);
   console.log('Decode time:', state.lastDecodeTime);
   console.log('Cache stats:', state.cacheStats);

Best Practices Summary
----------------------

Write-Time Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~

1. ✅ Enable spatial indexing (``enable_spatial_index=True``)
2. ✅ Choose chunk size based on usage (512-4096 points)
3. ✅ Use compound ordering for nD data
4. ✅ Process data in batches for large datasets
5. ✅ Use ``EncodingMode.AUTO`` unless you have specific needs

Runtime Optimizations
~~~~~~~~~~~~~~~~~~~~~~

1. ✅ Enable prefetching (default: on)
2. ✅ Configure cache sizes based on available RAM
3. ✅ Use maxPoints limit for consistent FPS
4. ✅ Enable frustum culling (default: on)
5. ✅ Monitor performance overlay for bottlenecks

GPU Optimizations
~~~~~~~~~~~~~~~~~

1. ✅ Use world-space sizing (automatic, no pixel-based scaling)
2. ✅ Limit point count per frame (<1M for 60 FPS)
3. ✅ Use additive blending (faster than premultiplied)
4. ✅ Minimize post-processing effects (bloom is expensive)

Expected Performance
--------------------

**Realistic Targets**:

.. code-block:: text

   Dataset Size │ Write Time │ View Load │ Navigate │ FPS
   ─────────────┼────────────┼───────────┼──────────┼─────
   100K points  │   <1s      │   <1s     │   <50ms  │  60
   1M points    │   ~1s      │   ~2s     │   ~50ms  │  60
   10M points   │   ~10s     │   ~5s     │   ~100ms │  60
   100M points  │   ~100s    │   ~10s    │   ~100ms │  60
   1B points    │   ~1000s   │   ~20s    │   ~150ms │  60

**Key Insight**: Performance scales logarithmically with dataset size due to spatial indexing!

Common Issues and Solutions
---------------------------

Issue: Slow Initial Load
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: First view takes 30+ seconds

**Causes**:

* No spatial indexing (scanning all chunks)
* Large chunk size (loading MB when need KB)
* No prefetching (loading chunks serially)

**Solutions**:

.. code-block:: python

   # Ensure spatial indexing enabled
   compiler = LuxarZarrCompiler(
       'scene.zarr',
       enable_spatial_index=True,  # CRITICAL!
       ordering_method="morton",   # Or hilbert
   )

   # Adjust chunk size for your data
   # Rule: 32KB - 1MB per chunk
   compiler = LuxarZarrCompiler('scene.zarr', chunk_size=2048)

Issue: Low FPS During Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: <30 FPS when rotating camera

**Causes**:

* Too many points rendered (>1M)
* Expensive post-processing
* GPU bottleneck

**Solutions**:

.. code-block:: text

   # URL parameters:
   ?src=data.zarr&maxPoints=500000     # Limit rendered points
   ?src=data.zarr&bloom=off            # Disable expensive effects
   ?src=data.zarr&pointSize=1.0        # Smaller points = faster

Issue: High Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Browser using >4GB RAM, crashes on mobile

**Causes**:

* L1 cache too large
* Loading too many chunks
* Not cleaning old chunks

**Solutions**:

.. code-block:: text

   # Configure cache sizes (URL or config):
   ?src=data.zarr&l1Cache=50MB   # Reduce L1 from default 100MB
   ?src=data.zarr&l2Cache=500MB  # Reduce L2 from default 2GB

   # Browser automatically evicts LRU chunks when full

Advanced: Custom Culling
-------------------------

**For Very Large Datasets** (>100M points):

Implement custom spatial culling on the Python side:

.. code-block:: python

   def add_points_with_region(scene, name, all_positions, region_bounds):
       """Only write points within a specific region."""

       # Filter points
       mask = (
           (all_positions[:, 0] >= region_bounds[0]) &
           (all_positions[:, 0] <= region_bounds[1]) &
           (all_positions[:, 1] >= region_bounds[2]) &
           (all_positions[:, 1] <= region_bounds[3])
       )

       filtered_positions = all_positions[mask]

       scene.add_points(name, filtered_positions, ...)

       return len(filtered_positions)

   # Create multiple regions for LOD
   add_points_with_region(scene, "core_detail", positions, core_bounds)  # High detail
   add_points_with_region(scene, "periphery", positions[::10], full_bounds)  # Downsampled

**Result**: Viewer loads high detail in center, lower detail in periphery.

Summary
-------

**Performance Hierarchy** (Most Important First):

1. **Spatial indexing** - Enables chunk-based queries (200,000× speedup)
2. **Chunk size** - Balances granularity vs overhead
3. **Encoding mode** - Compression reduces network I/O
4. **Prefetching** - Cache hits eliminate network latency
5. **GPU budget** - Limit points for consistent frame rate

**Key Metrics to Watch**:

* **Cache hit rate**: Should be >80% with prefetching
* **FPS**: Should stay at 60 during navigation
* **Load time**: First view <10s, navigation <100ms
* **Memory**: <2GB browser RAM for typical datasets

**When Performance is Poor**:

1. Check spatial indexing is enabled
2. Verify chunk sizes are reasonable (1-4K points)
3. Monitor cache hit rate (enable prefetch)
4. Limit rendered points if GPU bound
5. Profile with browser dev tools

The combination of spatial indexing + intelligent caching + GPU rendering enables Luxar to handle billion-point datasets interactively!
