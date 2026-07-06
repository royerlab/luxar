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
   compiler = LuxarZarrCompiler('scene.luxar.zarr')

   # For time-series (4D with time animation)
   compiler = LuxarZarrCompiler('timeseries.luxar.zarr')

   # For massive static datasets with best compression
   compiler = LuxarZarrCompiler(
       'huge.luxar.zarr',
       ordering_method="hilbert",  # Better compression for large datasets
   )

**Calculation**:

Target chunk size in bytes: 32KB - 1MB

.. code-block:: text

   3D positions (3 x float32 = 12 bytes per point)
   + RGB colors (3 x uint8 = 3 bytes per point)
   = 15 bytes per point

   Target 64KB chunks:
   chunk_size = 64 * 1024 / 15 = ~4369 points

   Round to power of 2: 4096 points

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

   # Default: Hilbert curve (best spatial locality, recommended)
   compiler = LuxarZarrCompiler('scene.luxar.zarr')  # ordering_method="hilbert" by default

   # Alternative: Morton curve (faster to compute, slightly worse locality)
   compiler = LuxarZarrCompiler('scene.luxar.zarr', ordering_method="morton")

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

   compiler = LuxarZarrCompiler('scene.luxar.zarr', encoding_mode=EncodingMode.AUTO)

   # Positions: Analyzes range, uses uint16 if |max-min| < 65536
   # Colors: uint8 for SDR, geolog_perchannel_u16 for HDR
   # Radii: Broadcasts if uniform, uint16 if varying

**PRECISION Mode** (Lossless):

Keeps float32 for everything:

.. code-block:: python

   compiler = LuxarZarrCompiler('scene.luxar.zarr', encoding_mode=EncodingMode.PRECISION)

   # Positions: float32 (no quantization)
   # Colors: float32 (full HDR range)
   # Radii: float32

   # Use when: Precision critical, storage not constrained

**MEMORY Mode** (Aggressive):

Maximum compression:

.. code-block:: python

   compiler = LuxarZarrCompiler('scene.luxar.zarr', encoding_mode=EncodingMode.MEMORY)

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

   ?src=data.luxar.zarr                # Prefetching enabled by default
   ?src=data.luxar.zarr&no-prefetch    # Disable prefetching (debugging)
   ?src=data.luxar.zarr&prefetch-debug # Enable prefetch debug logging

**Impact**:

* Prefetching enabled (default): 80-95% cache hit rate
* Prefetching disabled (``no-prefetch``): 20-30% cache hit rate (only repeat views cached)

Optimization 5: Viewer Configuration
-------------------------------------

**Automatic Optimizations**:

The viewer applies several optimizations automatically:

* **World-space point sizing**: Points specified in world units are sized based on camera distance
* **Frustum culling**: Only points in the camera's field of view are rendered (typically excludes 90%+ of points)
* **Chunk-based spatial queries**: Only chunks overlapping the view are loaded

**Cache Control** (URL parameters):

.. code-block:: text

   ?src=data.luxar.zarr                # Default: all caching enabled
   ?src=data.luxar.zarr&no-cache       # Disable L0/L1/L2 caching
   ?src=data.luxar.zarr&clear-cache    # Clear all caches on startup
   ?src=data.luxar.zarr&cache-debug    # Enable cache debug logging

**Rendering Configuration** can be controlled via ``viewer_config`` in the Zarr scene metadata
(set at write time in Python) or interactively via the rendering controls panel (press **R**).

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
       'billion_points.luxar.zarr',
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

   Press P to toggle performance stats (FPS, frame time, memory)
   Press M to cycle the data loading monitor (chunk loading, cache stats)
   Press H for help overlay with all keyboard shortcuts

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
   from luxar import LuxarZarrCompiler, Dimensions

   dims = Dimensions.default_3d()
   start = time.time()
   with LuxarZarrCompiler('scene.luxar.zarr') as compiler:
       scene = compiler.create_scene(dimensions=dims)
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

**Realistic Targets** (approximate, hardware-dependent):

.. code-block:: text

   Dataset Size │ Write Time │ View Load │ Navigate │ FPS
   ─────────────┼────────────┼───────────┼──────────┼─────
   100K points  │   <1s      │   <1s     │   <50ms  │  60
   1M points    │   ~1s      │   ~2s     │   ~50ms  │  60
   10M points   │   ~10s     │   ~5s     │   ~100ms │  60
   100M points  │   ~100s    │   ~10s    │   ~100ms │  60
   1B points    │   ~1000s   │   ~20s    │   ~150ms │  60

.. note::

   These are order-of-magnitude estimates on typical hardware (modern laptop/desktop with dedicated GPU, SSD, broadband). Actual performance varies with hardware, dataset structure, and network conditions.

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
       'scene.luxar.zarr',
       enable_spatial_index=True,  # CRITICAL!
       ordering_method="hilbert",  # Default, best locality
   )

   # Use spatial ordering for best query performance
   compiler = LuxarZarrCompiler('scene.luxar.zarr')  # Hilbert by default

Issue: Low FPS During Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: <30 FPS when rotating camera

**Causes**:

* Too many points rendered (>1M)
* Expensive post-processing
* GPU bottleneck

**Solutions**:

* Reduce point count by using smaller datasets or chunked loading
* Disable expensive post-processing effects via the rendering controls panel (press **R**)
* Configure rendering settings via ``viewer_config`` in scene metadata at write time

Issue: High Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Browser using >4GB RAM, crashes on mobile

**Causes**:

* L1 cache too large
* Loading too many chunks
* Not cleaning old chunks

**Solutions**:

* The viewer uses LRU eviction automatically when caches are full
* Use ``?clear-cache`` URL parameter to reset caches on startup
* Use ``?no-cache`` to disable caching entirely for debugging
* Cache sizes are managed automatically by the viewer

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
