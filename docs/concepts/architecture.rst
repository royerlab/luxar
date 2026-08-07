Architecture and Core Concepts
================================

This guide explains the fundamental ideas, design philosophy, and architectural decisions behind Luxar.

Big Picture: What is Luxar?
----------------------------

Luxar is a **high-performance system for compiling and visualizing arbitrary-sized n-dimensional scenes** containing points, lines, Gaussian splats, and triangle meshes. It enables:

* **Interactive exploration** of billion-primitive datasets (points, lines, splats; meshes to millions of triangles) at 60 FPS
* **Arbitrary dimensionality** (3D, 4D, 5D, nD) with intuitive navigation
* **Memory efficiency** through progressive loading and intelligent caching
* **High quality rendering** with HDR support and post-processing effects
* **Multi-primitive scenes**: Points, lines, Gaussian splats, and triangle meshes (the meshes are shaded; the other three are emissive)

**Key Innovation**: Combine spatial indexing with nD hypersphere slicing to enable interactive exploration of datasets that don't fit in memory.

Luxar delivers visualization performance limited only by your graphics card, display resolution, and network bandwidth—not by software constraints.

Design Philosophy
-----------------

Luxar is built on several core principles:

1. **Progressive Everything**

   *Motivation*: Billion-point datasets cannot fit in memory.

   *Solution*: Write data progressively (no caching), load data lazily (chunk-based), render incrementally (spatial culling).

   *Benefit*: Handle TB-scale datasets on GB-scale machines.

2. **Spatial Locality is Sacred**

   *Motivation*: Adjacent points in space should be adjacent in memory for compression and cache efficiency.

   *Technique*: Morton/Hilbert space-filling curves map nD coordinates to 1D ordering while preserving locality.

   *Impact*: 2-10× better compression, 5-20× faster queries.

3. **nD First, 3D Second**

   *Philosophy*: Don't reduce nD data to 3D - visualize it natively.

   *Approach*: Display 3 dimensions, slice through others with hypersphere visibility.

   *Result*: Time-series, multi-channel, hyperspectral data visualized naturally.

4. **Zero-Copy When Possible**

   *Motivation*: Memory copies are expensive for large datasets.

   *Implementation*: Use views, references, broadcasting instead of duplication.

   *Example*: Uniform color stored once, broadcasted to 1M points = 99.9% memory savings.

5. **Semantic Awareness**

   *Idea*: Know what data represents (positions, colors, radii) to make smart encoding decisions.

   *Benefit*: Automatic quantization with appropriate precision for each data type.

Architectural Layers
--------------------

Luxar consists of five distinct layers:

Server Layer (FastAPI)
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Serve Zarr datasets over HTTP for browser-based visualization

**Key Components**:

* ``create_server_app()``: Factory function for FastAPI applications
* ``DirectoryListingStaticFiles``: Custom static file handler with Zarr support
* ``NetworkSimulationMiddleware``: Test performance under realistic network conditions
* CORS middleware: Enable cross-origin requests from viewer

**Architecture**:

The server layer sits between storage and viewer, providing:

1. **Static File Serving**: Zarr chunks and metadata served as static files
2. **Directory Listing**: JSON directory listing for Zarr structure discovery
3. **Health Checks**: ``/health`` endpoint for monitoring
4. **Network Simulation**: Optional middleware to simulate bandwidth, latency, packet loss

**Design Decisions**:

*Why FastAPI?*

* Modern async framework (handles many concurrent requests efficiently)
* Automatic OpenAPI documentation
* Type hints for request/response validation
* Easy middleware integration

*Why static file serving?*

* Zarr is designed for HTTP range requests
* No need for complex query API - chunks are addressed directly
* Browser can cache chunks efficiently
* CDN-friendly for production deployments

*Why directory listing?*

* Viewer needs to discover Zarr structure
* Standard HTTP directory indices do not work for all browsers
* JSON format enables programmatic access

**Server Creation Example**::

   from luxar.cli.main import create_server_app
   import uvicorn

   # Create configured FastAPI app
   app = create_server_app("/path/to/data.luxar.zarr")

   # Run server
   uvicorn.run(app, host="127.0.0.1", port=8000)

The server automatically:

* Serves ``.zmetadata`` for fast initialization
* Provides directory listings for Zarr groups
* Handles CORS for cross-origin viewer access
* Includes health check at ``/health``

**Integration Testing**:

The server is designed for testing without mocking:

* Create real server instances in tests
* Make actual HTTP requests
* Verify end-to-end behavior
* Test with real Zarr data

See :doc:`../tutorials/programmatic_server` for detailed examples.



Python Layer (Data Creation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Create and encode visualization data

**Packages**:

* ``luxar.core`` - Scene graph data structures
* ``luxar.validation`` - Input validation with helpful errors
* ``luxar.encoding`` - Semantic type-aware array encoding
* ``luxar.io`` - Progressive writing with spatial ordering

**Key Concepts**:

* **Scene Graph**: Hierarchical organization (Scene → Groups → DataNodes)
* **Dimensions**: Define nD space with units, ranges, navigation
* **Transforms**: 4×4 matrices compose hierarchically (parent × child)
* **Semantic Types**: COORDINATE, COLOR, POSITIVE_SCALAR, BOUNDED_SCALAR
* **Encoding Modes**: AUTO (smart), PRECISION (lossless), MEMORY (aggressive compression)

**Data Flow**:

::

    User Data (numpy arrays)
        ↓
    Validation (check shapes, ranges, types)
        ↓
    Semantic Typing (understand what data represents)
        ↓
    Encoding (quantize based on semantic type)
        ↓
    Spatial Ordering (Morton/Hilbert curves)
        ↓
    Progressive Writing (no memory cache, stream to disk)
        ↓
    Zarr Store (compressed, indexed, ready to serve)

Storage Layer (Zarr Format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Efficient storage and streaming

**Key Innovations**:

* **Chunk-based spatial index**: Precomputed bounding boxes enable O(chunks × dims) queries
* **Compound ordering**: Discrete dimensions (time, channel) → spatial dimensions (x, y, z)
* **Compression pipeline**: Quantization (2-4×) + blosc/zstd (2-10×) = 4-40× total
* **Metadata consolidation**: Single .zmetadata file for fast initialization

**Design Decisions**:

*Why Zarr?*

* Cloud-native format (HTTP range requests)
* Chunked storage (load only what's visible)
* Compression support (built-in)
* Python and JavaScript libraries available

*Why Morton/Hilbert ordering?*

* Space-filling curves preserve spatial locality
* Adjacent points in nD space → adjacent in 1D storage
* Better compression ratios (similar values group together)
* Faster queries (scan fewer chunks)

TypeScript Layer (Viewer)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Interactive visualization in the browser

**Packages**:

* ``cache`` - Three-level caching (Memory, OPFS, HTTP)
* ``data`` - Scene loading, spatial queries, nD slicing
* ``rendering`` - WebGL materials, post-processing
* ``scene`` - Scene graph management, animation
* ``controls`` - Camera controls (orbit, fly)
* ``input`` - Keyboard/mouse handling
* ``ui`` - GUI components (sliders, monitors)

**Key Algorithms**:

1. **Chunk Query (O(chunks × dims))**:

   Test each chunk's bounding box for AABB intersection with query region.
   Load only chunks whose bounds overlap the view.

2. **nD Hypersphere Slicing**:

   Points with radius R are visible if their distance in non-displayed
   dimensions ≤ R. Effective radius shrinks as you navigate away:

   ``effective_radius = √(radius² - distance_in_nondisplayed_dims²)``

3. **Intelligent Prefetching**:

   When loading chunk N, predict user will navigate to adjacent chunks
   (±1 in each dimension). Prefetch in background for instant loading.

4. **Array Decoding**:

   Reverse Python encoding: dequantize integers → floats, expand
   broadcasted values, resolve LUT indices.

**Design Decisions**:

*Why multiple cache levels?*

* L0 (decompressed in-memory): Ready-to-use decoded chunks, no re-decompress
* L1 (memory, segmented LRU): Ultra-fast (~1μs) compressed chunks
* L2 (OPFS): Persistent (~1ms) across page reloads
* HTTP: Unlimited (~100ms) source of truth

(An additional "S-cache" — the SliceCache — sits above L0 and LRU-caches
fully *decoded* per-slice geometry ladders, so revisiting a slice skips the
whole query→fetch→decode pipeline.)

*Why nD slicing vs dimension reduction?*

* Preserves all data dimensions
* Intuitive keyboard navigation (1-9 select dimension, [ ] navigate)
* No information loss from projection

WebGL Layer (Rendering)
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: GPU-accelerated rendering

**Key Techniques**:

1. **World-Space Point Sizing**:

   Points are specified in world units (micrometers, etc.) and
   automatically sized based on camera distance using angular diameter.

2. **Gaussian Splatting**:

   Smooth point rendering using 2D Gaussian kernel in fragment shader.
   Creates soft, anti-aliased points.

3. **HDR Rendering**:

   Float16 framebuffers support colors > 1.0 for scientific accuracy.
   Tone mapping converts to display range.

4. **Multi-Pass Effects**:

   Post-processing effects (bloom, detector noise) composed dynamically
   based on enabled features.

Core Concepts Deep Dive
------------------------

nD Visualization Paradigm
~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Problem**: How do you visualize 5D data (x, y, z, time, channel) on a 2D screen?

**Traditional Approach** (Dimension Reduction):

* Project 5D → 3D using PCA, t-SNE, UMAP
* **Loss**: Lose actual spatial relationships, lose 2 dimensions of information

**Luxar Approach** (nD Slicing):

* Display 3 dimensions (x, y, z)
* Slice through 2 dimensions (time, channel)
* Points visible based on hypersphere intersection

**Benefits**:

* No information loss - all 5 dimensions preserved
* Intuitive - navigate time like any other dimension
* Fast - spatial index makes queries O(chunks) not O(points)

Spatial Indexing Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Problem**: Loading all points to find which are visible is too slow for billion-point datasets.

**Solution**: Chunk-based AABB spatial index

**How it Works**:

1. **Preprocessing** (Python, one-time):

   * Divide points into chunks (e.g., 2,048 points per chunk)
   * Calculate bounding box for each chunk
   * Store ``chunk_bounds`` array: ``[xmin, xmax, ymin, ymax, ...]`` per chunk

2. **Query** (TypeScript, real-time):

   * User navigates to position P with tolerance T
   * Test each chunk: does its AABB intersect the query box?
   * Load only matching chunks (typically 1-10 out of 100-1000 chunks)

3. **Point Filtering** (TypeScript, real-time):

   * Within loaded chunks, test each point for visibility
   * Display points within hypersphere

**Complexity**:

* Preprocessing: O(N) to calculate bounds
* Query: O(chunks × dims) - typically 1000 chunks × 5 dims = 5000 ops
* vs. Naive: O(N) - would be 1,000,000,000 ops for billion points!
* **Speedup**: ~200,000× faster

Compound Ordering for nD Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Problem**: How to organize nD data for efficient time-series animation AND spatial queries?

**Naive Approach**: Morton order on all dimensions

* Pros: Simple, preserves locality
* Cons: Time frames scattered throughout file, poor compression

**Luxar Approach**: Two-level compound ordering

1. **Primary**: Group by discrete dimensions (time, channel)

   * All time=0 points together
   * All time=1 points together
   * etc.

2. **Secondary**: Morton order within each group (x, y, z)

   * Spatial locality within each time frame
   * Better compression (similar time frames compress together)

**Result**:

* Time-series animation: Load contiguous chunks (excellent I/O)
* Spatial queries: Morton order within groups (excellent locality)
* Compression: Grouping similar data (2-3× better ratios)

Gaussian Splatting Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Idea**: Fit smooth Gaussian "splats" to images instead of discrete pixels

**Motivation**:

* Scientific images are inherently continuous (microscopy, astronomy)
* Gaussian splats are a natural representation (PSF = Gaussian)
* Enables super-resolution, denoising, compression

**Luxar Integration**:

* Fit splats using ``luxar.gsplats.fit_gaussian_splats()``
* Save as a ``GSplats`` node in the scene graph
* Render using ``GaussianSplatModel`` (differentiable PyTorch model)
* Visualize in viewer with same spatial indexing as points

**Use Cases**:

* Compress microscopy images 10-100×
* Denoise while preserving features
* Super-resolution reconstruction
* Multi-scale image analysis

Transform Composition
~~~~~~~~~~~~~~~~~~~~~

**The Problem**: Matrix multiplication order is confusing (left vs right multiply)

**Luxar Convention**: Right-multiply with explicit ``compose()`` function

.. code-block:: python

   from luxar.core.transforms import translate, rotate, compose

   # Create transforms
   T1 = translate(10, 0, 0)   # Move right
   T2 = rotate(90, 'z')       # Rotate 90° around Z
   T3 = translate(0, 5, 0)    # Move up

   # Compose: applies T1 first, T3 last
   result = compose(T1, T2, T3)

   # Mathematically: result = I @ T1 @ T2 @ T3
   # Applied to point: (T1 @ T2 @ T3) @ point

**Key Insight**: ``compose(A, B, C)`` reads left-to-right but uses right-multiplication internally.

**Why This Matters**:

* Transform composition order affects results (not commutative)
* Parent transforms apply after child transforms (hierarchical)
* NumPy (row-major) vs THREE.js (column-major) requires transposition

Encoding Strategy Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Problem**: One encoding doesn't fit all data types

**Semantic Type System**:

Each array is classified by what it represents:

* **COORDINATE**: Positions in space (bounds can vary, quantize with tolerance)
* **COLOR**: RGB values (bounded [0,1] for SDR, unbounded for HDR)
* **POSITIVE_SCALAR**: Radii, amplitudes (always ≥0, relative precision matters)
* **BOUNDED_SCALAR**: Sharpness, opacity (fixed range, absolute precision)

**Encoding Decision Tree**:

.. code-block:: text

   Is data uniform (all values identical)?
   ├─ YES → Broadcast (store 1 value, 99.9% savings)
   └─ NO ↓

   Does exact duplicate exist in registry?
   ├─ YES → Array Reference (store pointer, 100% dedup)
   └─ NO ↓

   Does data have ≤256 unique values?
   ├─ YES → LUT Encoding (palette + indices, 4-8× savings)
   └─ NO ↓

   Dtype Encoding based on semantic type:
   - COORDINATE → uint16 (±32K range, ~0.1% precision)
   - COLOR (SDR) → uint8 (256 levels per channel)
   - COLOR (HDR) → geolog_perchannel_u16 (per-channel true-log; decoded to float32)
   - POSITIVE_SCALAR → uint16 (log quantization for range)
   - BOUNDED_SCALAR → uint8 (linear quantization)

**Result**: Automatic optimal encoding with 4-40× compression while preserving visual quality.

Performance Model
-----------------

Understanding Luxar's performance characteristics.

.. note::

   The numbers below are approximate order-of-magnitude estimates based on typical hardware (modern laptop/desktop with dedicated GPU, SSD storage, broadband network). Actual performance varies with hardware, dataset characteristics, and network conditions.

Write Performance (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   1M points, 3D:
   - Without spatial ordering: ~5M points/sec
   - With Morton ordering: ~1M points/sec (sorting overhead)
   - With chunk bounds: ~800K points/sec (AABB calculation)

   Tradeoff: 5× slower write, 200,000× faster queries (well worth it!)

Read Performance (TypeScript)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Query for visible points:
   - Chunk query: ~1ms (O(chunks × dims), typically 1000 × 5)
   - HTTP fetch: ~100ms per chunk (network latency)
   - Decode: ~5ms per chunk (dequantization)
   - Slice: ~10ms per chunk (visibility filtering)

   Total: ~115ms for first load, ~16ms for cached
   With prefetching: ~16ms average (cache hit rate >80%)

Rendering Performance (WebGL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   GPU rendering (60 FPS target):
   - 100K points: Vertex shader ~2ms, Fragment ~4ms = ~6ms ✓
   - 1M points: Vertex ~8ms, Fragment ~8ms = ~16ms ✓
   - 10M points: Vertex ~60ms, Fragment ~60ms = ~120ms (~8 FPS) ⚠

   Optimization: Spatial culling reduces points to render by 10-100×

When to Use Luxar
-----------------

Luxar is Ideal For:
~~~~~~~~~~~~~~~~~~~

* ✅ **Large datasets** (>100K primitives, up to billions of points/lines/splats, or millions of triangles)
* ✅ **nD scientific data** (microscopy time-series, multi-channel imaging)
* ✅ **Interactive exploration** (need to navigate/inspect data)
* ✅ **Remote visualization** (data on server, view in browser)
* ✅ **Publication figures** (HDR, high quality rendering)

Consider Alternatives For:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ❌ **Small datasets** (<10K points) - overhead not worth it, use matplotlib/plotly
* ❌ **Real-time streaming** - Luxar is for static datasets, not live data streams
* ❌ **Mesh authoring / CAD** - Luxar *renders* triangle meshes (``add_mesh``, ``luxar mesh import``), but it does not edit them, and a mesh gets no level-of-detail or spatial partitioning (see ``docs/specs/MESH_NODE_SPEC.md`` §9), so one very large surface loads whole and up front, with no progressive refinement
* ❌ **2D plots** - Use specialized 2D libraries (bokeh, plotly)

Common Workflows
----------------

Workflow 1: Microscopy Time-Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # You have: 4D microscopy data (X, Y, Z, Time)
   # Goal: Visualize cell tracking over time

   from luxar.core import Scene, Dimensions, Dimension
   from luxar.io import LuxarZarrCompiler
   import numpy as np

   # Define 4D dimensions
   dims = Dimensions([
       Dimension("X", unit="um", spatial=True, display=True),
       Dimension("Y", unit="um", spatial=True, display=True),
       Dimension("Z", unit="um", spatial=True, display=True),
       Dimension("Time", discrete=True, display=False, step=0.5),
   ])

   # Write with compound ordering for efficient time navigation
   with LuxarZarrCompiler('cells.luxar.zarr') as c:  # Hilbert ordering by default
       scene = c.create_scene(dimensions=dims)

       # Track cell positions over time
       for t, positions_t in enumerate(cell_positions_by_time):
           # Add time coordinate
           positions_4d = np.column_stack([positions_t, np.full(len(positions_t), t * 0.5)])
           scene.add_points(f"cells_t{t}", positions_4d, colors=track_colors[t])

   # Result: Navigate time with [ ] keys, see cells move

Workflow 2: Multi-Channel Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # You have: 5D data (X, Y, Z, Channel, Wavelength)
   # Goal: Explore different fluorescence markers

   dims = Dimensions([
       Dimension("X", unit="um", spatial=True),
       Dimension("Y", unit="um", spatial=True),
       Dimension("Z", unit="um", spatial=True),
       Dimension("Channel", discrete=True, display=False, categories=["DAPI", "GFP", "mCherry"]),
       Dimension("Wavelength", unit="nm", display=False, discrete=True),
   ])

   # At most 3 dimensions can be displayed, so Channel and Wavelength are
   # non-displayed (navigated with the [ ] keys). A non-spatial, non-displayed
   # dimension must be discrete, so Wavelength is marked discrete=True
   # explicitly (otherwise Luxar sets it automatically and emits a warning).

   # Categorical dimension allows channel selection by name
   # Discrete + spatial ordering groups all DAPI points together

Workflow 3: Gaussian Splat Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # You have: 2D/3D microscopy image
   # Goal: Fit Gaussian splats for compression/denoising

   from luxar.gsplats import fit_gaussian_splats

   # Fit splats to image
   result = fit_gaussian_splats(
       image,
       n_iters=300,
       lr=0.01,
       loss_type="l1",
       seed_method="edges",  # Edge-based seeding
   )

   # Save for visualization
   result.save("fitted.gsplats.zarr")

   # Reconstruction (render back to image/volume)
   reconstructed = result.render_to_volume(shape=image.shape)
   compression_ratio = image.nbytes / result.centers.nbytes
   print(f"Compression: {compression_ratio:.1f}×")

Key Takeaways
-------------

1. **Luxar is designed for scale**: Billion points, TB datasets, interactive exploration
2. **Spatial indexing is the foundation**: Makes everything fast
3. **nD is first-class**: Not an afterthought, built-in from the start
4. **Semantic awareness**: Understanding data types enables smart encoding
5. **Progressive architecture**: No memory bottlenecks, handle any size data
6. **Browser-based**: No installation, works on any device, shareable URLs

Next Steps
----------

* **Try it**: Start with a Quick Start from any package README
* **Explore code**: Well-documented with extensive examples
* **Ask questions**: GitHub issues/discussions
