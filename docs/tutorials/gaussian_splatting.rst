Gaussian Splatting for Images
===========================================

Learn the concepts behind Gaussian splatting and when/how to use it effectively.

What is Gaussian Splatting?
----------------------------

**Core Idea**: Represent images as a collection of 2D/3D Gaussians instead of pixels/voxels.

**Why Gaussians?**

* **Physical accuracy**: Microscope PSF (point spread function) is Gaussian
* **Smoothness**: Natural for scientific imaging (continuous underlying signal)
* **Compression**: 100-1000 splats can represent 256×256 image (100× compression)
* **Denoising**: Fitting process naturally smooths noise
* **Differentiable**: Can optimize with gradient descent

**Mathematical Foundation**:

A 2D Gaussian splat is defined by:

.. math::

   G(x, y) = A \\cdot \\exp\\left(-\\frac{1}{2}[(x-\\mu_x)^2/\\sigma_x^2 + (y-\\mu_y)^2/\\sigma_y^2]\\right)

Where:

* ``(μ_x, μ_y)`` = center position
* ``(σ_x, σ_y)`` = width (standard deviation)
* ``A`` = amplitude (brightness)
* Optional: rotation, color

When to Use Gaussian Splats
----------------------------

**Ideal Use Cases**:

✅ **Microscopy images** - PSF is already Gaussian, natural representation
✅ **Sparse features** - Images with distinct objects on dark background
✅ **Compression** - Need 10-100× smaller files
✅ **Denoising** - Smoothing while preserving features
✅ **Super-resolution** - Upsample using fitted Gaussians

**Not Recommended**:

❌ **Dense textures** - Natural images (photos) better with JPEG/PNG
❌ **Sharp edges** - Text, CAD drawings need crisp boundaries
❌ **Binary masks** - Segmentations better as compressed arrays
❌ **Real-time processing** - Fitting is iterative (100-1000 iterations)

.. note::

   Gaussian splatting requires optional dependencies (PyTorch, SciPy). Install them with::

      pip install 'luxar[gsplats]'

Basic Gaussian Splat Fitting
-----------------------------

.. code-block:: python

   from luxar.gsplats import fit_gaussian_splats
   import numpy as np
   from skimage import data

   # Load or create image
   image = data.camera().astype(np.float32) / 255.0  # Normalize to [0, 1]

   # Fit Gaussian splats
   result = fit_gaussian_splats(
       image,
       n_iters=300,           # More iterations = better fit
       lr=0.01,               # Learning rate
       loss_type="l1",        # default; alternatives: "mse", "poisson"
       seed_method="edges",   # Edge-based initialization
   )

   # Results
   print(f"Fitted {len(result.centers)} splats")
   print(f"Final loss: {result.stats['final_loss']:.4f}")

   # Reconstruction (render back to volume/image)
   reconstructed = result.render_to_volume(shape=image.shape)

   # Quality metrics
   mse = np.mean((image - reconstructed) ** 2)
   psnr = 10 * np.log10(1.0 / mse)
   print(f"PSNR: {psnr:.2f} dB")

**What's Happening Internally**:

1. **Seed Generation**: Find initial splat centers

   * ``auto``: Fast edges + grid combination (default, recommended)
   * ``edges``: Edge-based seeding via Sobel gradients (fast)
   * ``decomposition``: Scale-hierarchical detection (principled)
   * ``grid``: Uniform grid seeding for spatial coverage
   * ``peaks``: Local maxima detection

2. **Initialization**: Create splats with reasonable parameters

   * Centers: From seeds
   * Widths: Based on local feature size
   * Amplitudes: From image intensity
   * Colors: From image (if RGB)

3. **Optimization**: Gradient descent to minimize reconstruction error

   * Render splats → Compare to target image → Compute loss
   * Backpropagate → Update splat parameters
   * Repeat for n_iters iterations

4. **Finalization**: Prune low-amplitude splats, pack results

Understanding Seed Generation Methods
--------------------------------------

Seed Selection is Critical
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Why it matters**: Poor initialization → poor fit, slow convergence

**Edges Method (Fast, Default)**:

* Uses Sobel gradients for edge detection and Poisson disk sampling
* Finds features at boundaries and transitions
* **Best for**: General-purpose microscopy images
* **Tradeoff**: May miss interior features

.. code-block:: python

   result = fit_gaussian_splats(
       image,
       seed_method="edges",
       seeds=1000,            # Number of seeds to generate (int = count)
   )

**Auto Method (Recommended)**:

* Combines edges + grid for balanced coverage
* Fast and robust default
* **Best for**: Most use cases

.. code-block:: python

   result = fit_gaussian_splats(
       image,
       seed_method="auto",    # Default — fast edges + grid combination
   )

**Decomposition Method (Principled)**:

* Scale-hierarchical detection via image decomposition
* Explicitly separates features by scale (coarse → fine)
* **Best for**: Noisy data, hierarchical structures
* **Tradeoff**: Slightly slower than edges method

.. code-block:: python

   result = fit_gaussian_splats(
       image,
       seed_method="decomposition",
   )

Optimization Parameters
-----------------------

Learning Rate
~~~~~~~~~~~~~

**What it is**: Step size for gradient descent

.. code-block:: python

   # Too low: Slow convergence, may not reach optimum
   result = fit_gaussian_splats(image, lr=0.001, n_iters=1000)  # Needs more iterations

   # Good: Balanced convergence speed and stability
   result = fit_gaussian_splats(image, lr=0.01, n_iters=300)  # Recommended

   # Too high: Unstable, overshoots, diverges
   result = fit_gaussian_splats(image, lr=0.1, n_iters=100)  # May fail!

**Rule of thumb**: Start with 0.01, increase if convergence is slow, decrease if training is unstable.

Loss Functions
~~~~~~~~~~~~~~

**L1 Loss** (Mean Absolute Error):

.. math::

   L = \\frac{1}{N}\\sum |I_{target} - I_{rendered}|

* Robust to outliers
* Good for noisy images
* **Recommended** for most cases

**L2 Loss** (Mean Squared Error):

.. math::

   L = \\frac{1}{N}\\sum (I_{target} - I_{rendered})^2

* Penalizes large errors heavily
* Good for clean images
* Can be sensitive to outliers/noise

.. code-block:: python

   # L1 for noisy microscopy
   result_l1 = fit_gaussian_splats(image, loss_type="l1")

   # L2 for clean synthetic data
   result_l2 = fit_gaussian_splats(image, loss_type="l2")

Advanced: Dynamic Operations
-----------------------------

**Concept**: Add/remove splats during optimization for better fit

**Motivation**: Fixed number of splats may be suboptimal:

* Too few: Can't represent fine details
* Too many: Overfitting, slow rendering

**Solution**: Dynamic splat management

.. code-block:: python

   from luxar.gsplats import DynamicOpsConfig

   result = fit_gaussian_splats(
       image,
       n_iters=500,
       enable_dynamic_ops=True,  # Note: True is already the default
       dynamic_config=DynamicOpsConfig(),  # Uses sensible defaults
   )

**How it works**:

Dynamic operations perform **fixed-pool relocation** — weak splats (lowest
amplitude) are relocated to high-residual regions. The total splat count
remains fixed throughout training. This avoids the complexity of managing a
changing number of splats while still allowing the optimization to adapt
spatial coverage to where it is needed most.

Saving and Visualizing Results
-------------------------------

.. code-block:: python

   from luxar.encoding import EncodingMode

   # Fit splats
   result = fit_gaussian_splats(image, n_iters=300)

   # Save to zarr
   result.save("fitted_splats.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)

   # Visualize in viewer
   # Terminal: luxar serve fitted_splats.gsplats.zarr --viewer

.. image:: ../images/docs/gsplats-scene.png
   :alt: Luxar viewer showing Gaussian splats
   :width: 100%

**What you'll see**:

* Each Gaussian rendered as a smooth blob
* Overlapping splats create the image
* Can navigate, inspect individual splats
* Zoom in: See individual Gaussian kernels

Quality vs Compression Tradeoff
--------------------------------

.. code-block:: python

   # High quality (more splats, more iterations)
   result_hq = fit_gaussian_splats(
       image,
       seed_method="edges",
       n_iters=1000,
       lr=0.005,  # Smaller steps for fine-tuning
       # Result: 2000 splats, PSNR 35dB, 50× compression
   )

   # Balanced (recommended)
   result_balanced = fit_gaussian_splats(
       image,
       n_iters=300,
       lr=0.01,
       # Result: 800 splats, PSNR 32dB, 100× compression
   )

   # High compression (fewer splats, faster)
   result_compressed = fit_gaussian_splats(
       image,
       n_iters=100,
       lr=0.02,
       seed_method="edges",  # Faster seeding
       # Result: 200 splats, PSNR 28dB, 500× compression
   )

**Choosing the Right Tradeoff**:

* **Interactive exploration**: Balanced (fast load, good quality)
* **Publication figures**: High quality (best reconstruction)
* **Network-limited**: High compression (fast transfer)

Calibrating K via Blind-Spot Cross-Validation
----------------------------------------------

Picking the splat count by eye is hard — too few and you under-fit, too
many and you start memorising noise. The ``luxar gsplat cal`` command
sweeps :math:`K` and reports the principled optimum :math:`K^{\star}`
(the held-out PSNR peak) plus the dataset's noise-floor PSNR ceiling.

.. code-block:: bash

   # Default 10-point sweep matching the manuscript ({1K, 2K, ..., 512K})
   luxar gsplat cal kidney_dapi.tiff cal.json --device cuda

   # Faster: 5-point sweep
   luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000

   # Explicit grid
   luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000'

   # Multi-page PDF report (with --keep-fits also enables slice montages)
   luxar gsplat cal volume.tiff cal.json --pdf cal.pdf --keep-fits fits/

Or programmatically:

.. code-block:: python

   from luxar.gsplats.calibration import calibrate, build_k_grid

   ks = build_k_grid(n_points=10, k_min=1_000, k_max=512_000)
   result = calibrate(volume, k_grid=ks, fit_kwargs={"device": "cuda"})

   print(f"K* = {result.held_out_peak.k_star:,}")
   print(f"type = {result.held_out_peak.type}")  # peak | plateau | signal_limited
   print(f"noise floor σ̂ = {result.noise_floor.sigma_hat:.4f}")

The blind-spot trick: 5% of voxels are masked and replaced with the
median of their 26-neighbour donut before fitting. The optimiser never
sees the original noisy values at those positions, so PSNR computed at
the held-out positions against the *original* values measures *signal*
recovery rather than fidelity to the noise. Adding capacity beyond
:math:`K^{\star}` starts memorising noise — held-out PSNR drops, even
though training PSNR keeps rising. See ``manuscript/supp_doc/splat_count_vs_quality/``
for the full theory and per-dataset curves.

After calibration, re-run the fit at the recommended budget:

.. code-block:: bash

   luxar gsplat fit volume.zarr out.gsplats.zarr --seeds <K*>

Building Streaming LOD Ladders
-------------------------------

A fitted ``.gsplats.zarr`` is a single flat container of N splats. For
progressive streaming and view-dependent rendering, build a representation
topology on top with ``luxar gsplat lod --recipe ...`` (the ``--recipe`` flag
is required). Recipes are ordered by dataset scale:

* **flat** — a single leaf (no LOD, no partition).
* **additive** — same N splats, *reordered* so the prefix sum at any k splats
  is the best L² approximation of the full scene (streaming-friendly).
* **partitioned** — a spatial BSP partition where each part carries its own
  additive ladder (frustum-cull off-screen parts; stream detail in view).
* **multiscale** — an unbalanced ``kind=lod``: a coarse substitutive cap for
  the far view plus a ``partitioned`` fine branch for close-up.

plus the lower-level primitives ``substitutive`` (synthesise M < N
representative splats per coarser level) and ``pyramid`` (the balanced
substitutive × additive matrix).

.. code-block:: bash

   # Canonical end-to-end pipeline: cal → fit → lod --recipe
   luxar gsplat cal volume.tiff cal.json --device cuda
   luxar gsplat fit volume.tiff fitted.gsplats.zarr --seeds <K*>

   # stream — a streaming prefix ladder
   luxar gsplat lod fitted.gsplats.zarr scene.gsplats.zarr --recipe stream --n-lods 4

   # tiles / overview — the large-data topologies
   luxar gsplat lod fitted.gsplats.zarr part.gsplats.zarr --recipe tiles --max-elements 250000
   luxar gsplat lod fitted.gsplats.zarr ms.gsplats.zarr --recipe overview --compression-factor 8

   # levels — synthesised representative levels
   luxar gsplat lod fitted.gsplats.zarr levels.gsplats.zarr --recipe levels -L 3 -K 4
   luxar gsplat lod fitted.gsplats.zarr pyramid.gsplats.zarr --recipe levels -K 4 -L 3 --n-lods 4

Programmatically:

.. code-block:: python

   from luxar.gsplats import (
       make_additive_lod, make_substitutive_lod, make_lod_pyramid,
   )

   additive = make_additive_lod(data, n_lods=4, method="greedy")
   # additive.additive_prefix(2) → prefix of additive sub-LODs 0+1+2

   pyramid = make_substitutive_lod(data, compression_factor=4, levels=3)
   # pyramid.n_substitutive == 4 (one GSplatData; finest at index 0, coarsest at 3)

   matrix = make_lod_pyramid(
       data, compression_factor=4, levels=3, n_additive_lods=4,
   )
   # matrix.n_substitutive == 4; each level carries its own 4-step additive ladder

See ``packages/luxar/src/luxar/gsplats/lod/README.md`` for the
full algorithm spec (greedy vs self-energy vs mass vs amplitude
orderings; k-means+Lloyd vs hierarchical greedy clustering; breakpoint
strategies; complexity).

Summary
-------

**Key Concepts**:

* Gaussian splats represent images as continuous functions
* Seed generation determines initialization quality
* Optimization is iterative gradient descent
* Dynamic operations adapt to image complexity
* Tradeoff: Quality vs compression vs speed

**When it Shines**:

* Microscopy with Gaussian PSF
* Sparse features on dark background
* Need extreme compression
* Want denoising + compression together

**Next**: :doc:`performance_optimization` - Handle billion-point datasets efficiently.
