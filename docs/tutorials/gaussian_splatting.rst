Tutorial 3: Gaussian Splatting for Images
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
* Optional: rotation, color, sharpness

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
       loss_type="l1",        # L1 loss is robust to outliers
       seed_method="gaussian"  # Smart initialization
   )

   # Results
   print(f"Fitted {len(result.centers)} splats")
   print(f"Final loss: {result.stats['final_loss']:.4f}")

   # Reconstruction
   reconstructed = result.model.render_numpy()

   # Quality metrics
   mse = np.mean((image - reconstructed) ** 2)
   psnr = 10 * np.log10(1.0 / mse)
   print(f"PSNR: {psnr:.2f} dB")

**What's Happening Internally**:

1. **Seed Generation**: Find initial splat centers

   * ``multiscale_gaussian``: Laplacian of Gaussian feature detection
   * ``clahe``: CLAHE-enhanced local maxima
   * ``uniform``: Regular grid (simple, fast)

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

**Multiscale Gaussian (Recommended)**:

* Detects features at multiple scales (large + small structures)
* Uses Laplacian of Gaussian pyramid
* **Best for**: General-purpose microscopy images
* **Tradeoff**: Slightly slower initialization (~1s for 512×512)

.. code-block:: python

   result = fit_gaussian_splats(
       image,
       seed_method="gaussian",
       seed_kwargs={
           "scales": (2.0, 4.0, 8.0, 16.0),  # Detection scales
           "percentile_thresh": 75.0,         # Intensity threshold
           "min_distance": 2.0,               # Min distance between seeds
       }
   )

**CLAHE (Best for Low-Contrast)**:

* Enhances local contrast before detection
* Finds features in dim regions
* **Best for**: Low-SNR images, uneven illumination
* **Tradeoff**: Can detect noise as features

.. code-block:: python

   result = fit_gaussian_splats(
       image,
       seed_method="gaussian",
       seed_kwargs={
           "apply_clahe": True,         # Enable CLAHE preprocessing
           "clahe_tile_size": 32,       # Local region size
           "clahe_clip_limit": 16.0,    # Contrast limit
           "percentile_thresh": 70.0,   # Detection threshold
       }
   )

**Uniform Grid (Fast Baseline)**:

* Regular grid of splats
* No feature detection
* **Best for**: Quick baseline, evenly distributed data
* **Tradeoff**: Many splats needed for good fit

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

   result = fit_gaussian_splats(
       image,
       n_iters=500,
       enable_dynamic_ops=True,
       dynamic_config={
           "split_threshold": 0.1,    # Split high-gradient splats
           "prune_threshold": 0.01,   # Remove low-amplitude splats
           "split_interval": 50,      # Check every 50 iterations
       }
   )

**Operations**:

* **Split**: Large splats in high-gradient regions → two smaller splats
* **Prune**: Low-amplitude splats → remove (don't contribute to image)
* **Clone**: Undersaturated regions → duplicate nearby splat

**Result**: Adapts number of splats to image complexity (500 → 1200 for complex images).

Saving and Visualizing Results
-------------------------------

.. code-block:: python

   # Fit splats
   result = fit_gaussian_splats(image, n_iters=300)

   # Save to zarr
   result.save("fitted_splats.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)

   # Visualize in viewer
   # Terminal: luxar serve fitted_splats.gsplats.zarr --viewer

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
       seed_method="gaussian",
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
       seed_method="uniform",
       # Result: 200 splats, PSNR 28dB, 500× compression
   )

**Choosing the Right Tradeoff**:

* **Interactive exploration**: Balanced (fast load, good quality)
* **Publication figures**: High quality (best reconstruction)
* **Network-limited**: High compression (fast transfer)

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

**Next Tutorial**: Performance optimization for billion-point datasets.
