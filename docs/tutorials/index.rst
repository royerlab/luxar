Tutorials
=========

Step-by-step guides for common Luxar workflows.

.. toctree::
   :maxdepth: 2

   basic_scene
   nd_navigation
   gaussian_splatting
   performance_optimization

Overview
--------

These tutorials demonstrate real-world usage of Luxar, explaining not just *how* to use the API, but *why* certain approaches work and *when* to use different features.

**What You'll Learn**:

* Creating scenes with proper dimensionality
* Efficient data encoding and compression
* nD navigation and hypersphere slicing
* Gaussian splat fitting and rendering
* Performance optimization techniques

**Prerequisites**:

* Python 3.10+
* NumPy basics
* (Optional) PyTorch for Gaussian splatting

Tutorial Index
--------------

1. **Basic Scene Creation** - Your first Luxar visualization

   * Understanding scene graphs
   * Adding points with attributes
   * Spatial ordering benefits
   * Viewing in the browser

2. **nD Navigation** - Working with multi-dimensional data

   * Defining dimensions with metadata
   * Discrete vs continuous dimensions
   * Hypersphere visibility concept
   * Keyboard navigation in viewer

3. **Gaussian Splatting** - Fitting splats to images

   * When to use Gaussian splats
   * Choosing seed generation methods
   * Optimizing fitting parameters
   * Analyzing results

4. **Performance Optimization** - Handling billion-point datasets

   * Choosing chunk sizes
   * Spatial ordering strategies
   * Encoding mode selection
   * Prefetching configuration
