Gaussian Splatting Package
===========================

The gsplats package provides tools for fitting and rendering Gaussian splats to images.

.. automodule:: luxar.gsplats
   :no-members:

Main API
--------

.. autofunction:: luxar.gsplats.fit_gaussian_splats

.. autofunction:: luxar.gsplats.fit_progressive_gaussian_splats

.. autoclass:: luxar.gsplats.GSplatData
   :members:
   :inherited-members:
   :undoc-members:
   :exclude-members: additive_sublods, centers, amplitudes, cholesky_factors, colors, stats
   :show-inheritance:

.. autoproperty:: luxar.gsplats.GSplatData.additive_sublods
   :no-index:

Tiled Fitting
-------------

Fit large volumes tile-by-tile with cosine (Hann) apodization for seamless
stitching. Used by ``luxar gsplat fit --tiling uniform/content``.

.. autofunction:: luxar.gsplats.fit_tiled_gaussian_splats

.. autofunction:: luxar.gsplats.fit_tile

.. autofunction:: luxar.gsplats.fit_tiled

.. automodule:: luxar.gsplats.tiling
   :members:
   :no-undoc-members:
   :show-inheritance:

Lifting Points and Lines to Splats
----------------------------------

Convert existing Points/Lines geometry into Gaussian splats.

.. autofunction:: luxar.gsplats.lift_points_to_gsplats

.. autofunction:: luxar.gsplats.lift_lines_to_gsplats

.. automodule:: luxar.gsplats.lift
   :no-members:

Fitting Pipeline
----------------

The modular fitting pipeline for Gaussian splat optimization.

.. automodule:: luxar.gsplats.fitting
   :members:
   :undoc-members:
   :show-inheritance:

Fitting Configuration
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: luxar.gsplats.fitting.config
   :members:
   :undoc-members:
   :no-index:

Fitting Stages
~~~~~~~~~~~~~~

.. automodule:: luxar.gsplats.fitting.validation
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.fitting.preprocessing
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.fitting.initialization
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.fitting.losses
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.fitting.optimization
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.fitting.results
   :members:
   :undoc-members:

Dynamic Operations
~~~~~~~~~~~~~~~~~~

.. automodule:: luxar.gsplats.fitting.dynamic_ops
   :members:
   :undoc-members:

Optimization
------------

Per-splat Adam optimizer with gradient dilution compensation.

.. automodule:: luxar.gsplats.optim
   :no-members:

.. autofunction:: luxar.gsplats.optim.create_optimizer_and_scheduler

Models
------

Rendering models for 2D and 3D Gaussian splats.

.. automodule:: luxar.gsplats.models
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Splat Models
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: luxar.gsplats.models.gsplats
   :no-members:

.. autoclass:: luxar.gsplats.models.gsplats.GaussianSplatModel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Model Utilities
~~~~~~~~~~~~~~~

.. automodule:: luxar.gsplats.models.utils
   :members:
   :undoc-members:

Multiscale Decomposition
-------------------------

Hierarchical multiscale Gaussian splat decomposition.

.. automodule:: luxar.gsplats.multiscale
   :members:
   :undoc-members:
   :show-inheritance:

I/O Operations
--------------

Save and load Gaussian splat results.

.. automodule:: luxar.gsplats.io
   :no-members:

.. autofunction:: luxar.gsplats.io.save_gsplats

.. autofunction:: luxar.gsplats.io.load_gsplats

Utilities
---------

Matrix packing/unpacking utilities for covariance matrices.

.. automodule:: luxar.gsplats.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: luxar.gsplats.utils.trils
   :members:
   :undoc-members:

Seed Generation
---------------

Strategies for generating initial seed points for splat fitting.

.. automodule:: luxar.gsplats.seeds
   :no-members:

.. autofunction:: luxar.gsplats.seeds.generate_seeds

.. autofunction:: luxar.gsplats.seeds.seed_from_decomposition

.. autofunction:: luxar.gsplats.seeds.seed_from_edges

.. autofunction:: luxar.gsplats.seeds.seed_from_grid

.. autofunction:: luxar.gsplats.seeds.seed_from_peaks

CLAHE Enhancement
-----------------

Contrast-Limited Adaptive Histogram Equalization for nD data.

.. automodule:: luxar.gsplats.clahe
   :no-members:

.. autofunction:: luxar.gsplats.clahe.apply_clahe

Batch Fitting
-------------

Scheduler-agnostic batch orchestration for fitting a whole nD dataset across its
axes — locally across multiple GPUs (``batch-fit run``) or on a Slurm cluster
(``batch-fit submit``).

.. automodule:: luxar.gsplats.batch
   :members:
   :undoc-members:
   :show-inheritance:

Culling
-------

Gaussian splat culling strategies for reducing splat count while preserving quality.

.. automodule:: luxar.gsplats.culling
   :members:
   :no-undoc-members:

Quality Metrics
---------------

PSNR, SSIM, and MSE metrics for evaluating reconstruction quality.

.. automodule:: luxar.gsplats.metrics
   :members:
   :undoc-members:

Calibration (Blind-Spot Cross-Validation)
-----------------------------------------

Noise2Self model selection for Gaussian splat fits: sweep splat count
``K``, fit each at against a 5%-donut-median-filled volume, and report
the held-out PSNR peak (``K*``) plus the dataset's noise-floor PSNR
ceiling. Used by the ``luxar gsplat cal`` CLI command.

.. automodule:: luxar.gsplats.calibration
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: luxar.gsplats.calibration_report
   :members:
   :undoc-members:

Level of Detail (LOD)
---------------------

Post-fit LOD construction for streaming and view-dependent rendering. Used by
the ``luxar gsplat lod --recipe ...`` CLI command (recipes ``flat`` / ``stream``
/ ``levels`` / ``tiles`` / ``overview`` / ``adaptive``).

* **stream** — same N splats, reordered into a prefix-monotone additive ladder
  (``make_additive_lod``). Loading the first k splats is the best L²
  approximation at that budget.
* **levels** — synthesise M < N representative splats per coarser
  level via Gaussian mixture reduction (``make_substitutive_lod``).
* **tiles** / **overview** / **adaptive** — spatial-partition topologies for
  large datasets (per-tile streaming ladders, an optional coarse overview cap,
  or per-tile level swaps).

.. automodule:: luxar.gsplats.lod
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: luxar.gsplats.lod.additive
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.lod.substitutive
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.lod.recipes
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.lod.annotate
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.lod.quality
   :members:
   :undoc-members:

.. automodule:: luxar.gsplats.lod.volume_refit
   :members:
   :undoc-members:

Node Tree
---------

The in-memory gsplat node tree (leaf / lod / partition nodes) shared by the
fitting, LOD, and I/O layers — the v3.3 ``.gsplats.zarr`` on-disk structure.

.. automodule:: luxar.gsplats.tree
   :members:
   :no-undoc-members:
   :show-inheritance:

Content Planning
----------------

Density-driven box planning for content-adaptive tiled fits
(``luxar gsplat fit --tiling content``).

.. automodule:: luxar.gsplats.planner
   :members:
   :undoc-members:
   :show-inheritance:

Scene Interop
-------------

Bridge fitted gsplats into Luxar scenes (``add_gsplats_from_file`` and related
conversion helpers).

.. automodule:: luxar.gsplats.interop
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
-------------

Volume preprocessing shared by fitting and calibration (background-floor
suppression, normalization, denoising).

.. automodule:: luxar.gsplats.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

GPU Profiling
-------------

GPU memory and performance profiling for automatic tile-size selection.

.. automodule:: luxar.gsplats.gpu_profile
   :members:
   :undoc-members:

Volume Rendering
----------------

Render Gaussian splats back to volume arrays for quality comparison.

.. automodule:: luxar.gsplats.rendering
   :members:
   :undoc-members:
   :show-inheritance:
