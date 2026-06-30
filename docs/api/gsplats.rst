Gaussian Splatting Package
===========================

The gsplats package provides tools for fitting and rendering Gaussian splats to images.

.. automodule:: luxar.gsplats
   :no-members:

Main API
--------

.. autofunction:: luxar.gsplats.fit_gaussian_splats

.. autoclass:: luxar.gsplats.GSplatData
   :members:
   :undoc-members:
   :show-inheritance:

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
   :undoc-members:

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
the ``luxar gsplat lod --recipe ...`` CLI command (recipes ``flat`` / ``additive``
/ ``partitioned`` / ``multiscale`` / ``mosaic`` / ``substitutive`` / ``pyramid``).

* **Additive** — same N splats, reordered into a prefix-monotone ladder
  (``make_additive_lod``). Loading the first k splats is the best L²
  approximation at that budget.
* **Substitutive** — synthesise M < N representative splats per coarser
  level via Gaussian mixture reduction (``make_substitutive_lod``).

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
