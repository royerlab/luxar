Gaussian Splatting Package
===========================

The gsplats package provides tools for fitting and rendering Gaussian splats to images.

.. automodule:: luxar.gsplats
   :members:
   :undoc-members:
   :show-inheritance:

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
   :members:
   :undoc-members:
   :show-inheritance:

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
   :members:
   :undoc-members:

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
   :members:
   :undoc-members:
   :show-inheritance:

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
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: luxar.gsplats.seeds.generate_seeds

.. autofunction:: luxar.gsplats.seeds.seed_from_decomposition

.. autofunction:: luxar.gsplats.seeds.seed_from_edges

.. autofunction:: luxar.gsplats.seeds.seed_from_grid

.. autofunction:: luxar.gsplats.seeds.seed_from_peaks

CLAHE Enhancement
-----------------

Contrast-Limited Adaptive Histogram Equalization for nD data.

.. automodule:: luxar.gsplats.clahe
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: luxar.gsplats.clahe.apply_clahe
