Luxar Documentation
===================

Welcome to Luxar's documentation! Luxar is a high-performance system for compiling and visualizing arbitrary-sized n-dimensional scenes containing points, lines, and Gaussian splats. Luxar delivers visualization performance limited only by your graphics card, display resolution, and network bandwidth—not by software constraints.

.. toctree::
   :maxdepth: 2
   :caption: Concepts & Architecture:

   concepts/architecture

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guides:

   guides/user/LUXAR_ZARR_FORMAT
   guides/user/HDR_GUIDE
   guides/user/E2E_TESTING_GUIDE

.. toctree::
   :maxdepth: 2
   :caption: Developer Guides:

   guides/developer/BUILD_SYSTEM_SPEC
   guides/developer/JSDOC_STYLE_GUIDE
   guides/developer/CONSOLE_OUTPUT_STYLE
   guides/developer/NETWORK_SIMULATION_SPEC
   guides/developer/PLAYWRIGHT_GUIDE
   guides/developer/TESTING_GUIDELINES
   guides/developer/ERROR_HANDLING_GUIDE
   guides/developer/PERFORMANCE_OPTIMIZATION_SPEC
   guides/developer/SCENE_UPDATE_OPTIMIZATION
   guides/developer/GSPLATS_VIEWER_IMPLEMENTATION
   guides/developer/METAL_SPLATTING_IMPLEMENTATION_SPEC
   guides/developer/SPLAT_MODEL_METAL

.. toctree::
   :maxdepth: 2
   :caption: Technical Specifications:

   guides/specs/CACHE_PREFETCHING_SPEC
   guides/specs/DIMENSION_INITIALIZATION_FIX
   guides/specs/LINES_SEGMENT_CHUNKING_BUG_FIX
   specs/GSPLATS_DIMENSION_MAPPING

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   api/core
   api/io
   api/encoding
   api/validation
   api/utils
   api/typing_utils
   api/cli
   api/gsplats

Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install luxar
   # or for development:
   git clone https://github.com/royerlab/luxar.git
   cd luxar
   hatch shell

Basic Usage
~~~~~~~~~~~

Create and visualize a scene with points:

.. code-block:: python

   from luxar.core import Dimensions
   from luxar.io import LuxarZarrCompiler
   import numpy as np

   # Create data
   positions = np.random.randn(1000, 3).astype(np.float32)
   colors = np.random.rand(1000, 3).astype(np.float32)
   dims = Dimensions.default_3d()

   # Write to zarr
   with LuxarZarrCompiler('scene.zarr') as compiler:
       scene = compiler.create_scene(dimensions=dims)
       scene.add_points('cloud', positions, colors, radii=0.1)

   # Serve with viewer
   # Terminal: luxar serve scene.zarr --viewer

Features
--------

* **nD Visualization**: Handle arbitrary-dimensional data with interactive slicing
* **Performance**: 100K-10M points at 60 FPS with WebGL rendering
* **Compression**: 4-40× data compression with lossy/lossless options
* **Streaming**: Memory-efficient lazy loading with intelligent caching
* **Spatial Indexing**: Morton/Hilbert ordering for efficient queries

Packages Overview
-----------------

Core Packages
~~~~~~~~~~~~~

* **luxar.core** - Scene graph, data structures, dimensions, transforms
* **luxar.io** - Read/write Zarr files with spatial indexing
* **luxar.encoding** - Array encoding with semantic types and quantization
* **luxar.validation** - Data validation with helpful error messages
* **luxar.utils** - Utilities and demo data generators
* **luxar.typing_utils** - Type definitions and constants
* **luxar.cli** - Command-line interface

Gaussian Splatting
~~~~~~~~~~~~~~~~~~

* **luxar.gsplats** - Fit and render Gaussian splats to images
* **luxar.gsplats.fitting** - Modular fitting pipeline
* **luxar.gsplats.optim** - Per-splat Adam optimizer
* **luxar.gsplats.models** - Rendering models
* **luxar.gsplats.io** - Save/load splat results
* **luxar.gsplats.utils** - Matrix utilities
* **luxar.gsplats.seeds** - Seed generation strategies
* **luxar.gsplats.clahe** - CLAHE-based sampling

Support
-------

* **Documentation**: https://royerlab.github.io/luxar
* **Issues**: https://github.com/royerlab/luxar/issues
* **Repository**: https://github.com/royerlab/luxar
