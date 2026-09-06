Luxar Documentation
===================

Welcome to Luxar's documentation! Luxar is a high-performance system for compiling and visualizing arbitrary-sized n-dimensional scenes containing points, lines, Gaussian splats, and triangle meshes. Luxar delivers visualization performance limited only by your graphics card, display resolution, and network bandwidth—not by software constraints.

.. admonition:: See it running before you install anything
   :class: tip

   `demos.luxarviewer.dev <https://demos.luxarviewer.dev>`_ hosts 85 of the
   bundled demos as live, interactive scenes — real compiled archives streamed
   from object storage, not videos.

   The viewer is also deployed on its own at
   `luxarviewer.dev <https://luxarviewer.dev>`_ and takes any reachable scene as
   a URL parameter, so you can point it at your own compiled archive::

      https://luxarviewer.dev/?src=https://example.org/path/to/scene.luxar.zarr

   The host serving that URL needs to allow cross-origin reads. See
   :doc:`guides/developer/DEMO_SITE_RUNBOOK` for how the demo corpus itself is
   hosted.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install luxar
   # or for development:
   git clone https://github.com/royerlab/luxar.git
   cd luxar
   make setup-dev
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
   with LuxarZarrCompiler('scene.luxar.zarr') as compiler:
       scene = compiler.create_scene(dimensions=dims)
       scene.add_points('cloud', positions, colors, radii=0.1)

   # Serve with viewer
   # Terminal: luxar serve scene.luxar.zarr --viewer

.. image:: images/docs/basic-3d-pointcloud.png
   :alt: Luxar viewer showing a 3D point cloud
   :width: 100%

Features
--------

* **nD Visualization**: Handle arbitrary-dimensional data with interactive slicing
* **Four Geometry Types**: Points (soft-edged spheres), Lines (width-tapered curves), Gaussian Splats (oriented Gaussians), and Mesh (shaded triangle surfaces)
* **Performance**: 100K-10M points at 60 FPS with WebGL rendering
* **Compression**: 4-40x data compression with lossy/lossless options
* **Streaming**: Memory-efficient lazy loading with intelligent caching
* **Spatial Indexing**: Morton/Hilbert ordering for efficient queries
* **Gaussian Splatting**: Fit splats to volumes for compression, denoising, and visualization

.. toctree::
   :maxdepth: 2
   :caption: Concepts & Architecture:

   concepts/architecture

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/basic_scene
   tutorials/nd_navigation
   tutorials/programmatic_server
   tutorials/gaussian_splatting
   tutorials/performance_optimization
   tutorials/distributing_scenes

.. toctree::
   :maxdepth: 2
   :caption: User Guides:

   guides/user/VIEWER_GUIDE
   guides/user/HDR_GUIDE

.. toctree::
   :maxdepth: 2
   :caption: Command-Line Interface:

   guides/user/CLI_REFERENCE

.. toctree::
   :maxdepth: 2
   :caption: Format Specifications:

   guides/user/LUXAR_ZARR_FORMAT
   specs/GSPLATS_ZARR_FORMAT
   specs/MESH_NODE_SPEC
   guides/user/FORMAT_AND_MIGRATION

.. toctree::
   :maxdepth: 2
   :caption: Technical Specifications:

   guides/specs/ND_TRANSFORMS_SPEC
   guides/specs/CACHE_PREFETCHING_SPEC
   guides/specs/GSPLAT_DEPTH_SORTING_SPEC
   guides/specs/VOLUMETRIC_BLENDING_SPEC
   guides/specs/SUBPIXEL_JITTER_TAA_SPEC
   guides/specs/LAYER_ORDER_SPEC
   specs/GSPLATS_DIMENSION_MAPPING
   specs/LUXAR_LAYER_SPEC

.. toctree::
   :maxdepth: 2
   :caption: Developer Guides:

   guides/developer/BUILD_SYSTEM_SPEC
   guides/developer/DOCUMENTATION_QUALITY
   guides/developer/TESTING_GUIDELINES
   guides/developer/PLAYWRIGHT_GUIDE
   guides/user/E2E_TESTING_GUIDE
   guides/developer/JSDOC_STYLE_GUIDE
   guides/developer/CONSOLE_OUTPUT_STYLE
   guides/developer/UI_DESIGN_GUIDE
   guides/developer/NETWORK_SIMULATION_SPEC
   guides/developer/ERROR_HANDLING_GUIDE
   guides/developer/DEBUG_INTERFACE_GUIDE
   guides/developer/INTENSITY_GAMMA_DESIGN
   guides/developer/DEMO_SITE_RUNBOOK
   guides/developer/BENCHMARK_FLOOR_DECISION
   guides/developer/VIEWER_PERFORMANCE_AUDIT_2026_09

.. toctree::
   :maxdepth: 3
   :caption: Python API Reference:

   api/core
   api/mesh
   api/shading
   api/io
   api/encoding
   api/validation
   api/demos
   api/utils
   api/typing_utils
   api/cli
   api/gsplats
   api/colormaps

.. toctree::
   :maxdepth: 1
   :caption: TypeScript API Reference:

   api/viewer_api

Packages Overview
-----------------

Core Packages
~~~~~~~~~~~~~

* **luxar.core** - Scene graph, data structures, dimensions, transforms
* **luxar.io** - Read/write Zarr files with spatial indexing
* **luxar.encoding** - Array encoding with semantic types and quantization
* **luxar.validation** - Data validation with helpful error messages
* **luxar.demos** - Dataset, download, and runtime helpers for executable demos
* **luxar.utils** - Utilities and demo data generators
* **luxar.typing_utils** - Type definitions and constants
* **luxar.cli** - Command-line interface

Geometry & Appearance
~~~~~~~~~~~~~~~~~~~~~

* **luxar.mesh** - Import, decimate, and split triangle meshes
* **luxar.mesh.interop** - Read PLY / OBJ / STL / VTP / glTF files
* **luxar.mesh.decimate** - Coarser surfaces for substitutive mesh LOD
* **luxar.mesh.split** - By-face re-indexing behind ``add_mesh(partition=…)``
* **luxar.shading** - Bake ambient occlusion into emissive geometry

Gaussian Splatting
~~~~~~~~~~~~~~~~~~

* **luxar.gsplats** - Fit and render Gaussian splats to volumes
* **luxar.gsplats.fitting** - Modular fitting pipeline
* **luxar.gsplats.optim** - Per-splat Adam optimizer
* **luxar.gsplats.models** - Rendering models
* **luxar.gsplats.io** - Save/load splat results
* **luxar.gsplats.utils** - Matrix utilities
* **luxar.gsplats.seeds** - Seed generation strategies
* **luxar.gsplats.clahe** - CLAHE-based sampling
* **luxar.gsplats.calibration** - Blind-spot cross-validation for splat count K
* **luxar.gsplats.lod** - Level-of-detail topology construction
* **luxar.gsplats.batch** - Batch fitting across GPUs / Slurm clusters

Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Support
-------

* **Documentation**: https://royerlab.github.io/luxar
* **Issues**: https://github.com/royerlab/luxar/issues
* **Repository**: https://github.com/royerlab/luxar
