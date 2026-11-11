Evaluate Luxar in ten minutes
=============================

This page is the shortest path through Luxar for someone trying it for the first time: see
it running, install it, build a scene, fit a volume and share the result. Every command
below runs as written on a fresh install.

See it running first
--------------------

Nothing needs to be installed to look at Luxar scenes:

* the gallery at https://demos.luxarviewer.dev holds the bundled demos as live scenes
  (light-sheet time-lapses, multichannel confocal volumes, single-molecule localisations,
  lineage tracking, and data from astronomy and the geosciences);
* the hosted viewer at https://luxarviewer.dev opens any publicly reachable
  ``.luxar.zarr`` archive named in its URL, ``https://luxarviewer.dev/?src=<archive URL>``.

Install
-------

The core package compiles and serves scenes; the ``gsplats`` extra adds Gaussian-splat
fitting (it installs PyTorch, a download of several gigabytes on first install):

.. code-block:: bash

   pip install luxar              # scenes, viewer, demos
   pip install "luxar[gsplats]"   # adds fitting volumes as Gaussian splats

Fitting runs on an NVIDIA GPU (CUDA), an Apple GPU (MPS) or, more slowly, the CPU.

Build a first scene
-------------------

A scene is described in Python and compiled once into a ``.luxar.zarr`` archive:

.. code-block:: python

   import numpy as np
   from luxar import LuxarZarrCompiler, Dimensions, Dimension

   dims = Dimensions([
       Dimension("x", unit="um", display=True),
       Dimension("y", unit="um", display=True),
       Dimension("z", unit="um", display=True),
   ])

   with LuxarZarrCompiler("my_data.luxar.zarr") as compiler:
       scene = compiler.create_scene(dimensions=dims)
       positions = np.random.randn(100_000, 3).astype(np.float32) * 50
       colors = np.random.rand(100_000, 3).astype(np.float32)
       scene.add_points("MyPoints", positions, colors=colors)

Then serve it with the viewer and open the printed link in a browser:

.. code-block:: bash

   luxar serve my_data.luxar.zarr --viewer

Run a bundled demo
------------------

``luxar demo`` lists the demos; ``luxar demo run <key>`` builds one (downloading its data
when needed) and opens it. ``luxar demo deps`` reports which optional packages a demo needs.

.. code-block:: bash

   luxar demo
   luxar demo run lorenz

Fit a volume as Gaussian splats
-------------------------------

Fitting turns an image volume (``.tiff``, ``.npy``, ``.npz``, ``.zarr`` or OME-Zarr) into
Gaussian splats; ``convert`` makes the fit a scene the viewer opens:

.. code-block:: bash

   luxar gsplat fit volume.tiff splats.gsplats.zarr --preset draft --seeds 2000
   luxar gsplat convert splats.gsplats.zarr scene.luxar.zarr --center --tone-mapping ACES
   luxar serve scene.luxar.zarr --viewer

A small volume fits in seconds on a GPU. To choose the number of splats for a real dataset,
``luxar gsplat cal`` runs the blind-spot cross-validation described in
:doc:`tutorials/gaussian_splatting`.

Share a scene
-------------

A compiled scene is a folder of static files: copy it to any static web host and share the
link ``https://luxarviewer.dev/?src=<URL of the folder>``. For an offline copy,
``luxar export scene.luxar.zarr -o my_export/`` writes the scene together with the viewer and
a small script that serves both locally (see :doc:`tutorials/distributing_scenes`).

Where to go next
----------------

* :doc:`tutorials/basic_scene` and the other tutorials;
* :doc:`guides/user/VIEWER_GUIDE` for the viewer's controls;
* :doc:`guides/user/CLI_REFERENCE` for every command;
* the source code and issue tracker at https://github.com/royerlab/luxar.
