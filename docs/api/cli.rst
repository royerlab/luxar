CLI Package
===========

The CLI package provides command-line tools for serving and inspecting Luxar datasets.

.. automodule:: luxar.cli
   :members:
   :undoc-members:
   :show-inheritance:

Main Application
----------------

The main application provides the ``luxar`` command-line interface with commands for
serving data, building the viewer, and inspecting datasets.

.. automodule:: luxar.cli.main
   :no-members:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: luxar.cli.main.create_server_app

   This function is particularly useful for:

   * **Integration Testing**: Create test servers that serve real Zarr data
   * **Programmatic Server Creation**: Embed Luxar server in larger applications
   * **Custom Deployments**: Configure and run servers with custom settings

   Example usage in tests::

      from luxar.cli.main import create_server_app
      import uvicorn

      # Create server app
      app = create_server_app("/path/to/data.luxar.zarr", serve_viewer=False)

      # Run with uvicorn
      uvicorn.run(app, host="127.0.0.1", port=8000)

Utilities
---------

Utility functions for port management, viewer building, and dataset inspection.

.. automodule:: luxar.cli.utils
   :members:
   :undoc-members:

Network Simulation
------------------

Network simulation middleware for testing viewer performance under various network conditions.

.. automodule:: luxar.cli.network_simulation
   :members:
   :undoc-members:

Network Profiles
~~~~~~~~~~~~~~~~

The network simulation supports several built-in profiles that simulate real-world network conditions:

* **3g**: Mobile 3G connection (384 kbps, 300ms latency, 1% packet loss)
* **4g**: Mobile 4G/LTE (10 Mbps, 100ms latency, 0.5% packet loss)
* **5g**: 5G mobile (100 Mbps, 30ms latency, 0.1% packet loss)
* **satellite**: Satellite internet (25 Mbps, 600ms latency, 1% packet loss)
* **rural**: Rural DSL (1 Mbps, 100ms latency, 2% packet loss)
* **congested**: Congested network (2 Mbps, 200ms latency, 3% packet loss)
* **slow-broadband**: Slow broadband (5 Mbps, 50ms latency, 0.5% packet loss)
* **broadband**: Home broadband (50 Mbps, 20ms latency, 0.1% packet loss)
* **fast-broadband**: Fast broadband (200 Mbps, 10ms latency, 0.05% packet loss)

Example usage::

   # Simulate 3G connection
   luxar serve data.luxar.zarr --profile 3g --viewer

   # Custom slow connection
   luxar serve data.luxar.zarr --bandwidth 500kbps --latency 200ms --packet-loss 2%

   # Override profile settings
   luxar serve data.luxar.zarr --profile 4g --latency 300ms

See the :doc:`../guides/developer/NETWORK_SIMULATION_SPEC` for detailed specifications.

Export
------

Export Luxar scenes as standalone offline viewer bundles.

.. automodule:: luxar.cli.export
   :members:
   :undoc-members:

Native Bundles
--------------

Producers for double-clickable native bundles wrapped around the Go-compiled
launcher binary (``packages/luxar-launcher``). Used internally by the
``luxar export --native`` subcommand to emit ``.app`` bundles on macOS and
portable folders on Linux. The bundle layout, ``Info.plist`` generator, and
launcher-binary lookup all live in this module.

.. automodule:: luxar.cli.native_app
   :members:
   :undoc-members:

GSplat Commands
---------------

CLI commands for fitting, converting, rendering, merging, and managing Gaussian splats.

.. automodule:: luxar.cli.gsplat_commands
   :members:
   :undoc-members:

GSplat Configuration
~~~~~~~~~~~~~~~~~~~~

Configuration loading and validation for GSplat CLI commands.

.. automodule:: luxar.cli.gsplat_config
   :members:
   :undoc-members:
