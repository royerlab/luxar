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
   :members:
   :undoc-members:

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
      app = create_server_app("/path/to/data.zarr", serve_viewer=False)

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

* **3g**: Mobile 3G connection (400 kbps, 200ms latency, 1% packet loss)
* **4g**: Mobile 4G/LTE (10 Mbps, 50ms latency)
* **5g**: 5G mobile (100 Mbps, 10ms latency)
* **broadband**: Home broadband (50 Mbps, 20ms latency)
* **satellite**: Satellite internet (25 Mbps, 600ms latency, 0.5% packet loss)
* **rural**: Rural DSL (2 Mbps, 100ms latency)
* **congested**: Congested network (1 Mbps, 300ms latency, 5% packet loss)

Example usage::

   # Simulate 3G connection
   luxar serve data.zarr --profile 3g --viewer

   # Custom slow connection
   luxar serve data.zarr --bandwidth 500kbps --latency 200ms --packet-loss 2%

   # Override profile settings
   luxar serve data.zarr --profile 4g --latency 300ms

See the :doc:`../guides/developer/NETWORK_SIMULATION_SPEC` for detailed specifications.
