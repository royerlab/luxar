Programmatic Server Creation
=============================

This tutorial demonstrates how to create and manage Luxar servers programmatically,
which is useful for integration testing, custom deployments, and embedding Luxar
in larger applications.

Overview
--------

The ``luxar.cli.main.create_server_app()`` function creates a configured FastAPI
application that serves Zarr datasets. This allows you to:

* Create test servers for integration testing
* Embed Luxar servers in larger applications
* Customize server configuration programmatically
* Run multiple servers with different datasets

Basic Usage
-----------

Creating a Simple Server
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from luxar.cli.main import create_server_app
   import uvicorn
   
   # Create FastAPI app
   app = create_server_app("/path/to/data.zarr", serve_viewer=False)
   
   # Run with uvicorn
   uvicorn.run(app, host="127.0.0.1", port=8000)

The server will serve the Zarr dataset at the root path with CORS enabled.

Integration Testing
-------------------

The function is designed for integration testing without mocking internal server logic.

Example::

   from luxar.cli.main import create_server_app
   
   # Create test server
   app = create_server_app(str(sample_scene_path), serve_viewer=False)
   
   # Use in tests with uvicorn

See ``packages/luxar/src/luxar/cli/tests/test_cli_integration.py`` for complete examples.

Next Steps
----------

* Read the :doc:`../guides/developer/NETWORK_SIMULATION_SPEC` for network testing
* Explore :doc:`../api/cli` for full API reference
* Check ``test_cli_integration.py`` for real-world examples
