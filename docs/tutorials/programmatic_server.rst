Programmatic Server Creation
=============================

This tutorial demonstrates how to create and manage Luxar data servers
programmatically using the ``create_server_app()`` function. You will learn
how to write integration tests against a real server, serve multiple datasets
from a single directory, and embed Luxar inside a larger web application.

What You Will Learn
-------------------

* How to create a Luxar server with ``create_server_app()``
* How to run integration tests against a real server using background threads
* How to serve multiple Zarr datasets from a parent directory
* How to mount Luxar as a sub-application inside a larger FastAPI app

Prerequisites
-------------

* A Luxar installation (``pip install luxar`` or ``hatch run pip install -e .``)
* A Zarr dataset (see :doc:`basic_scene` to create one)
* ``uvicorn`` and ``requests`` (included with Luxar dependencies)


Creating a Basic Server
-----------------------

The entry point is ``create_server_app()`` from ``luxar.cli.main``. It accepts
the path to a directory or Zarr dataset and returns a standard FastAPI
application with CORS headers and a health endpoint already configured.

.. code-block:: python

   from luxar.cli.main import create_server_app
   import uvicorn

   # Create a FastAPI app serving a single Zarr dataset
   app = create_server_app("/path/to/scene.luxar.zarr")

   # Start the server
   uvicorn.run(app, host="127.0.0.1", port=8000)

**Parameters:**

``path`` (str)
   Path to a directory or Zarr dataset to serve. If it points to a single
   ``.zarr`` directory, that dataset is served at the root. If it points to a
   regular directory, all contents (including multiple ``.zarr`` datasets) are
   served, and the root returns a JSON directory listing.

``cors_origin`` (str, keyword-only, default ``"local"``)
   Which browser origins may fetch data via CORS:

   * ``"local"`` (default) — allow only loopback origins
     (``localhost`` / ``127.0.0.1`` / ``[::1]`` on any port). This is the safe
     development default; a remote page cannot read the served data.
   * ``"*"`` — allow any origin (credentials are disabled in this mode).
   * a comma-separated list of explicit origins, e.g.
     ``"https://viewer.example.com"``.

The returned app includes:

* **CORS middleware** configured from ``cors_origin`` (loopback-only by
  default; pass ``cors_origin="*"`` to allow any origin).
* A ``/health`` endpoint that returns ``{"status": "ok"}``.
* A static file mount at ``/`` backed by ``DirectoryListingStaticFiles``.


Integration Testing
-------------------

Because ``create_server_app()`` returns a real FastAPI application, you can
spin up a real HTTP server in a background thread and write integration tests
that exercise the full network stack -- no mocking required.

The pattern below mirrors the approach used in the Luxar test suite itself
(see ``packages/luxar/src/luxar/cli/tests/test_cli_integration.py``).

.. code-block:: python

   import threading
   import time
   import requests
   import uvicorn
   from luxar.cli.main import create_server_app


   def run_server(app, host, port, started_event):
       """Run uvicorn in a background thread, signalling when ready."""
       config = uvicorn.Config(app, host=host, port=port, log_level="warning")
       server = uvicorn.Server(config)

       # Notify the main thread once the server is accepting connections
       original_startup = server.startup

       async def _startup_wrapper(*args, **kwargs):
           await original_startup(*args, **kwargs)
           started_event.set()

       server.startup = _startup_wrapper
       server.run()


   def wait_for_server(base_url, timeout=10):
       """Poll the /health endpoint until the server is ready."""
       deadline = time.time() + timeout
       while time.time() < deadline:
           try:
               resp = requests.get(f"{base_url}/health", timeout=1)
               if resp.status_code == 200:
                   return
           except requests.ConnectionError:
               pass
           time.sleep(0.1)
       raise TimeoutError(f"Server at {base_url} did not start within {timeout}s")


   def test_server_health(sample_zarr_path):
       """Verify that the server starts and reports healthy."""
       app = create_server_app(str(sample_zarr_path))
       host, port = "127.0.0.1", 9123
       base_url = f"http://{host}:{port}"

       started = threading.Event()
       thread = threading.Thread(
           target=run_server, args=(app, host, port, started), daemon=True
       )
       thread.start()
       wait_for_server(base_url)

       # Health check
       resp = requests.get(f"{base_url}/health")
       assert resp.status_code == 200
       assert resp.json() == {"status": "ok"}

       # CORS headers should be present for an allowed (loopback) origin.
       # The default cors_origin="local" allows localhost/127.0.0.1/[::1] on
       # any port, so a same-machine viewer can read the data.
       resp = requests.options(
           f"{base_url}/health",
           headers={
               "Origin": "http://localhost:5173",
               "Access-Control-Request-Method": "GET",
           },
       )
       assert "access-control-allow-origin" in resp.headers

       # To allow any browser origin, create the app with cors_origin="*":
       #   app = create_server_app(path, cors_origin="*")

       # Zarr metadata is accessible
       resp = requests.get(f"{base_url}/.zattrs")
       assert resp.status_code == 200
       metadata = resp.json()
       assert "luxar" in metadata  # Luxar scenes store config under this key


   def test_scene_metadata(sample_zarr_path):
       """Check that scene-level metadata is valid."""
       app = create_server_app(str(sample_zarr_path))
       host, port = "127.0.0.1", 9124
       base_url = f"http://{host}:{port}"

       started = threading.Event()
       thread = threading.Thread(
           target=run_server, args=(app, host, port, started), daemon=True
       )
       thread.start()
       wait_for_server(base_url)

       resp = requests.get(f"{base_url}/.zattrs")
       attrs = resp.json()

       # Verify scene-level keys
       luxar_meta = attrs["luxar"]
       assert "scene" in luxar_meta
       assert "children" in luxar_meta["scene"]

**Key points:**

* The server runs in a **daemon thread** so it is automatically cleaned up
  when the test process exits.
* ``wait_for_server()`` polls ``/health`` in a retry loop to avoid race
  conditions between server startup and the first test request.
* Tests use the standard ``requests`` library, keeping assertions simple and
  readable.
* Each test should use a **unique port** to avoid conflicts when tests run in
  parallel.


Serving Multiple Datasets
--------------------------

If you point ``create_server_app()`` at a regular directory that contains
several ``.zarr`` datasets, all of them become accessible under their
respective names. The root path returns a JSON directory listing.

.. code-block:: python

   import requests
   from luxar.cli.main import create_server_app

   # Assume /data/ contains:
   #   /data/neurons.luxar.zarr/
   #   /data/vasculature.luxar.zarr/
   app = create_server_app("/data")

   # After starting the server on port 8000 (see above for the threading
   # pattern), the root returns a listing:
   resp = requests.get("http://127.0.0.1:8000/")
   listing = resp.json()
   # listing == {
   #     "entries": [
   #         {"name": "neurons.luxar.zarr", "type": "zarr", "size": 4096},
   #         {"name": "vasculature.luxar.zarr", "type": "zarr", "size": 8192},
   #     ]
   # }

   # Each dataset is served at its own subpath:
   resp = requests.get("http://127.0.0.1:8000/neurons.luxar.zarr/.zattrs")
   assert resp.status_code == 200

   resp = requests.get("http://127.0.0.1:8000/vasculature.luxar.zarr/.zattrs")
   assert resp.status_code == 200

The directory listing JSON structure uses three possible values for the
``type`` field: ``"file"``, ``"directory"``, or ``"zarr"``. The ``size``
field is reported in bytes.

This approach is the simplest way to serve multiple datasets. It requires
no extra configuration -- just place all your ``.zarr`` directories inside a
common parent and pass that parent to ``create_server_app()``.

.. note::

   When connecting the Luxar viewer to a dataset served this way, prefer the
   canonical URL without a trailing slash, for example
   ``http://127.0.0.1:8000/neurons.luxar.zarr``. The viewer accepts the form
   ending in ``/`` too, but normalizing examples and logs to one spelling
   avoids duplicate-looking URLs.


Embedding in a Larger Application
----------------------------------

Because ``create_server_app()`` returns a standard FastAPI instance, you can
mount it as a sub-application inside a larger web service. This is useful
when you want to combine Luxar data serving with your own API endpoints,
authentication middleware, or other services.

.. code-block:: python

   from fastapi import FastAPI
   from luxar.cli.main import create_server_app

   # Your main application
   main_app = FastAPI(title="My Research Platform")


   @main_app.get("/api/experiments")
   async def list_experiments():
       return {"experiments": ["exp_001", "exp_002"]}


   # Mount Luxar as a sub-application
   luxar_app = create_server_app("/data/scenes")
   main_app.mount("/data", luxar_app)

   # After starting main_app:
   # - GET /api/experiments      -> your custom endpoint
   # - GET /data/health          -> Luxar health check
   # - GET /data/.zattrs         -> Zarr metadata
   # - GET /data/neurons.luxar.zarr/   -> dataset files (if /data/scenes/ is a directory)
   # - GET /data/                -> JSON directory listing

.. note::

   ``create_server_app()`` serves data only -- it does not bundle the browser
   viewer. To serve the interactive Luxar viewer alongside a dataset, use the
   CLI instead: ``luxar serve <data.luxar.zarr> --viewer`` (which spins up the
   viewer static files via the internal ``_serve_viewer`` helper).

You can also mount multiple independent Luxar apps at different paths:

.. code-block:: python

   from fastapi import FastAPI
   from luxar.cli.main import create_server_app

   main_app = FastAPI()

   # Each dataset gets its own isolated server
   main_app.mount("/neurons", create_server_app("/data/neurons.luxar.zarr"))
   main_app.mount("/vessels", create_server_app("/data/vasculature.luxar.zarr"))

   # GET /neurons/health   -> {"status": "ok"}
   # GET /neurons/.zattrs  -> neuron scene metadata
   # GET /vessels/health   -> {"status": "ok"}
   # GET /vessels/.zattrs  -> vasculature scene metadata

This second pattern gives you fine-grained control over which datasets are
exposed at which paths, and lets you apply different middleware or
authentication to each mount point if needed.


Next Steps
----------

* :doc:`basic_scene` -- Learn how to create Zarr scenes to serve
* :doc:`nd_navigation` -- Explore multi-dimensional datasets in the viewer
* :doc:`../api/cli` -- Full API reference for the CLI module
* :doc:`../guides/developer/NETWORK_SIMULATION_SPEC` -- Simulate network
  conditions for performance testing
