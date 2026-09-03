Distributing Scenes
===================

This tutorial walks through the ways Luxar gets a finished scene in front of
someone else: a **standalone folder** anyone with Python 3 can serve, a
**double-clickable native bundle** with no Python dependency at all, and —
when you would rather send a link than a folder — **hosting the archive** and
opening it in the deployed viewer. We also cover how to share bundles across
machines and operating systems without tripping macOS Gatekeeper.

What You Will Learn
-------------------

* How to export a Luxar scene as a self-contained folder anyone with
  Python 3 can run
* How to produce double-clickable native bundles for macOS (`.app`)
  and Linux (portable folder)
* How to share bundles across machines without tripping macOS Gatekeeper
* How the embedded launcher's lifecycle works (close window → graceful
  server shutdown)
* When to use which option, including hosting the archive and sharing a viewer link

Prerequisites
-------------

* A Luxar installation with the viewer built (``make build-viewer``)
* For native bundles only: the launcher binaries built locally
  (``make install-go && make build-launchers``) — see
  :doc:`../guides/developer/BUILD_SYSTEM_SPEC` for the full toolchain
  setup
* A Zarr scene to export (see :doc:`basic_scene` to create one)


Folder export (zero deps for the recipient)
-------------------------------------------

The default ``luxar export`` produces a self-contained folder with the
viewer, the zarr dataset, a tiny stdlib-only ``serve.py`` script, and
a ``README.txt``:

.. code-block:: bash

   luxar export my_scene.luxar.zarr -o my_export/

Output:

.. code-block:: text

   my_export/
     viewer/        # Luxar viewer (HTML, JS, CSS, WASM)
     data/          # Zarr dataset (copied as-is)
     serve.py       # Local HTTP server (Python 3 stdlib only)
     README.txt     # Quick-start instructions

The recipient runs:

.. code-block:: bash

   cd my_export/
   python serve.py

That starts a local server on a free port and opens the viewer in their
default browser. The only dependency is **any Python 3** — no
``pip install``, no Node, no Rust. Add ``--open`` to launch a browser
on your end immediately:

.. code-block:: bash

   luxar export my_scene.luxar.zarr -o my_export/ --open


Native bundles (no Python dependency at all)
--------------------------------------------

For a recipient who shouldn't have to install or even know about
Python, use ``--native``. The output is a double-clickable bundle
wrapping a Go-compiled launcher around the viewer + data; the launcher
opens an embedded native WebView (WKWebView on macOS, WebKitGTK on
Linux) and serves the bundled zarr internally:

.. code-block:: bash

   # macOS .app
   luxar export my_scene.luxar.zarr -o out/ --native macos

   # Linux portable folder for x86_64
   luxar export my_scene.luxar.zarr -o out/ --native linux-amd64

   # All three platforms at once, with a custom name
   luxar export my_scene.luxar.zarr -o out/ \
       --native macos,linux-amd64,linux-arm64 \
       --name MyScene

**Output structure (macOS)**:

.. code-block:: text

   out/
     MyScene-README.txt           # Sibling README (xattr -cr recovery, etc.)
     MyScene.app/
       Contents/
         Info.plist               # Bundle metadata + icon reference
         MacOS/
           launcher               # Universal Mach-O (arm64 + amd64), +x
         Resources/
           AppIcon.icns           # App icon
           viewer/                # Luxar viewer
           data/                  # Zarr dataset

**Output structure (Linux)**:

.. code-block:: text

   out/
     MyScene-linux-amd64/
       luxar-launcher             # ELF, +x (needs webkit2gtk-4.0 runtime)
       viewer/                    # Luxar viewer
       data/                      # Zarr dataset
       MyScene.png                # Icon (FreeDesktop convention)
       README.txt                 # Quick start + libwebkit2gtk dep note

**Prerequisite**: ``make build-launchers`` must have been run on a host
of the matching OS before ``luxar export --native ...`` can produce
that platform's binary. CGO blocks pure cross-compilation, so a macOS
host cannot produce Linux binaries (and vice-versa) without a CGO
cross-toolchain. In practice you'd build each OS in CI.


Sharing bundles across machines
-------------------------------

The bundles are vanilla files — zip them and send them. There's a
quirk on macOS though: most "Internet-y" delivery channels mark the
download with the ``com.apple.quarantine`` extended attribute, which
Gatekeeper interprets as "this app came from outside; refuse to launch
unsigned code". The recipient sees:

  *"MyScene.app cannot be opened because the developer cannot be
  verified"*

…or, on older flows:

  *"MyScene.app is damaged and can't be opened. You should move it to
  the Trash"*

**Channels that DO set quarantine** (recipient hits Gatekeeper):

* Web download via any browser
* Email attachments (Mail, Outlook, Gmail web…)
* Slack / Discord / Teams / Messages
* AirDrop *(yes — modern macOS sets quarantine on AirDrop now)*
* GitHub Releases / Dropbox public link / S3 download

**Channels that DO NOT set quarantine** (recipient just double-clicks):

* ``scp`` / ``rsync`` / ``sftp`` over SSH
* Shared NAS / SMB / NFS mounts
* iCloud Drive sync between two Macs in the same Apple ID
* USB stick formatted exFAT or APFS
* ``curl`` / ``wget`` from a terminal
* Local builds (which is why your own freshly-built ``.app`` runs without
  complaint)

**Recovery for any quarantined bundle** — copy this snippet into the
recipient's Terminal:

.. code-block:: bash

   xattr -cr ~/Downloads/MyScene.app
   open ~/Downloads/MyScene.app

That strips the quarantine attribute. One-time fix per copy of the app.
The bundle generator drops a sibling ``MyScene-README.txt`` next to
every ``.app`` with this exact recipe so you don't have to re-explain
it each time.


Window lifecycle
----------------

The native launcher's design philosophy is "close the window → close
everything":

* Click the window's red close button → the embedded WebView's main
  loop exits → deferred ``Destroy()`` runs → deferred HTTP server
  ``Shutdown()`` runs → the launcher process exits cleanly. No zombie
  WebKit child processes.
* Press **Cmd+Q** (macOS) or send the launcher **SIGINT** / **SIGTERM**
  → identical clean-shutdown path. The ``LSUIElement`` flag is
  intentionally not set, so the app shows in the Dock and Cmd+Tab
  switcher.
* The launcher binds an ephemeral localhost port (no port collision),
  so multiple bundles can run side-by-side without interfering.

**Browser fallback** (headless smoke tests, or developers who want to
inspect with browser devtools): set the ``LUXAR_LAUNCHER_NO_WEBVIEW``
environment variable:

.. code-block:: bash

   LUXAR_LAUNCHER_NO_WEBVIEW=1 ./luxar-launcher

The launcher then opens the system default browser instead of an
embedded WebView; the local HTTP server still runs, you press
**Ctrl+C** to stop. This does *not* let the prebuilt Linux binary run
without ``libwebkit2gtk`` — WebKit is linked at build time, so the loader
aborts before the launcher can read this variable on a system missing the
``webkit2gtk-4.0`` runtime.


Looking ahead: signing and notarization
---------------------------------------

The ``xattr -cr`` workaround is fine for lab-internal sharing but not
for public distribution. The proper fix is an Apple Developer ID
signature (~$99/yr) plus the ``xcrun notarytool`` notarization step;
once notarized, macOS Gatekeeper accepts the bundle from any download
channel without warnings. The same idea applies on Windows
(Authenticode certificates) for when we add Windows support. None of
this changes the on-disk bundle format — signing and notarization
attach metadata to the existing ``.app``, they don't restructure it.

For now, ``xattr -cr`` covers all internal use cases. If your scene
needs to ship to people outside the lab, plan for the Developer ID
signing step.


When to use which
-----------------

Use the **folder export** when:

* The recipient has Python 3 (true on virtually any developer machine,
  any HPC node, any Linux distro)
* You need the smallest possible artifact (the launcher binary adds
  ~6–11 MB; the folder is just viewer + data)
* You want the recipient to be able to inspect / re-host with their
  own tooling (the ``serve.py`` is plain stdlib HTTP)

Use **native bundles** when:

* The recipient should never see a terminal
* You're sharing with non-developers (collaborators, presentation
  audiences, paper reviewers)
* You want a "real app" experience: Dock icon, Cmd+Q, native window
  controls


Sharing a link instead: the hosted viewer
------------------------------------------

Both options above ship the viewer *alongside* the data. There is a third
route that ships neither: put the compiled archive on any web host and hand
someone a URL into the deployed viewer at
`luxarviewer.dev <https://luxarviewer.dev>`_.

.. code-block:: text

   https://luxarviewer.dev/?src=https://example.org/data/scene.luxar.zarr

The viewer is a static build parameterised entirely by ``?src=``, so nothing
needs to be deployed per scene — the same viewer opens any archive it can
reach. The public demo gallery at
`demos.luxarviewer.dev <https://demos.luxarviewer.dev>`_ works exactly this
way: 86 archives on object storage, one viewer.

Requirements on the host serving the data depend on the store shape:

* **Directory ``.luxar.zarr`` stores** need CORS: the host must send
  ``Access-Control-Allow-Origin``. Their metadata and chunks use simple GETs,
  so byte-range support is not required.
* **Zipped ``.zarr.zip`` stores** additionally need byte-range support. The
  host must honour ``Range``, allow the ``Range`` request header in CORS, and
  expose ``Content-Range``, ``Content-Length``, ``Accept-Ranges``, and ``ETag``.
  The range headers validate partial responses; ``ETag`` preserves archive
  identity across cross-origin cache validation.

A plain static file host with CORS enabled is sufficient for directory stores.
Use this when the recipient just needs to *look* at the scene and you would
rather send a link than a multi-gigabyte folder — and note the data stays
wherever you put it, so the link is only as durable, and as private, as that
host.

:doc:`../guides/developer/DEMO_SITE_RUNBOOK` documents how the demo corpus is
hosted this way, including the CORS configuration and its failure modes.
