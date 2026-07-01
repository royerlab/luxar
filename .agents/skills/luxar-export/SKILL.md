---
name: luxar-export
description: >-
  Package a Luxar scene for sharing — as a standalone offline folder (viewer +
  data + a zero-dependency serve.py) or as a native desktop app bundle (macOS
  .app, Linux portable folder). Use when a user wants to ship, distribute, or
  hand off a .luxar.zarr scene so others can open it without installing Luxar,
  or wants an embedded-WebView native launcher. Covers luxar export, --native
  bundles, the make build-launchers prerequisite, and the no-webview override.
---

# Export & share a Luxar scene

`luxar export` turns a built `.luxar.zarr` scene into a self-contained artifact that
runs without a Luxar install. Two output kinds:

1. **Standalone offline folder** (default) — viewer bundle + data + a Python-stdlib
   `serve.py`. Recipient runs `python serve.py`.
2. **Native app bundle** (`--native`) — a macOS `.app` (embedded WebView) or a Linux
   portable folder, each with the launcher binary, viewer, and data inside.

## Standalone offline folder (default)

```bash
luxar export scene.luxar.zarr -o my_export/                # build the folder
luxar export scene.luxar.zarr -o my_export/ --open         # build, then serve + open browser
luxar export scene.luxar.zarr -o my_export/ --overwrite    # replace existing output
luxar export scene.luxar.zarr -o my_export/ --open -p 9000 # custom serve port
```

The folder contains:
```
my_export/
├── viewer/        # viewer bundle (index.html, assets/, wasm/) with paths rewritten relative
├── data/          # the zarr dataset, copied as-is
├── serve.py       # zero-dependency Python 3 HTTP server (CORS, auto-port from 8000)
└── README.txt
```
Recipient just needs Python 3: `python serve.py` (`--port`, `--no-open` available). No
Luxar, no Node, no pip.

## Native app bundles (`--native`)

```bash
luxar export scene.luxar.zarr -o out/ --native macos                        # macOS .app (+ .app.zip)
luxar export scene.luxar.zarr -o out/ --native macos,linux-amd64,linux-arm64 --name MyScene
luxar export scene.luxar.zarr -o out/ --native macos --no-zip               # skip the .app.zip
```

Platform values: **`macos`**, **`linux-amd64`**, **`linux-arm64`** (comma-separated).
`--name` sets the bundle name (defaults to the zarr stem). `--zip/--no-zip` (default on
for macOS) also emits `<name>.app.zip` via `ditto` so a single file carries the `.app`
with permissions/resource forks intact.

Output per platform:
- macOS → `<name>.app` (`Contents/MacOS/launcher`, `Resources/viewer/`, `Resources/data/`,
  `Info.plist`; bundle id `org.czbiohub.luxar.<slug>`) + sibling `<name>.app.zip`.
- Linux → `<name>-linux-<arch>/` folder with `luxar-launcher`, `viewer/`, `data/`, `README.txt`.

### Prerequisite: build the launcher binaries FIRST

`--native` needs the host-platform Go launcher binary. If it's missing the bundler
raises a clear "run make build-launchers" error. Build it once:

```bash
make install-go        # no sudo: brew (macOS) / official tarball (Linux)
make build-launchers   # builds the host-platform binary into cli/_launchers/
```

CGO blocks pure cross-compilation, so **each platform's binary must be built on that
platform** (macOS produces a universal arm64+amd64 binary via `lipo`; Linux builds the
host arch and needs `libwebkit2gtk-4.1-dev`). In CI, each runner builds its own; the
binaries also ride along into wheel builds when present. So locally you can typically
only produce the `--native` bundle for the OS you're on.

### Runtime: headless / no-WebView

```bash
LUXAR_LAUNCHER_NO_WEBVIEW=1 ./luxar-launcher
```
Opens the system default browser instead of the embedded WebView — useful for headless
smoke tests and minimal Linux installs without `libwebkit2gtk`.

## Export vs. serve

`luxar export` produces a *portable artifact*. To just view a scene interactively on
your own machine (no packaging), use `luxar serve` instead:

```bash
luxar serve scene.luxar.zarr --viewer --open
#   --host (127.0.0.1) --port/-p (8000) --viewer-port (5173) --viewer-only
#   --cors-origin (local|*|<origins>)
#   network sim: --profile --bandwidth --latency --jitter --packet-loss
```

## Notes

- Build the scene first with the **`luxar-visualization`** skill (or `gsplat convert`).
- `--open` on export serves the freshly exported folder; without it, export just writes
  the folder and exits.
- Never commit `.luxar.zarr` or export folders (they're in .gitignore).
