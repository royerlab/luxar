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
   `serve.py`. Recipient runs `python3 serve.py`.
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
Recipient just needs Python 3: `cd my_export && python3 serve.py`. Say `python3`, not
`python`: stock Ubuntu (and macOS) have no `python` command; on Windows it is
`python serve.py`. No Luxar, no Node, no pip. Options:

```bash
python3 serve.py --port 9000        # default 8000; the next free port is used if busy
python3 serve.py --no-open          # don't launch a browser (headless / ssh box)
python3 serve.py --host 0.0.0.0     # reachable from other machines (default 127.0.0.1)
```

`--host` defaults to `127.0.0.1` (local only). A scene that declares a touch control
panel defaults to `0.0.0.0` and starts its relay (`--control`; `--no-control`,
`--control-token SECRET`); `--host 127.0.0.1` keeps it local. Viewing from another
machine over ssh without exposing a port: `ssh -L 8000:127.0.0.1:8000 host` and open
`http://127.0.0.1:8000` locally.

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

### Prerequisite: build the launcher binaries FIRST (source checkout only)

A plain `pip install luxar` normally carries no launcher binary, so `--native` from a
pip install stops at the "run make build-launchers" error — and `make` targets exist only
in a clone. If the user installed from PyPI, ship the standalone folder instead (it needs
only Python 3 on the recipient's side), or clone the repo to build launchers.

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
smoke tests. It does not let the prebuilt Linux binary run without `libwebkit2gtk`: WebKit
is linked at build time, so the `webkit2gtk-4.1` runtime must be present to start.

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
