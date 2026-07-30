# luxar-launcher

Native Go launcher embedded in `luxar export --native`-produced bundles
(`.app` on macOS, portable folder on Linux). Serves the bundled viewer
and zarr data over a local HTTP server and presents it inside a system
WebView window. Setting `LUXAR_LAUNCHER_NO_WEBVIEW=1` falls back to the
user's default browser (no native window) — useful for headless smoke
tests and minimal Linux installs without `libwebkit2gtk`.

The launcher forces browser revalidation for mutable `/data` responses while
leaving content-hashed `/viewer/assets` cacheable. It also passes a cache budget
to the viewer via `?cacheBudgetMB=<N>` (default **2048** — desktop-class).
WebKit (WKWebView / WebKitGTK) does not implement `performance.memory`, so the
viewer cannot auto-size its in-memory caches from the JS heap the way it does
in Chrome. Without this override, it would fall back to a tiny fixed budget and
re-decode timelapse
frames on every loop. Override with `LUXAR_CACHE_BUDGET_MB=<N>` on a
memory-constrained machine (e.g. `=512`).

## Build

From the project root:

```bash
make install-go        # one-time, only if Go is not installed
make build-launchers   # builds the host-platform launcher (CGO required)
```

Output lands in `packages/luxar/src/luxar/cli/_launchers/`. On macOS the
target is `darwin-universal` (arm64 + amd64 merged with `lipo`); on
Linux it's `linux-<arch>`. Linux + Windows binaries cannot be cross-
compiled from macOS (CGO blocks pure cross-compile) — they must be
built on a host of the matching OS, typically via CI.

`luxar export --native …` copies the appropriate binary into the bundle
it produces. The binaries themselves are gitignored.

## Runtime layout

The launcher locates the viewer and data via paths relative to its own
executable:

| Platform | Layout |
|----------|--------|
| macOS    | `<App>.app/Contents/MacOS/launcher` ⟶ `<App>.app/Contents/Resources/{viewer,data}` |
| Linux    | `<dir>/luxar-launcher` ⟶ `<dir>/{viewer,data}` |

If neither layout matches, the launcher prints a clear error and exits.

## Source

`main.go` (~220 lines) using only the Go stdlib plus the WebView binding:

- `github.com/webview/webview_go` for the native window (WKWebView /
  WebView2 / WebKitGTK)
- `net/http` + `http.FileServer` for serving viewer and data
- a small cache-policy wrapper that revalidates `/data` but not viewer assets
- `os/exec` for the browser fallback (`open` / `xdg-open` / `rundll32`)
- `os/signal` for graceful Ctrl-C shutdown

## Assets (app icon)

`assets/` holds two macOS-only helper scripts that regenerate the icon
shipped in the `.app` bundle. Their rendered outputs live alongside the
Python package (`packages/luxar/src/luxar/cli/_launcher_assets/`) so they
ride along with `pip install luxar`; the scripts only need re-running when
the logo changes.

| Script | Purpose |
|--------|---------|
| `build_logo.py` | Renders the 🌌 emoji (Apple Color Emoji, native 160 px strike upscaled with LANCZOS) to a 1024×1024 `luxar-logo.png`. Run via `hatch run python packages/luxar-launcher/assets/build_logo.py`. |
| `build_icons.sh` | Packs `luxar-logo.png` into `AppIcon.icns` using macOS `sips` + `iconutil`. macOS-only; skips with a notice on other platforms. |

## Build dependencies (Linux)

The WebView binding links against `libwebkit2gtk-4.1` (or `4.0` on older
distros). Install before running `make build-launchers`:

```bash
sudo apt-get install -y libwebkit2gtk-4.1-dev pkg-config
```

End users who run the prebuilt binary need only the runtime library
(`libwebkit2gtk-4.1` without `-dev`), which is present on every modern
desktop Linux distribution.
