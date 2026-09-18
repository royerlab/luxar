# luxar-launcher

Native Go launcher embedded in `luxar export --native`-produced bundles
(`.app` on macOS, portable folder on Linux). Serves the bundled viewer
and zarr data over a local HTTP server and presents it inside a system
WebView window. Setting `LUXAR_LAUNCHER_NO_WEBVIEW=1` falls back to the
user's default browser (no native window) — useful for headless smoke
tests. It does *not* let the launcher run without `libwebkit2gtk`: the
prebuilt Linux binary links WebKit at build time and will not start
without the webkit2gtk-4.1 runtime (see "Build dependencies (Linux)"
below).

The launcher forces browser revalidation for everything it serves — mutable
`/data` responses and the unhashed viewer shell (`index.html`, wasm) alike —
while leaving content-hashed `/viewer/assets` cacheable. It also passes a cache budget
to the viewer via `?cacheBudgetMB=<N>` (default **2048** — desktop-class).
WebKit (WKWebView / WebKitGTK) does not implement `performance.memory`, so the
viewer cannot auto-size its in-memory caches from the JS heap the way it does
in Chrome. Without this override, it would fall back to a tiny fixed budget and
re-decode timelapse
frames on every loop. Override with `LUXAR_CACHE_BUDGET_MB=<N>` on a
memory-constrained machine (e.g. `=512`).

## Kiosk mode (remote control)

The app can host the remote-control relay, so an exported scene drives a kiosk
without a Python checkout. Off by default, because a native app that silently
started listening would be a surprise nobody asked for:

```bash
# Loopback only — useful for a second browser window on the same machine.
LUXAR_LAUNCHER_CONTROL=1 ./luxar-launcher

# A tablet on the LAN. The token is strongly advised: without one, anything
# that can reach this machine can drive the display.
LUXAR_LAUNCHER_CONTROL=1 \
  LUXAR_LAUNCHER_HOST=0.0.0.0 \
  LUXAR_LAUNCHER_CONTROL_TOKEN=$(openssl rand -hex 8) \
  ./luxar-launcher
```

The app prints the touch-panel URL on stderr. `LUXAR_LAUNCHER_HOST` resolves a
wildcard bind to a concrete address before printing, because `http://0.0.0.0:PORT`
is not something a tablet can dial.

`hub.go` is the relay, and it is the **second** implementation of one — the
first is `luxar.cli.control_hub`. Everything the two must agree on (roles, the
close code for a refused handshake, the JSON-RPC error codes, the pending cap)
is generated into `control_contract.go` from `control-contract/contract.yaml`;
`hatch run check-control-contract` fails the build if this copy drifts from the
Python one. Do not edit the generated file.

The relay refuses three things, mirroring the Python hub: an unknown `?role=`
(refused rather than defaulted, since a typo attaching as a *controller* would
attach with authority), a wrong token (compared in constant time), and a
cross-origin browser handshake — a WebSocket handshake is not subject to the
same-origin policy and has no CORS preflight, so without that check any page a
visitor opened could drive the display, and binding loopback would not help.

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
- a small cache-policy wrapper that revalidates everything except the
  content-hashed `/viewer/assets` chunks
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

The pinned `webview_go` still requests `webkit2gtk-4.0` from pkg-config, but
Luxar supplies a compatibility module that resolves that request to
**`webkit2gtk-4.1`**. Install the 4.1 dev package before running
`make build-launchers`:

```bash
sudo apt-get install -y libwebkit2gtk-4.1-dev pkg-config
```

The Makefile enables the compatibility module only when pkg-config can resolve
4.1; on an older development host with only 4.0, the native 4.0 module remains
available instead of being shadowed by the shim.

End users who run the prebuilt binary need only the runtime library, not the
`-dev` package: the SONAME `libwebkit2gtk-4.1.so.0`, shipped on Debian/Ubuntu
as **`libwebkit2gtk-4.1-0`**.

`LUXAR_LAUNCHER_NO_WEBVIEW=1` still cannot rescue a missing runtime: it is read
by Go code after the dynamic loader resolves WebKitGTK. The env var is for when
the library is present and you simply do not want a native window (for example,
in a headless smoke test).
