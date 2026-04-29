# luxar-launcher

Native Go launcher embedded in `luxar export --native`-produced bundles
(`.app` on macOS, portable folder on Linux). Serves the bundled viewer
and zarr data over a local HTTP server and presents it inside a system
WebView window. Setting `LUXAR_LAUNCHER_NO_WEBVIEW=1` falls back to the
user's default browser (no native window) — useful for headless smoke
tests and minimal Linux installs without `libwebkit2gtk`.

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

`main.go` (~150 lines) using only the Go stdlib plus the WebView binding:

- `github.com/webview/webview_go` for the native window (WKWebView /
  WebView2 / WebKitGTK)
- `net/http` + `http.FileServer` for serving viewer and data
- `os/exec` for the browser fallback (`open` / `xdg-open` / `rundll32`)
- `os/signal` for graceful Ctrl-C shutdown
- CORS middleware mirroring the policy used by `luxar serve`

## Build dependencies (Linux)

The WebView binding links against `libwebkit2gtk-4.1` (or `4.0` on older
distros). Install before running `make build-launchers`:

```bash
sudo apt-get install -y libwebkit2gtk-4.1-dev pkg-config
```

End users who run the prebuilt binary need only the runtime library
(`libwebkit2gtk-4.1` without `-dev`), which is present on every modern
desktop Linux distribution.
