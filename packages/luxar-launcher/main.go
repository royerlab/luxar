// Luxar standalone launcher: serves a bundled viewer + zarr scene over a
// local HTTP server and presents it inside a native WebView window.
//
// Layout discovered at runtime, relative to the launcher binary:
//
//	macOS .app:     <bundle>/Contents/MacOS/<exe>
//	                resources at <bundle>/Contents/Resources/{viewer,data}
//	Linux folder:   <dir>/<exe>
//	                resources at <dir>/{viewer,data}
//
// Window lifecycle: the embedded WebView's main loop blocks until the
// user closes the window; on close we shut down the HTTP server. Setting
// LUXAR_LAUNCHER_NO_WEBVIEW=1 falls back to the system default browser
// (no native window, server runs until SIGINT/SIGTERM). This is useful
// for headless smoke-tests and for environments without a WebView
// runtime (e.g. minimal Linux installs missing libwebkit2gtk).
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
	"time"

	webview "github.com/webview/webview_go"
)

const (
	defaultWindowWidth  = 1400
	defaultWindowHeight = 900
)

// resolveResourceRoot returns the directory containing viewer/ and data/
// relative to the running executable. It checks the macOS .app layout
// first, then falls back to the sibling-of-binary layout used on Linux.
func resolveResourceRoot() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	exe, err = filepath.EvalSymlinks(exe)
	if err != nil {
		return "", err
	}
	dir := filepath.Dir(exe)

	// macOS .app: <bundle>/Contents/MacOS/<exe> -> <bundle>/Contents/Resources
	if filepath.Base(dir) == "MacOS" {
		candidate := filepath.Join(dir, "..", "Resources")
		if hasViewerAndData(candidate) {
			return filepath.Clean(candidate), nil
		}
	}

	// Linux/portable: viewer/ and data/ as siblings of the binary
	if hasViewerAndData(dir) {
		return dir, nil
	}

	return "", fmt.Errorf("could not locate viewer/ and data/ near %s", exe)
}

func hasViewerAndData(root string) bool {
	v, err := os.Stat(filepath.Join(root, "viewer", "index.html"))
	if err != nil || v.IsDir() {
		return false
	}
	d, err := os.Stat(filepath.Join(root, "data"))
	if err != nil || !d.IsDir() {
		return false
	}
	return true
}

func openSystemBrowser(url string) error {
	var cmd string
	var args []string
	switch runtime.GOOS {
	case "darwin":
		cmd = "open"
	case "windows":
		cmd = "rundll32"
		args = []string{"url.dll,FileProtocolHandler"}
	default:
		cmd = "xdg-open"
	}
	args = append(args, url)
	return exec.Command(cmd, args...).Start()
}

// startServer binds a free localhost port and starts the file server on
// it in a background goroutine. The returned URL is what the WebView (or
// fallback browser) should load; the returned *http.Server is the handle
// for graceful shutdown.
func startServer(root string) (string, *http.Server, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", nil, fmt.Errorf("bind localhost: %w", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	dataURL := fmt.Sprintf("http://127.0.0.1:%d/data", port)
	// Inject a cache budget: the viewer auto-sizes its in-memory caches from
	// `performance.memory`, which WKWebView (WebKit) does not implement — so
	// without this the app would fall back to a tiny fixed budget and re-decode
	// timelapse frames every loop. We know this is a desktop app, so we supply a
	// generous default the viewer splits across its cache tiers.
	viewerURL := fmt.Sprintf("http://127.0.0.1:%d/viewer/?src=%s&cacheBudgetMB=%d", port, dataURL, cacheBudgetMB())

	// No CORS headers: the viewer (/viewer) and its data (/data) are served
	// from this one origin, so same-origin fetches need none. A wildcard here
	// would only let an unrelated web page read the locally-served scene.
	srv := &http.Server{
		Handler:           http.FileServer(http.Dir(root)),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("luxar-launcher: server error: %v", err)
		}
	}()
	return viewerURL, srv, nil
}

// cacheBudgetMB is the total in-memory cache pool (MB) the launcher tells the
// viewer to use, via `?cacheBudgetMB=`. WKWebView has no `performance.memory`,
// so the viewer cannot auto-size from the heap; as a desktop app we supply a
// generous default (the viewer caps individual tiers internally). Override with
// LUXAR_CACHE_BUDGET_MB on a memory-constrained machine (e.g. `=512`).
func cacheBudgetMB() int {
	const def = 2048 // desktop-class default (~2 GB total cache pool)
	if v := os.Getenv("LUXAR_CACHE_BUDGET_MB"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func shutdownServer(srv *http.Server) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}

// newViewerWindow creates the WebView, points it at the viewer URL, and
// returns the live handle to the caller. The caller must call Run() on
// the main OS thread (post-LockOSThread) and Destroy() afterwards. The
// returned handle's Terminate() method is goroutine-safe and used by
// the signal handler to break out of Run() without os.Exit (which would
// leak WebKit child processes by skipping Destroy()).
func newViewerWindow(viewerURL string) webview.WebView {
	w := webview.New(false)
	w.SetTitle("Luxar Viewer")
	w.SetSize(defaultWindowWidth, defaultWindowHeight, webview.HintNone)
	w.Navigate(viewerURL)
	return w
}

// runBrowserFallback opens the system browser at viewerURL and blocks
// until SIGINT/SIGTERM is received. Selected by setting
// LUXAR_LAUNCHER_NO_WEBVIEW=1.
func runBrowserFallback(viewerURL string) {
	if err := openSystemBrowser(viewerURL); err != nil {
		log.Printf("luxar-launcher: could not open browser (%v); navigate manually to %s", err, viewerURL)
	}
	fmt.Printf("Luxar viewer: %s\n", viewerURL)
	fmt.Println("Press Ctrl+C to stop the server.")
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig
}

func main() {
	// The webview library requires the main OS thread for its UI loop.
	// Lock unconditionally — cheap, and harmless when the fallback
	// path is taken.
	runtime.LockOSThread()

	root, err := resolveResourceRoot()
	if err != nil {
		log.Fatalf("luxar-launcher: %v", err)
	}

	viewerURL, srv, err := startServer(root)
	if err != nil {
		log.Fatalf("luxar-launcher: %v", err)
	}
	defer shutdownServer(srv)

	// Browser-fallback mode: useful for headless smoke tests, for
	// minimal Linux environments without libwebkit2gtk, and for users
	// who explicitly want a real browser tab (devtools, extensions).
	if os.Getenv("LUXAR_LAUNCHER_NO_WEBVIEW") == "1" {
		runBrowserFallback(viewerURL)
		return
	}

	w := newViewerWindow(viewerURL)
	defer w.Destroy()

	// SIGINT / SIGTERM → break out of the WebView's run loop the same
	// way the user clicking the close button does. webview.Terminate
	// is documented as goroutine-safe; calling it lets w.Run() return
	// naturally so deferred Destroy() and shutdownServer() both fire
	// — no leaked WebKit child processes, no os.Exit shortcut.
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		w.Terminate()
	}()

	fmt.Printf("Luxar viewer: %s\n", viewerURL)
	w.Run()
	// Window closed (or signal-triggered Terminate) → fall through to
	// deferred Destroy() then shutdownServer().
}
