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
// for headless smoke-tests where no window is wanted; it does NOT let the
// binary run without a WebView runtime — cgo links libwebkit2gtk-4.1 at
// build time, so the loader aborts before main() on a system lacking it.
package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
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

// withCachePolicy forces revalidation for everything the launcher serves —
// mutable scene data under /data and the unhashed viewer shell (index.html,
// wasm) alike — EXCEPT the content-hashed viewer chunks under /viewer/assets/,
// whose filenames embed a build hash and are safe to cache indefinitely.
func withCachePolicy(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/viewer/assets/") {
			if w.Header().Get("Cache-Control") == "" {
				w.Header().Set("Cache-Control", "no-cache")
			}
		}
		next.ServeHTTP(w, r)
	})
}

// bindHost is the address the app serves on.
//
// Loopback unless LUXAR_LAUNCHER_HOST says otherwise. A native app that
// silently started listening on the network would be a surprise nobody asked
// for.
func bindHost() string {
	if v := strings.TrimSpace(os.Getenv("LUXAR_LAUNCHER_HOST")); v != "" {
		return v
	}
	return "127.0.0.1"
}

// dialHost is the address to put in a URL for a server bound to `host`.
//
// A wildcard bind is not a destination: `http://0.0.0.0:PORT` is unreachable
// from the tablet a kiosk panel runs on, so the URL needs a concrete address.
// Asks the routing table which interface would carry outbound traffic; nothing
// is sent, because connect() on a UDP socket only fixes a route.
func dialHost(host string) string {
	if host != "0.0.0.0" && host != "::" && host != "" {
		return host
	}
	conn, err := net.Dial("udp", "192.0.2.1:9") // RFC 5737, never routed.
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()
	if addr, ok := conn.LocalAddr().(*net.UDPAddr); ok {
		return addr.IP.String()
	}
	return "127.0.0.1"
}

var hostLabel = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$`)

// advertisedHost validates the name embedded in the panel, viewer and data URLs.
func advertisedHost(bind string) (string, error) {
	value := strings.TrimSpace(os.Getenv("LUXAR_LAUNCHER_ADVERTISE_HOST"))
	if value == "" {
		return dialHost(bind), nil
	}
	if strings.HasPrefix(value, "[") != strings.HasSuffix(value, "]") {
		return "", fmt.Errorf("invalid LUXAR_LAUNCHER_ADVERTISE_HOST %q", value)
	}
	bare := strings.TrimPrefix(strings.TrimSuffix(value, "]"), "[")
	if ip := net.ParseIP(bare); ip != nil {
		if ip.IsUnspecified() || ip.IsMulticast() || strings.Contains(bare, "%") {
			return "", fmt.Errorf("LUXAR_LAUNCHER_ADVERTISE_HOST must name a reachable address")
		}
		return ip.String(), nil
	}
	name := strings.TrimSuffix(value, ".")
	labels := strings.Split(name, ".")
	if len(name) > 253 || name == "" {
		return "", fmt.Errorf("invalid LUXAR_LAUNCHER_ADVERTISE_HOST %q", value)
	}
	for _, label := range labels {
		if !hostLabel.MatchString(label) {
			return "", fmt.Errorf("invalid LUXAR_LAUNCHER_ADVERTISE_HOST %q", value)
		}
	}
	if last := labels[len(labels)-1]; strings.IndexFunc(last, func(r rune) bool { return r < '0' || r > '9' }) == -1 {
		return "", fmt.Errorf("invalid LUXAR_LAUNCHER_ADVERTISE_HOST %q", value)
	}
	return name, nil
}

func listenPort() (string, error) {
	value := strings.TrimSpace(os.Getenv("LUXAR_LAUNCHER_PORT"))
	if value == "" {
		return "0", nil
	}
	port, err := strconv.Atoi(value)
	if err != nil || port < 1 || port > 65535 {
		return "", fmt.Errorf("LUXAR_LAUNCHER_PORT must be between 1 and 65535")
	}
	return strconv.Itoa(port), nil
}

// panelFiles live beside the .app or portable folder, outside the served root.
func panelFiles(root string) (string, string) {
	bundle := root
	if filepath.Base(root) == "Resources" && filepath.Base(filepath.Dir(root)) == "Contents" {
		bundle = filepath.Dir(filepath.Dir(root))
	}
	base := filepath.Join(filepath.Dir(bundle), filepath.Base(bundle))
	return base + ".control-panel.txt", base + ".control-token.txt"
}

func controlToken(root, host string) (string, error) {
	if os.Getenv("LUXAR_LAUNCHER_NO_CONTROL_TOKEN") == "1" || os.Getenv("LUXAR_LAUNCHER_CONTROL_TOKEN") == "off" {
		return "", nil
	}
	if token := os.Getenv("LUXAR_LAUNCHER_CONTROL_TOKEN"); token != "" {
		if token != strings.TrimSpace(token) {
			return "", fmt.Errorf("LUXAR_LAUNCHER_CONTROL_TOKEN must have no surrounding whitespace")
		}
		return token, nil
	}
	if host == "127.0.0.1" || host == "::1" || strings.EqualFold(host, "localhost") {
		return "", nil
	}
	_, path := panelFiles(root)
	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 {
			return "", fmt.Errorf("unsafe control token file %s", path)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		if token := strings.TrimSpace(string(data)); token != "" && token == string(data) {
			return token, nil
		}
		return "", fmt.Errorf("invalid control token file %s", path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	bytes := make([]byte, 24)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	token := base64.RawURLEncoding.EncodeToString(bytes)
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", err
	}
	if _, err = f.WriteString(token); err != nil {
		f.Close()
		os.Remove(path)
		return "", err
	}
	if err = f.Close(); err != nil {
		os.Remove(path)
		return "", err
	}
	return token, nil
}

func writePanelURL(path, panelURL string) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".luxar-panel-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.WriteString(panelURL + "\n"); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// startServer binds the configured port (or a free port) and starts the file server on it in a
// background goroutine. The returned URL is what the WebView (or fallback
// browser) should load; the returned *http.Server is the handle for graceful
// shutdown.
//
// With LUXAR_LAUNCHER_CONTROL=1 it also mounts the remote-control relay at
// /control and prints the touch-panel URL, so an exported app can run a kiosk
// without a Python checkout.
func startServer(root string) (string, *http.Server, error) {
	host := bindHost()
	portSpec, err := listenPort()
	if err != nil {
		return "", nil, err
	}
	reachable, err := advertisedHost(host)
	if err != nil {
		return "", nil, err
	}
	control := os.Getenv("LUXAR_LAUNCHER_CONTROL") == "1"
	token := ""
	if control {
		token, err = controlToken(root, host)
		if err != nil {
			return "", nil, err
		}
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(host, portSpec))
	if err != nil {
		return "", nil, fmt.Errorf("bind %s: %w", host, err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	origin := "http://" + net.JoinHostPort(reachable, strconv.Itoa(port))
	dataURL := origin + "/data"
	// Inject a cache budget: the viewer auto-sizes its in-memory caches from
	// `performance.memory`, which WKWebView (WebKit) does not implement — so
	// without this the app would fall back to a tiny fixed budget and re-decode
	// timelapse frames every loop. We know this is a desktop app, so we supply a
	// generous default the viewer splits across its cache tiers.
	viewerURL := fmt.Sprintf("%s/viewer/?src=%s&cacheBudgetMB=%d", origin, dataURL, cacheBudgetMB())

	// No CORS headers: the viewer (/viewer) and its data (/data) are served
	// from this one origin, so same-origin fetches need none. A wildcard here
	// would only let an unrelated web page read the locally-served scene.
	// The relay shares this origin with both pages, so neither needs a URL to
	// find the other and the same-host Origin check below is meaningful.
	var handler http.Handler = withCachePolicy(http.FileServer(http.Dir(root)))
	if control {
		relay := newHub(token)
		mux := http.NewServeMux()
		// Registered BEFORE the catch-all: Go's mux prefers the longer
		// pattern, but keeping the order explicit matches the Python side,
		// where a Mount at "/" would swallow the socket route entirely.
		mux.HandleFunc("/control", relay.handler())
		mux.Handle("/", handler)
		handler = mux
		viewerURL += "&control"
		panelURL := origin + "/viewer/control.html?control"
		if token != "" {
			viewerURL += "&controlToken=" + url.QueryEscape(token)
			panelURL += "&controlToken=" + url.QueryEscape(token)
		} else if host != "127.0.0.1" && host != "localhost" && host != "::1" {
			fmt.Fprintln(os.Stderr,
				"warning: control relay reachable from the network without a token")
		}
		path, _ := panelFiles(root)
		if err := writePanelURL(path, panelURL); err != nil {
			ln.Close()
			return "", nil, fmt.Errorf("write panel URL to %s: %w", path, err)
		}
		fmt.Fprintf(os.Stderr, "control panel: %s\n", panelURL)
		fmt.Fprintf(os.Stderr, "control panel URL saved to %s\n", path)
	}

	srv := &http.Server{
		Handler:           handler,
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

	// Browser-fallback mode: useful for headless smoke tests and for
	// users who explicitly want a real browser tab (devtools, extensions).
	// Note this cannot rescue a system missing libwebkit2gtk — cgo links
	// the runtime at build time, so the loader aborts before we get here.
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
