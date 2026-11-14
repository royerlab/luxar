package main

import (
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/coder/websocket"
)

func TestAdvertisedHostValidation(t *testing.T) {
	for _, tc := range []struct{ input, want string }{
		{"kiosk.local", "kiosk.local"},
		{"kiosk.local.", "kiosk.local"},
		{"[2001:db8::1]", "2001:db8::1"},
		{"192.0.2.10", "192.0.2.10"},
		{"http://kiosk.local", ""},
		{"kiosk.local:8000", ""},
		{"a..b", ""},
		{"0.0.0.0", ""},
		{"[::]", ""},
		{"224.0.0.1", ""},
		{"123.456", ""},
		{"kiosk.3com", "kiosk.3com"},
		{"[127.0.0.1", ""},
	} {
		t.Run(tc.input, func(t *testing.T) {
			t.Setenv("LUXAR_LAUNCHER_ADVERTISE_HOST", tc.input)
			got, err := advertisedHost("127.0.0.1")
			if tc.want == "" {
				if err == nil {
					t.Fatalf("accepted invalid host %q as %q", tc.input, got)
				}
			} else if err != nil || got != tc.want {
				t.Fatalf("advertisedHost = %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}

func TestAdvertisedHostAndFixedPort(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	t.Setenv("LUXAR_LAUNCHER_HOST", "127.0.0.1")
	t.Setenv("LUXAR_LAUNCHER_ADVERTISE_HOST", "kiosk.local")
	t.Setenv("LUXAR_LAUNCHER_PORT", strconv.Itoa(port))
	t.Setenv("LUXAR_LAUNCHER_CONTROL", "1")
	root := t.TempDir()
	viewer, srv, err := startServer(root)
	if err != nil {
		t.Fatal(err)
	}
	defer shutdownServer(srv)
	parsed, err := url.Parse(viewer)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Host != net.JoinHostPort("kiosk.local", strconv.Itoa(port)) {
		t.Fatalf("viewer host = %q", parsed.Host)
	}
	if got := parsed.Query().Get("src"); got != "http://"+parsed.Host+"/data" {
		t.Fatalf("data URL = %q", got)
	}
	panelPath, _ := panelFiles(root)
	panel, err := os.ReadFile(panelPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(panel), "http://"+parsed.Host+"/viewer/control.html?control") {
		t.Fatalf("panel URL = %q", panel)
	}
	if _, _, err := startServer(root); err == nil {
		t.Fatal("second server silently moved a configured port")
	}
}

func TestInvalidConfiguredPortIsRejected(t *testing.T) {
	for _, value := range []string{"0", "-1", "65536", "abc"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("LUXAR_LAUNCHER_PORT", value)
			if _, _, err := startServer(t.TempDir()); err == nil {
				t.Fatal("invalid configured port was accepted")
			}
		})
	}
}

func TestPanelFilesOutsideMacAppAndPortableFolder(t *testing.T) {
	for _, tc := range []struct{ root, want string }{
		{"/opt/scene.app/Contents/Resources", "/opt/scene.app.control-panel.txt"},
		{"/opt/scene", "/opt/scene.control-panel.txt"},
	} {
		panel, token := panelFiles(tc.root)
		if panel != tc.want || token != strings.TrimSuffix(tc.want, "panel.txt")+"token.txt" {
			t.Fatalf("panelFiles(%q) = %q, %q", tc.root, panel, token)
		}
	}
}

func TestControlPanelTokenSurvivesRestartOutsideServedRoot(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(filepath.Dir(root), filepath.Base(root)+".control-panel.txt")
	t.Cleanup(func() { _ = os.Remove(path) })
	t.Cleanup(func() { _ = os.Remove(filepath.Join(filepath.Dir(root), filepath.Base(root)+".control-token.txt")) })
	t.Setenv("LUXAR_LAUNCHER_HOST", "0.0.0.0")
	t.Setenv("LUXAR_LAUNCHER_CONTROL", "1")
	t.Setenv("LUXAR_LAUNCHER_CONTROL_TOKEN", "")
	t.Setenv("LUXAR_LAUNCHER_NO_CONTROL_TOKEN", "")

	viewer, srv, err := startServer(root)
	if err != nil {
		t.Fatal(err)
	}
	panelBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	panel := strings.TrimSpace(string(panelBytes))
	parsed, err := url.Parse(panel)
	if err != nil {
		t.Fatal(err)
	}
	token := parsed.Query().Get("controlToken")
	if token == "" {
		t.Fatal("generated token missing from panel URL")
	}
	if !strings.Contains(viewer, "controlToken="+token) {
		t.Fatal("viewer does not carry relay token")
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("panel file mode = %o", info.Mode().Perm())
	}
	local := "ws://127.0.0.1:" + parsed.Port() + "/control"
	conn := dial(t, local, "?role=controller", nil)
	expectClose(t, conn, websocket.StatusPolicyViolation)
	conn = dial(t, local, "?role=controller&token="+url.QueryEscape(token), nil)
	send(t, conn, `{"jsonrpc":"2.0","id":1,"method":"getDimensions","params":[]}`)
	if got := recv(t, conn)["id"]; got != float64(1) {
		t.Fatalf("authenticated controller reply id = %v", got)
	}
	_ = conn.CloseNow()
	resp, err := http.Get("http://127.0.0.1:" + parsed.Port() + "/control-panel.txt")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		t.Fatal("panel file is exposed over HTTP")
	}
	shutdownServer(srv)

	_, srv2, err := startServer(root)
	if err != nil {
		t.Fatal(err)
	}
	defer shutdownServer(srv2)
	again, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	againURL, err := url.Parse(strings.TrimSpace(string(again)))
	if err != nil {
		t.Fatal(err)
	}
	if got := againURL.Query().Get("controlToken"); got != token {
		t.Fatalf("restart token = %q, want %q", got, token)
	}
}

func TestCacheBudgetMBDefault(t *testing.T) {
	t.Setenv("LUXAR_CACHE_BUDGET_MB", "")
	if got := cacheBudgetMB(); got != 2048 {
		t.Fatalf("default cacheBudgetMB = %d, want 2048", got)
	}
}

func TestCacheBudgetMBOverride(t *testing.T) {
	t.Setenv("LUXAR_CACHE_BUDGET_MB", "512")
	if got := cacheBudgetMB(); got != 512 {
		t.Fatalf("override cacheBudgetMB = %d, want 512", got)
	}
}

func TestCacheBudgetMBIgnoresInvalid(t *testing.T) {
	for _, bad := range []string{"0", "-5", "notanumber", "12.5"} {
		t.Setenv("LUXAR_CACHE_BUDGET_MB", bad)
		if got := cacheBudgetMB(); got != 2048 {
			t.Fatalf("cacheBudgetMB(%q) = %d, want fallback 2048", bad, got)
		}
	}
}

func TestHasViewerAndData(t *testing.T) {
	root := t.TempDir()
	// Missing everything.
	if hasViewerAndData(root) {
		t.Fatal("empty dir should not be a resource root")
	}
	// viewer/index.html only — still missing data/.
	viewer := filepath.Join(root, "viewer")
	if err := os.MkdirAll(viewer, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(viewer, "index.html"), []byte("<html></html>"), 0o644); err != nil {
		t.Fatal(err)
	}
	if hasViewerAndData(root) {
		t.Fatal("viewer without data/ should not be a resource root")
	}
	// Add data/ dir — now complete.
	if err := os.MkdirAll(filepath.Join(root, "data"), 0o755); err != nil {
		t.Fatal(err)
	}
	if !hasViewerAndData(root) {
		t.Fatal("viewer/index.html + data/ should be a valid resource root")
	}
	// data/ as a FILE (not dir) must be rejected.
	root2 := t.TempDir()
	v2 := filepath.Join(root2, "viewer")
	_ = os.MkdirAll(v2, 0o755)
	_ = os.WriteFile(filepath.Join(v2, "index.html"), []byte("x"), 0o644)
	_ = os.WriteFile(filepath.Join(root2, "data"), []byte("x"), 0o644)
	if hasViewerAndData(root2) {
		t.Fatal("data/ as a regular file should be rejected")
	}
}

func TestStartServerServesSameOriginNoCORS(t *testing.T) {
	root := t.TempDir()
	viewer := filepath.Join(root, "viewer")
	data := filepath.Join(root, "data")
	if err := os.MkdirAll(viewer, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(data, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(viewer, "index.html"), []byte("<html>ok</html>"), 0o644); err != nil {
		t.Fatal(err)
	}
	assets := filepath.Join(viewer, "assets")
	if err := os.MkdirAll(assets, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(assets, "app.01234567.js"), []byte("export {};"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(data, ".zattrs"), []byte(`{"type":"scene"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	viewerURL, srv, err := startServer(root)
	if err != nil {
		t.Fatalf("startServer: %v", err)
	}
	defer shutdownServer(srv)

	if viewerURL == "" {
		t.Fatal("expected a non-empty viewer URL")
	}

	// The data file is served...
	resp, err := http.Get(dataURLFrom(viewerURL))
	if err != nil {
		t.Fatalf("GET data: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET data status = %d, want 200", resp.StatusCode)
	}
	if len(body) == 0 {
		t.Fatal("expected non-empty data body")
	}
	// ...with NO cross-origin header (same-origin needs none), but with forced
	// revalidation because a scene can be replaced in place under the same URL.
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("unexpected CORS header %q; same-origin bundle must not emit one", got)
	}
	if got := resp.Header.Get("Cache-Control"); got != "no-cache" {
		t.Fatalf("data Cache-Control = %q, want no-cache", got)
	}

	// Content-hashed viewer assets are immutable and must not pay a revalidation
	// round trip on every launch.
	queryStart := indexOf(viewerURL, "?")
	if queryStart < 0 {
		t.Fatalf("viewer URL has no query string: %q", viewerURL)
	}
	assetURL := viewerURL[:queryStart] + "assets/app.01234567.js"
	assetResp, err := http.Get(assetURL)
	if err != nil {
		t.Fatalf("GET viewer asset: %v", err)
	}
	defer assetResp.Body.Close()
	if assetResp.StatusCode != http.StatusOK {
		t.Fatalf("GET viewer asset status = %d, want 200", assetResp.StatusCode)
	}
	if got := assetResp.Header.Get("Cache-Control"); got == "no-cache" {
		t.Fatal("content-hashed viewer asset must not be forced to revalidate")
	}

	// The unhashed viewer shell IS replaced in place on rebuild, so it must
	// revalidate just like mutable data (unlike the content-hashed assets).
	shellURL := viewerURL[:queryStart] + "index.html"
	shellResp, err := http.Get(shellURL)
	if err != nil {
		t.Fatalf("GET viewer shell: %v", err)
	}
	defer shellResp.Body.Close()
	if shellResp.StatusCode != http.StatusOK {
		t.Fatalf("GET viewer shell status = %d, want 200", shellResp.StatusCode)
	}
	if got := shellResp.Header.Get("Cache-Control"); got != "no-cache" {
		t.Fatalf("viewer shell Cache-Control = %q, want no-cache", got)
	}
}

// dataURLFrom derives the /data/.zattrs URL from the viewer URL, which is of the
// form http://127.0.0.1:<port>/viewer/?src=...  We only need the host:port.
func dataURLFrom(viewerURL string) string {
	// viewerURL: http://127.0.0.1:PORT/viewer/?src=...
	// Cut at "/viewer/".
	const marker = "/viewer/"
	i := indexOf(viewerURL, marker)
	if i < 0 {
		return viewerURL
	}
	return viewerURL[:i] + "/data/.zattrs"
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
