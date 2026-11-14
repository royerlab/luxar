package main

import (
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
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

func TestLocalViewerURLKeepsDataSameOrigin(t *testing.T) {
	for _, tc := range []struct{ bind, want string }{
		{"0.0.0.0", "127.0.0.1"},
		{"::", "[::1]"},
		{"192.0.2.10", "192.0.2.10"},
	} {
		got := viewerURLForOrigin("http://"+net.JoinHostPort(localHost(tc.bind), "8123"), 2048, true, "abcdefgh")
		parsed, err := url.Parse(got)
		if err != nil {
			t.Fatal(err)
		}
		origin := "http://" + net.JoinHostPort(strings.Trim(tc.want, "[]"), "8123")
		if parsed.Scheme+"://"+parsed.Host != origin || parsed.Query().Get("src") != origin+"/data" {
			t.Fatalf("local URL for %q = %q; want same-origin %s", tc.bind, got, origin)
		}
		if parsed.Query().Get("controlToken") != "abcdefgh" {
			t.Fatalf("token lost: %q", got)
		}
	}
}

func TestAdvertisedHostAndFixedPort(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	t.Setenv("LUXAR_LAUNCHER_HOST", "0.0.0.0")
	t.Setenv("LUXAR_LAUNCHER_ADVERTISE_HOST", "kiosk.local")
	t.Setenv("LUXAR_LAUNCHER_PORT", strconv.Itoa(port))
	t.Setenv("LUXAR_LAUNCHER_CONTROL", "1")
	t.Setenv("LUXAR_LAUNCHER_CONTROL_TOKEN", "")
	t.Setenv("LUXAR_LAUNCHER_NO_CONTROL_TOKEN", "")
	root := t.TempDir()
	viewer, local, srv, err := startServer(root)
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
	localParsed, err := url.Parse(local)
	if err != nil {
		t.Fatal(err)
	}
	if localParsed.Host != net.JoinHostPort("127.0.0.1", strconv.Itoa(port)) ||
		localParsed.Query().Get("src") != "http://"+localParsed.Host+"/data" {
		t.Fatalf("local viewer URL is not same-origin: %q", local)
	}
	panelPath, _ := panelFiles(root)
	panel, err := os.ReadFile(panelPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(panel), "http://"+parsed.Host+"/viewer/control.html?control") {
		t.Fatalf("panel URL = %q", panel)
	}
	if token := parsed.Query().Get("controlToken"); token == "" {
		t.Fatal("viewer URL has no generated token")
	}
	if _, _, _, err := startServer(root); err == nil {
		t.Fatal("second server silently moved a configured port")
	}
	resp, err := http.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/viewer/")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	shutdownServer(srv)
	_, _, restarted, err := startServer(root)
	if err != nil {
		t.Fatal(err)
	}
	defer shutdownServer(restarted)
	again, err := os.ReadFile(panelPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(again) != string(panel) {
		t.Fatalf("panel URL moved across restart: %q → %q", panel, again)
	}
}

func TestInvalidConfiguredPortIsRejected(t *testing.T) {
	for _, value := range []string{"0", "-1", "65536", "abc"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("LUXAR_LAUNCHER_PORT", value)
			if _, _, _, err := startServer(t.TempDir()); err == nil {
				t.Fatal("invalid configured port was accepted")
			}
		})
	}
}

func TestPanelWriteFailureUsesStateDirectory(t *testing.T) {
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := probe.Addr().(*net.TCPAddr).Port
	probe.Close()
	t.Setenv("LUXAR_LAUNCHER_HOST", "127.0.0.1")
	t.Setenv("LUXAR_LAUNCHER_PORT", strconv.Itoa(port))
	t.Setenv("LUXAR_LAUNCHER_CONTROL", "1")
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	panel, _ := panelFiles(root)
	if err := os.Mkdir(panel, 0o700); err != nil {
		t.Fatal(err)
	}
	_, _, srv, err := startServer(root)
	if err != nil {
		t.Fatal(err)
	}
	defer shutdownServer(srv)
	fallbackPanel, _, err := statePanelFiles(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.ReadFile(fallbackPanel); err != nil {
		t.Fatalf("panel URL not saved in state directory: %v", err)
	}
	resp, err := http.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/viewer/")
	if err != nil {
		t.Fatalf("server stopped after panel write failure: %v", err)
	}
	resp.Body.Close()
}

func TestReadOnlyAppParentKeepsControlAvailable(t *testing.T) {
	for _, host := range []string{"127.0.0.1", "0.0.0.0"} {
		t.Run(host, func(t *testing.T) {
			t.Setenv("LUXAR_LAUNCHER_HOST", host)
			t.Setenv("LUXAR_LAUNCHER_CONTROL", "1")
			t.Setenv("LUXAR_LAUNCHER_CONTROL_TOKEN", "")
			t.Setenv("LUXAR_LAUNCHER_NO_CONTROL_TOKEN", "")
			t.Setenv("XDG_STATE_HOME", t.TempDir())
			t.Setenv("HOME", t.TempDir())
			parent := t.TempDir()
			root := filepath.Join(parent, "scene")
			if err := os.Mkdir(root, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(parent, 0o555); err != nil {
				t.Fatal(err)
			}
			defer os.Chmod(parent, 0o700)
			viewer, _, srv, err := startServer(root)
			if err != nil {
				t.Fatal(err)
			}
			shutdownServer(srv)
			panel, tokenPath, err := statePanelFiles(root)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := os.ReadFile(panel); err != nil {
				t.Fatalf("panel URL missing from state directory: %v", err)
			}
			if host == "0.0.0.0" {
				token := mustReadToken(t, tokenPath)
				if got := mustParseURL(t, viewer).Query().Get("controlToken"); got != token {
					t.Fatalf("viewer token = %q, saved token = %q", got, token)
				}
				_, _, restarted, err := startServer(root)
				if err != nil {
					t.Fatal(err)
				}
				shutdownServer(restarted)
				if got := mustReadToken(t, tokenPath); got != token {
					t.Fatalf("token changed across restart: %q to %q", token, got)
				}
			}
		})
	}
}

func TestStateDirectoryCannotBeServed(t *testing.T) {
	root := t.TempDir()
	t.Setenv("XDG_STATE_HOME", filepath.Join(root, "state"))
	t.Setenv("HOME", root)
	if _, _, err := statePanelFiles(root); err == nil {
		t.Fatal("accepted a control token path inside the served root")
	}
	if runtime.GOOS != "darwin" {
		outside := t.TempDir()
		link := filepath.Join(outside, "state")
		if err := os.Symlink(root, link); err != nil {
			t.Fatal(err)
		}
		t.Setenv("XDG_STATE_HOME", link)
		if _, _, err := statePanelFiles(root); err == nil {
			t.Fatal("accepted a symlinked control token path inside the served root")
		}
	}
}

func mustReadToken(t *testing.T, path string) string {
	t.Helper()
	token, err := readControlToken(path)
	if err != nil {
		t.Fatal(err)
	}
	return token
}

func mustParseURL(t *testing.T, value string) *url.URL {
	t.Helper()
	parsed, err := url.Parse(value)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}

func TestControlTokenFileValidation(t *testing.T) {
	t.Setenv("LUXAR_LAUNCHER_CONTROL_TOKEN", "")
	t.Setenv("LUXAR_LAUNCHER_NO_CONTROL_TOKEN", "")
	for _, tc := range []struct {
		name, contents string
		mode           os.FileMode
		want           string
	}{
		{"read-only file with newline", "\ufeffabcdefgh12345678\n", 0o400, "abcdefgh12345678"},
		{"ordinary copied file", "abcdefgh12345678\n", 0o644, ""},
		{"invalid punctuation", "abc defgh12345678", 0o600, ""},
		{"short token", "abc", 0o600, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			_, path := panelFiles(root)
			t.Cleanup(func() { _ = os.Remove(path) })
			if err := os.WriteFile(path, []byte(tc.contents), tc.mode); err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(path, tc.mode); err != nil {
				t.Fatal(err)
			}
			got, err := controlToken(root, "0.0.0.0")
			if tc.want == "" {
				if err == nil {
					t.Fatalf("accepted unsafe or invalid token %q", got)
				}
			} else if err != nil || got != tc.want {
				t.Fatalf("controlToken = %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}

func TestConcurrentTokenCreationReusesExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "token.txt")
	type result struct {
		token string
		err   error
	}
	results := make(chan result, 2)
	var ready sync.WaitGroup
	ready.Add(2)
	start := make(chan struct{})
	for _, candidate := range []string{"abcdefgh12345678", "other-token"} {
		go func(candidate string) {
			ready.Done()
			<-start
			token, err := createControlToken(path, candidate)
			results <- result{token, err}
		}(candidate)
	}
	ready.Wait()
	close(start)
	first, second := <-results, <-results
	if first.err != nil || second.err != nil || first.token == "" || first.token != second.token {
		t.Fatalf("concurrent creation = %q, %v and %q, %v", first.token, first.err, second.token, second.err)
	}
	if saved := mustReadToken(t, path); saved != first.token {
		t.Fatalf("saved token = %q, returned token = %q", saved, first.token)
	}
}

func TestBracketedIPv6LoopbackNeedsNoToken(t *testing.T) {
	t.Setenv("LUXAR_LAUNCHER_HOST", "[::1]")
	t.Setenv("LUXAR_LAUNCHER_CONTROL_TOKEN", "")
	t.Setenv("LUXAR_LAUNCHER_NO_CONTROL_TOKEN", "")
	if got := bindHost(); got != "::1" {
		t.Fatalf("bind host = %q", got)
	}
	if token, err := controlToken(t.TempDir(), bindHost()); err != nil || token != "" {
		t.Fatalf("loopback token = %q, %v", token, err)
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

	viewer, _, srv, err := startServer(root)
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
	_, tokenPath := panelFiles(root)
	tokenInfo, err := os.Stat(tokenPath)
	if err != nil {
		t.Fatal(err)
	}
	if tokenInfo.Mode().Perm() != 0o600 {
		t.Fatalf("token file mode = %o", tokenInfo.Mode().Perm())
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

	_, _, srv2, err := startServer(root)
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

	viewerURL, _, srv, err := startServer(root)
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
