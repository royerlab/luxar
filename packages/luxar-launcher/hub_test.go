// Tests for the exported app's remote-control relay.
//
// The relay is a security boundary, and it is the SECOND implementation of one
// — the Python hub is the first. So these tests deliberately mirror
// `packages/luxar/src/luxar/cli/tests/test_control_hub.py`: id remapping,
// two controllers not colliding, "no viewer attached", event fan-out, and each
// of the three ways a handshake is refused. If the two suites ever disagree
// about an answer, one of the implementations is wrong.
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// serve starts the relay on a test server and returns its base ws:// URL.
func serve(t *testing.T, token string) (*hub, string) {
	t.Helper()
	h := newHub(token)
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	host := strings.TrimPrefix(srv.URL, "http://")
	mux.HandleFunc("/control", h.handler(host))
	return h, "ws" + strings.TrimPrefix(srv.URL, "http") + "/control"
}

func dial(t *testing.T, base, query string, header http.Header) *websocket.Conn {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, base+query, &websocket.DialOptions{HTTPHeader: header})
	if err != nil {
		t.Fatalf("dial %s: %v", query, err)
	}
	t.Cleanup(func() { _ = conn.CloseNow() })
	return conn
}

func send(t *testing.T, conn *websocket.Conn, payload string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := conn.Write(ctx, websocket.MessageText, []byte(payload)); err != nil {
		t.Fatalf("write: %v", err)
	}
}

func recv(t *testing.T, conn *websocket.Conn) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var frame map[string]any
	if err := json.Unmarshal(data, &frame); err != nil {
		t.Fatalf("unmarshal %q: %v", data, err)
	}
	return frame
}

// waitFor polls until `cond` holds, so a test never races the relay's
// registration without also hanging forever when it genuinely fails.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestRelayRemapsRequestIDs(t *testing.T) {
	h, base := serve(t, "")
	viewer := dial(t, base, "?role=viewer", nil)
	waitFor(t, "viewer to attach", h.viewerAttached)
	controller := dial(t, base, "?role=controller", nil)
	waitFor(t, "controller to attach", func() bool { return h.controllerCount() == 1 })

	send(t, controller, `{"jsonrpc":"2.0","id":1,"method":"recenterCamera","params":[]}`)
	forwarded := recv(t, viewer)
	// The id the VIEWER sees is the relay's, not the controller's: two
	// controllers both numbering from 1 would otherwise cross their replies.
	if forwarded["method"] != "recenterCamera" {
		t.Fatalf("method not forwarded: %v", forwarded)
	}
	hubID := forwarded["id"]

	send(t, viewer, `{"jsonrpc":"2.0","id":`+jsonNumber(hubID)+`,"result":null}`)
	reply := recv(t, controller)
	// And the id the CONTROLLER sees is its own again.
	if got := jsonNumber(reply["id"]); got != "1" {
		t.Fatalf("reply id = %s, want the controller's own 1", got)
	}
}

func TestTwoControllersDoNotCollide(t *testing.T) {
	// Both number from 1. Without remapping, one would receive the other's
	// reply and neither would notice.
	h, base := serve(t, "")
	viewer := dial(t, base, "?role=viewer", nil)
	waitFor(t, "viewer", h.viewerAttached)
	first := dial(t, base, "?role=controller", nil)
	second := dial(t, base, "?role=controller", nil)
	waitFor(t, "two controllers", func() bool { return h.controllerCount() == 2 })

	send(t, first, `{"jsonrpc":"2.0","id":1,"method":"getDimensions","params":[]}`)
	firstHubID := jsonNumber(recv(t, viewer)["id"])
	send(t, second, `{"jsonrpc":"2.0","id":1,"method":"getCameraPose","params":[]}`)
	secondHubID := jsonNumber(recv(t, viewer)["id"])
	if firstHubID == secondHubID {
		t.Fatalf("both calls got hub id %s", firstHubID)
	}

	// Answer the SECOND one only; it must reach the second controller.
	send(t, viewer, `{"jsonrpc":"2.0","id":`+secondHubID+`,"result":"for-second"}`)
	if got := recv(t, second)["result"]; got != "for-second" {
		t.Fatalf("second controller got %v", got)
	}
}

func TestNoViewerAttached(t *testing.T) {
	// A kiosk script WAITS for the display to boot, so this is an answer with
	// a code rather than a dropped connection.
	_, base := serve(t, "")
	controller := dial(t, base, "?role=controller", nil)
	send(t, controller, `{"jsonrpc":"2.0","id":7,"method":"recenterCamera","params":[]}`)
	frame := recv(t, controller)
	errObj, ok := frame["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected an error frame, got %v", frame)
	}
	if code, _ := errObj["code"].(float64); int(code) != CodeNoViewer {
		t.Fatalf("code = %v, want %d", errObj["code"], CodeNoViewer)
	}
	if got := jsonNumber(frame["id"]); got != "7" {
		t.Fatalf("id = %s, want 7", got)
	}
}

func TestEventsReachEveryController(t *testing.T) {
	h, base := serve(t, "")
	viewer := dial(t, base, "?role=viewer", nil)
	waitFor(t, "viewer", h.viewerAttached)
	first := dial(t, base, "?role=controller", nil)
	second := dial(t, base, "?role=controller", nil)
	waitFor(t, "two controllers", func() bool { return h.controllerCount() == 2 })

	send(t, viewer, `{"jsonrpc":"2.0","method":"event","params":["dimensions-changed",{"n":3}]}`)
	for i, conn := range []*websocket.Conn{first, second} {
		frame := recv(t, conn)
		if frame["method"] != EventMethod {
			t.Fatalf("controller %d got %v", i, frame)
		}
	}
}

func TestPendingIsBounded(t *testing.T) {
	// A wedged viewer never replies. Without a cap, one map entry leaks per
	// tap for the lifetime of the process.
	h, base := serve(t, "")
	viewer := dial(t, base, "?role=viewer", nil)
	waitFor(t, "viewer", h.viewerAttached)
	controller := dial(t, base, "?role=controller", nil)
	waitFor(t, "controller", func() bool { return h.controllerCount() == 1 })

	for i := 0; i < MaxPendingPerController+25; i++ {
		send(t, controller, `{"jsonrpc":"2.0","id":`+itoa(i+1)+`,"method":"recenterCamera","params":[]}`)
		recv(t, viewer) // Drain so the relay is not blocked on the write.
	}
	waitFor(t, "pending to settle at the cap", func() bool {
		return h.pendingCount() == MaxPendingPerController
	})
	_ = viewer
}

func TestControllerDisconnectPurgesPending(t *testing.T) {
	h, base := serve(t, "")
	viewer := dial(t, base, "?role=viewer", nil)
	waitFor(t, "viewer", h.viewerAttached)
	controller := dial(t, base, "?role=controller", nil)
	waitFor(t, "controller", func() bool { return h.controllerCount() == 1 })
	send(t, controller, `{"jsonrpc":"2.0","id":1,"method":"recenterCamera","params":[]}`)
	recv(t, viewer)
	waitFor(t, "one pending", func() bool { return h.pendingCount() == 1 })

	_ = controller.CloseNow()
	waitFor(t, "pending purged", func() bool { return h.pendingCount() == 0 })
}

// ── refusals ────────────────────────────────────────────────────────────────

func TestUnknownRoleIsRefused(t *testing.T) {
	// Refused rather than defaulted: a typo'd role silently attaching as a
	// CONTROLLER would attach with authority.
	h, base := serve(t, "")
	conn := dial(t, base, "?role=banana", nil)
	// Asserted BEFORE reading the close frame, deliberately: if the check were
	// removed the socket would attach and the read would block forever, and a
	// hanging test is a far worse failure than an assertion.
	waitFor(t, "no controller to attach", func() bool { return h.controllerCount() == 0 })
	expectClose(t, conn, ClosePolicyViolation)
}

func TestBadTokenIsRefused(t *testing.T) {
	h, base := serve(t, "hunter2")
	conn := dial(t, base, "?role=controller&token=wrong", nil)
	waitFor(t, "no controller to attach", func() bool { return h.controllerCount() == 0 })
	expectClose(t, conn, ClosePolicyViolation)

	// And the right one works.
	dial(t, base, "?role=controller&token=hunter2", nil)
	waitFor(t, "controller to attach", func() bool { return h.controllerCount() == 1 })
}

func TestCrossOriginBrowserHandshakeIsRefused(t *testing.T) {
	// Cross-Site WebSocket Hijacking: a WS handshake is not subject to the
	// same-origin policy and has no CORS preflight, so without this ANY page a
	// visitor opened could drive the display — and a loopback bind does not
	// help, because the visitor's browser is inside the trust boundary.
	h, base := serve(t, "")
	header := http.Header{}
	header.Set("Origin", "http://evil.example")
	conn := dial(t, base, "?role=controller", header)
	waitFor(t, "no controller to attach", func() bool { return h.controllerCount() == 0 })
	expectClose(t, conn, ClosePolicyViolation)
}

func TestMissingOriginIsAllowed(t *testing.T) {
	// Required, not sloppy: every non-browser client sends no Origin, and
	// refusing them would break every script.
	h, base := serve(t, "")
	dial(t, base, "?role=controller", nil)
	waitFor(t, "controller to attach", func() bool { return h.controllerCount() == 1 })
}

func TestOriginAllowed(t *testing.T) {
	cases := []struct {
		origin, host string
		want         bool
	}{
		{"", "kiosk.local:8080", true}, // non-browser client
		{"http://kiosk.local:8080", "kiosk.local:8080", true},
		{"https://kiosk.local:8080", "kiosk.local:8080", true}, // scheme ignored
		{"http://KIOSK.local:8080", "kiosk.local:8080", true},  // case-insensitive
		{"http://evil.example", "kiosk.local:8080", false},
		{"http://kiosk.local:9999", "kiosk.local:8080", false}, // port matters here
		{"null", "kiosk.local:8080", false},                    // sandboxed iframe
		{"not a url", "kiosk.local:8080", false},
	}
	for _, c := range cases {
		if got := originAllowed(c.origin, c.host); got != c.want {
			t.Errorf("originAllowed(%q, %q) = %v, want %v", c.origin, c.host, got, c.want)
		}
	}
}

func TestDialHostResolvesAWildcardBind(t *testing.T) {
	// `http://0.0.0.0:PORT` is not reachable from the tablet a panel runs on,
	// so a printed URL must carry a concrete address.
	for _, wildcard := range []string{"0.0.0.0", "::", ""} {
		if got := dialHost(wildcard); got == wildcard {
			t.Errorf("dialHost(%q) returned the sentinel itself", wildcard)
		}
	}
	if got := dialHost("10.0.0.5"); got != "10.0.0.5" {
		t.Errorf("dialHost kept a concrete address as %q", got)
	}
}

// ── helpers ─────────────────────────────────────────────────────────────────

func expectClose(t *testing.T, conn *websocket.Conn, want websocket.StatusCode) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, _, err := conn.Read(ctx)
	if websocket.CloseStatus(err) != want {
		t.Fatalf("close status = %v (err %v), want %v", websocket.CloseStatus(err), err, want)
	}
}

// jsonNumber renders a decoded JSON number without scientific notation, so an
// id can be spliced back into a frame.
func jsonNumber(v any) string {
	switch n := v.(type) {
	case float64:
		return itoa(int(n))
	case string:
		return `"` + n + `"`
	default:
		return "null"
	}
}

func itoa(i int) string {
	return json.Number(strings.TrimSuffix(strings.TrimSpace(formatInt(i)), ".0")).String()
}

func formatInt(i int) string {
	b, _ := json.Marshal(i)
	return string(b)
}
