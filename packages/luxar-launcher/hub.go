// The remote-control relay, so an exported native app can drive a kiosk.
//
// A second implementation of the same relay as `luxar.cli.control_hub`, and
// deliberately a DUMB one: it forwards frames between one display and any
// number of controllers, remaps request ids so two controllers cannot collide,
// and refuses anything that has no business connecting. It understands no
// methods. Every value it decides with — roles, close code, error codes, the
// pending cap — is GENERATED into control_contract.go from
// control-contract/contract.yaml, so this file and the Python hub cannot drift
// apart without the build failing.
//
// Why a relay at all: a browser cannot host a server, so the display and the
// touch panel need something in the middle. `luxar serve --control` is that for
// a dev checkout; this is that for the exported `.app`, which has no Python.
//
// Off unless asked for. `LUXAR_LAUNCHER_CONTROL=1` enables it, and the app
// still binds loopback until `LUXAR_LAUNCHER_HOST` names something else,
// because a native app that silently started listening on the LAN would be a
// surprise nobody asked for.
package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/coder/websocket"
)

// hub relays JSON-RPC frames between one viewer and many controllers.
type hub struct {
	token string

	mu sync.Mutex
	// The display. Only one is useful; a second replaces it, because a
	// reloaded page must be able to take over from its own dead socket.
	viewer *peer
	// Attached controllers, keyed by an id we assign.
	controllers map[int64]*peer
	nextPeerID  int64
	// Request ids we handed the viewer, mapped back to the controller that
	// asked. Two controllers both numbering from 1 would otherwise each
	// receive the other's replies.
	pending    map[int64]pendingCall
	nextHubID  int64
	pendingPer map[int64]int
}

type pendingCall struct {
	controller int64
	// The id the CONTROLLER used, restored on the way back so it can match
	// the reply to its own request.
	originalID json.RawMessage
}

// peer is one socket plus a serialising writer.
//
// The mutex matters: a controller's reply and a broadcast event can be written
// concurrently, and two goroutines writing one WebSocket interleave frames into
// garbage.
type peer struct {
	conn *websocket.Conn
	mu   sync.Mutex
}

func (p *peer) send(ctx context.Context, payload []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.conn.Write(ctx, websocket.MessageText, payload)
}

func newHub(token string) *hub {
	return &hub{
		token:       token,
		controllers: make(map[int64]*peer),
		pending:     make(map[int64]pendingCall),
		pendingPer:  make(map[int64]int),
	}
}

// originAllowed mirrors the Python hub's Cross-Site WebSocket Hijacking check.
//
// A WebSocket handshake is not subject to the same-origin policy and carries no
// CORS preflight, so without this ANY page a visitor opened could connect to
// this relay and drive the display — and binding loopback does not help,
// because the visitor's own browser is inside the trust boundary.
//
// A MISSING Origin is allowed: non-browser clients send none, and refusing them
// would break every script. The check exists to stop a *browser* being used as
// the attacker's proxy.
func originAllowed(origin, host string) bool {
	if origin == "" {
		return true
	}
	parsed, err := url.Parse(strings.TrimSpace(origin))
	if err != nil || parsed.Host == "" {
		// Unparseable, or `null` from a sandboxed iframe. Never matches.
		return false
	}
	return strings.EqualFold(parsed.Host, strings.TrimSpace(host))
}

// authorized compares the presented token in constant time.
//
// `subtle.ConstantTimeCompare` over bytes, like the Python side's
// `hmac.compare_digest` over UTF-8 — a length-independent comparison would leak
// the secret one byte at a time.
func (h *hub) authorized(token string) bool {
	if h.token == "" {
		return true
	}
	return subtle.ConstantTimeCompare([]byte(token), []byte(h.token)) == 1
}

// handler serves the relay endpoint.
func (h *hub) handler(host string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		role := r.URL.Query().Get(ParamRole)
		if role == "" {
			role = RoleController
		}
		// Accept FIRST, then refuse with a close code. Closing before the
		// handshake completes surfaces at the client as a bare HTTP error with
		// no code, so a controller could not tell "no relay here" from "your
		// token is wrong".
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			// We do the Origin check ourselves, below, against the Host this
			// request arrived on — the library's own list cannot know it.
			InsecureSkipVerify: true,
		})
		if err != nil {
			return
		}
		conn.SetReadLimit(MaxFrameBytes)

		ctx := r.Context()
		switch {
		case role != RoleViewer && role != RoleController:
			_ = conn.Close(ClosePolicyViolation, "unknown role")
			return
		case !originAllowed(r.Header.Get("Origin"), host):
			_ = conn.Close(ClosePolicyViolation, "cross-origin")
			return
		case !h.authorized(r.URL.Query().Get(ParamToken)):
			_ = conn.Close(ClosePolicyViolation, "bad token")
			return
		}

		p := &peer{conn: conn}
		if role == RoleViewer {
			h.serveViewer(ctx, p)
			return
		}
		h.serveController(ctx, p)
	}
}

func errorFrame(id json.RawMessage, code int, message string) []byte {
	if len(id) == 0 {
		id = json.RawMessage("null")
	}
	return []byte(fmt.Sprintf(
		`{"jsonrpc":%q,"id":%s,"error":{"code":%d,"message":%q}}`,
		JSONRPCVersion, id, code, message,
	))
}

// ── viewer side ─────────────────────────────────────────────────────────────

func (h *hub) serveViewer(ctx context.Context, p *peer) {
	h.mu.Lock()
	h.viewer = p
	h.mu.Unlock()
	defer func() {
		h.mu.Lock()
		if h.viewer == p {
			h.viewer = nil
		}
		h.mu.Unlock()
	}()

	for {
		_, data, err := p.conn.Read(ctx)
		if err != nil {
			return
		}
		h.fromViewer(ctx, data)
	}
}

// fromViewer routes a reply back to whoever asked, or fans an event out.
func (h *hub) fromViewer(ctx context.Context, data []byte) {
	var frame struct {
		ID     *json.RawMessage `json:"id"`
		Method string           `json:"method"`
	}
	if json.Unmarshal(data, &frame) != nil {
		return // Malformed from our own display: nothing useful to answer.
	}
	if frame.ID == nil {
		// A notification. `event` fans out; anything else is not ours.
		if frame.Method == EventMethod {
			h.broadcast(ctx, data)
		}
		return
	}

	var hubID int64
	if json.Unmarshal(*frame.ID, &hubID) != nil {
		return // Not an id we could have issued.
	}
	h.mu.Lock()
	call, ok := h.pending[hubID]
	if ok {
		delete(h.pending, hubID)
		h.pendingPer[call.controller]--
	}
	target := h.controllers[call.controller]
	h.mu.Unlock()
	if !ok || target == nil {
		return
	}
	_ = target.send(ctx, restoreID(data, call.originalID))
}

// restoreID swaps our hub id back for the controller's own.
func restoreID(data []byte, original json.RawMessage) []byte {
	var generic map[string]json.RawMessage
	if json.Unmarshal(data, &generic) != nil {
		return data
	}
	generic["id"] = original
	out, err := json.Marshal(generic)
	if err != nil {
		return data
	}
	return out
}

func (h *hub) broadcast(ctx context.Context, data []byte) {
	h.mu.Lock()
	targets := make([]*peer, 0, len(h.controllers))
	for _, c := range h.controllers {
		targets = append(targets, c)
	}
	h.mu.Unlock()
	for _, c := range targets {
		_ = c.send(ctx, data)
	}
}

// ── controller side ─────────────────────────────────────────────────────────

func (h *hub) serveController(ctx context.Context, p *peer) {
	h.mu.Lock()
	h.nextPeerID++
	id := h.nextPeerID
	h.controllers[id] = p
	h.mu.Unlock()
	defer h.dropController(id)

	for {
		_, data, err := p.conn.Read(ctx)
		if err != nil {
			return
		}
		h.fromController(ctx, p, id, data)
	}
}

// dropController forgets a controller AND the replies it will never read.
func (h *hub) dropController(id int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.controllers, id)
	delete(h.pendingPer, id)
	for hubID, call := range h.pending {
		if call.controller == id {
			delete(h.pending, hubID)
		}
	}
}

func (h *hub) fromController(ctx context.Context, p *peer, id int64, data []byte) {
	var frame struct {
		ID     *json.RawMessage `json:"id"`
		Method string           `json:"method"`
	}
	if err := json.Unmarshal(data, &frame); err != nil {
		_ = p.send(ctx, errorFrame(nil, CodeParseError, "invalid JSON"))
		return
	}
	if frame.Method == "" {
		_ = p.send(ctx, errorFrame(idOrNull(frame.ID), CodeInvalidRequest, "no method"))
		return
	}

	h.mu.Lock()
	viewer := h.viewer
	var hubID int64
	if frame.ID != nil && viewer != nil {
		h.nextHubID++
		hubID = h.nextHubID
		h.remember(hubID, id, *frame.ID)
	}
	h.mu.Unlock()

	if viewer == nil {
		// Not an error to abort on: a kiosk script waits for the display to
		// boot. Only a request gets an answer; a notification is dropped.
		if frame.ID != nil {
			_ = p.send(ctx, errorFrame(idOrNull(frame.ID), CodeNoViewer, "no viewer attached"))
		}
		return
	}
	if frame.ID == nil {
		_ = viewer.send(ctx, data) // Notification: forward as-is.
		return
	}
	if err := viewer.send(ctx, replaceID(data, hubID)); err != nil && !errors.Is(err, context.Canceled) {
		_ = p.send(ctx, errorFrame(idOrNull(frame.ID), CodeNoViewer, "viewer unreachable"))
	}
}

func idOrNull(id *json.RawMessage) json.RawMessage {
	if id == nil {
		return json.RawMessage("null")
	}
	return *id
}

// remember records a pending call, evicting this controller's oldest past the
// cap. Caller holds the lock.
//
// A wedged viewer never replies, and without a bound one map entry would leak
// per tap for the lifetime of the process. Eviction drops a reply the
// controller was never going to receive anyway.
func (h *hub) remember(hubID, controller int64, original json.RawMessage) {
	if h.pendingPer[controller] >= MaxPendingPerController {
		var oldest int64 = -1
		for candidate, call := range h.pending {
			if call.controller != controller {
				continue
			}
			if oldest == -1 || candidate < oldest {
				oldest = candidate
			}
		}
		if oldest != -1 {
			delete(h.pending, oldest)
			h.pendingPer[controller]--
		}
	}
	h.pending[hubID] = pendingCall{controller: controller, originalID: original}
	h.pendingPer[controller]++
}

func replaceID(data []byte, hubID int64) []byte {
	var generic map[string]json.RawMessage
	if json.Unmarshal(data, &generic) != nil {
		return data
	}
	generic["id"] = json.RawMessage(fmt.Sprintf("%d", hubID))
	out, err := json.Marshal(generic)
	if err != nil {
		return data
	}
	return out
}

// counts are for tests and status lines.
func (h *hub) controllerCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.controllers)
}

func (h *hub) viewerAttached() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.viewer != nil
}

func (h *hub) pendingCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.pending)
}
