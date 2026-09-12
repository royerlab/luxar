# `core/app/control` — driving the viewer from outside the page

This directory is the viewer's side of the remote-control channel: a big display
renders a scene while something else — a kiosk touch panel, a script, an agent —
tells it what to show. The transport is a WebSocket to a relay hub served by
`luxar serve --control`, because a browser cannot host a server and several
parties may want to attach at once.

Full contract: `docs/guides/specs/REMOTE_CONTROL_SPEC.md` §3.

## The one rule

> The wire surface is every public `LuxarApp` method, minus
> `CONTROL_EXCLUDED_METHODS` (each with a recorded reason), plus
> `CONTROL_WIRE_ONLY_METHODS` (`subscribe` / `unsubscribe`).

There is no second vocabulary to learn, and no second vocabulary to keep in
sync. `method-policy.ts` states the rule, and
`src/tests/unit/core/app/control/method-policy.test.ts` scans `core/app.ts` and
fails if any public member is in neither list — so the next embedder method
cannot quietly land outside the policy.

## Files

| File                | What it owns                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------- |
| `method-policy.ts`  | Which methods a controller may call, and why the rest are refused.                        |
| `control-client.ts` | Socket lifecycle, request dispatch, event forwarding, reconnect.                          |
| `wire-values.ts`    | Turning viewer values into JSON — `Error` → `{name, message}`, `Blob` → `{mime, base64}`. |

Framing itself lives in `src/utils/json-rpc.ts`, which is transport-free and
shared with anything else that needs to speak the protocol.

## Things worth knowing before you change this

**`?control` is a bare flag.** The hub rides on the app that served the page, so
the socket address is derived from `location`. That keeps it working behind an
origin-rooted reverse proxy, under `luxar export` and in the native launcher. A
path-prefixed proxy uses an explicit path such as `?control=/exhibit/control`.
`?control=<url>` exists for a split origin and is same-origin-only unless
`?controlAllowCrossOrigin` is also given — see `normalizeControlSocketUrl` in
`src/config/url-params.ts` for why that matters.

**`params` are positional.** JSON-RPC allows named parameters; we do not use
them. A named form needs a table mapping every method's parameter names,
maintained on both sides of the wire, and that table drifts the first time a
signature changes.

**Events are attached eagerly, at construction.** Not laziness: the viewer only
provisions picking if a `selection` / `element-*` listener exists _at
dataset-load time_. A client that waited for a `subscribe` call would leave
those three events permanently dead with nothing to explain it. So the client
listens to everything from the start, and `subscribe` only gates _forwarding_.
The cost is that picking runs for any scene loaded with `?control` present —
opt-in by construction.

**`camera-changed` is throttled.** It fires at frame rate, and an auto-rotating
kiosk never stops moving. Unthrottled it would crowd out the taps that matter.

**`dispose` is not on the wire.** It tears the viewer down in one frame and
`init` is not exposed, so there would be no way back — a single frame would
blank an exhibit until someone power-cycles it. Every other verb can be undone
by sending another one.

**`switchDataset` is on the wire, and the dispatcher validates its argument.**
`LuxarApp.switchDataset` validates nothing; the in-process caller is the
standalone bootstrap, which already ran `?src` through `normalizeDataSourceUrl`.
A controller has not, so the boundary does it here. Without that step any peer
that can reach the socket could hand the display a `file:` or `javascript:` URL.

**The hub is open unless `--control-token` is set.** Without a token, browser
sockets must have the same host as the hub; non-browser clients send no
`Origin`. A valid token is also the explicit allowance for split-origin and
proxied deployments. The designed deployment is a LAN the operator owns. That
is a deliberate choice, and it is the reason the two paragraphs above exist.
