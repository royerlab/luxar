/**
 * The viewer's side of the remote-control channel.
 *
 * Activated by `?control` (see `config/url-params.ts`), this attaches to the
 * hub as a **viewer** and does exactly one thing: dispatch incoming JSON-RPC
 * method calls onto the live `LuxarApp`, and push subscribed events back. It
 * has no idea what any method *does*, which is what keeps the wire surface and
 * the embedder API one thing rather than two.
 *
 * Everything it needs arrives as ports — `invoke`, `subscribe`, an
 * `EventGroup`, and a socket factory — so the whole dispatch and reconnect
 * story is unit-testable against a fake socket and a fake app, with no browser
 * and no WebGL context. Same shape as `WaypointDriver`.
 *
 * @see docs/guides/specs/REMOTE_CONTROL_SPEC.md §3.3
 */

import { normalizeDataSourceUrl } from '../../../config/url-params';
import type { EventGroup } from '../../../utils/cross-layer/event-group';
import {
  decodeFrame,
  errorFrame,
  notificationFrame,
  successFrame,
  JSON_RPC_INTERNAL_ERROR,
  JSON_RPC_INVALID_PARAMS,
  JSON_RPC_METHOD_NOT_FOUND,
  type JsonRpcId,
} from '../../../utils/json-rpc';
import { log, Modules } from '../../../utils/log';
import {
  CONTROL_SUBSCRIBE,
  CONTROL_UNSUBSCRIBE,
  controlRefusalReason,
  isControlMethodAllowed,
} from './method-policy';
import { encodeBlobForWire, isBlobLike, sanitizeForWire } from './wire-values';

/** First reconnect delay after an unexpected close. */
export const CONTROL_RECONNECT_BASE_MS = 500;
/** Ceiling for the doubling reconnect delay. */
export const CONTROL_RECONNECT_MAX_MS = 30_000;
/**
 * Minimum gap between forwarded `camera-changed` events (50 ms = 20 Hz).
 *
 * That event fires at frame rate whenever the camera moves, and an
 * auto-rotating kiosk means it never stops. Unthrottled it would saturate the
 * socket with frames no controller can use, and starve the taps that matter.
 */
export const CONTROL_CAMERA_EVENT_MIN_INTERVAL_MS = 50;

/**
 * Events the client attaches to eagerly, at construction.
 *
 * Eager attachment is load-bearing, not laziness: the viewer only provisions
 * its picking pipeline if a `selection` / `element-*` listener exists **at
 * dataset-load time** (`core/app.ts`, `hasSelectionConsumer`). A client that
 * waited for a `subscribe` call would leave those three events silently dead
 * forever, with no error to explain it. So the client listens to everything
 * from the start and `subscribe` / `unsubscribe` only gate *forwarding*.
 *
 * The cost is that picking runs for any scene loaded with `?control` present.
 * That is opt-in by construction — no `?control`, no client, no cost.
 */
export const CONTROL_FORWARDED_EVENTS: readonly string[] = [
  'camera-changed',
  'dataset-error',
  'dataset-fault',
  'dataset-loaded',
  'dimensions-changed',
  'element-click',
  'element-contextmenu',
  'selection',
  'sound-ended',
  'sound-started',
  'waypoint-arrived',
  'waypoint-departed',
];

/**
 * The subset of `WebSocket` this client uses, so a test can supply a fake.
 *
 * Only the event payloads this client reads are declared. A real `WebSocket`
 * therefore satisfies this structurally except for its handler arity, which
 * the module-private `openPlatformSocket` bridges.
 */
export interface ControlSocketLike {
  send(data: string): void;
  close(): void;
  onopen: (() => void) | null;
  onclose: ((event: { code?: number; reason?: string }) => void) | null;
  onerror: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
}

/** Everything the client needs from the app around it. */
export interface ControlClientPorts {
  /** Resolved socket URL (already validated by `normalizeControlSocketUrl`). */
  socketUrl: string;
  /** Shared secret to present as `?token=`, or null for an open hub. */
  token: string | null;
  /**
   * Call an embedder method. May return a value or a promise, and may throw —
   * `LuxarApp` throws `... called before init()` for most methods, and that
   * error is the honest answer to send back.
   */
  invoke: (method: string, params: unknown[]) => unknown;
  /** Attach an embedder event listener, returning its unsubscribe. */
  subscribe: (event: string, listener: (payload: unknown) => void) => () => void;
  /** Teardown registry. The client registers its socket and timers here. */
  events: EventGroup;
  /** Socket factory. Defaults to the platform `WebSocket`. */
  openSocket?: (url: string) => ControlSocketLike;
  /** Clock, injectable so a test can drive the camera throttle. */
  now?: () => number;
}

/**
 * Build the URL actually dialled: the validated address plus the two query
 * parameters the hub reads. `role` is ours to set — a client of this module is
 * always the party being driven.
 */
export function buildViewerSocketUrl(socketUrl: string, token: string | null): string {
  const url = new URL(socketUrl);
  url.searchParams.set('role', 'viewer');
  if (token !== null && token.length > 0) url.searchParams.set('token', token);
  return url.href;
}

/**
 * Open a real browser WebSocket as a {@link ControlSocketLike}.
 *
 * The narrowing is a one-line cast rather than a forwarding wrapper: the DOM
 * handler types differ only in that they receive an `Event` this client never
 * reads, and a wrapper would need a setter per handler to stay assignable.
 */
function openPlatformSocket(url: string): ControlSocketLike {
  return new WebSocket(url) as unknown as ControlSocketLike;
}

export class ControlClient {
  private readonly ports: ControlClientPorts;
  private socket: ControlSocketLike | null = null;
  private reconnectDelayMs = CONTROL_RECONNECT_BASE_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  private disposed = false;
  /** Event names currently being pushed to controllers. */
  private readonly forwarding = new Set<string>();
  private lastCameraEventAt = 0;

  constructor(ports: ControlClientPorts) {
    this.ports = ports;
    this.attachEventListeners();
    this.ports.events.add(() => this.dispose());
  }

  /** Dial the hub. Safe to call once; reconnects are scheduled internally. */
  connect(): void {
    if (this.disposed || this.socket !== null) return;
    const open = this.ports.openSocket ?? openPlatformSocket;
    let socket: ControlSocketLike;
    try {
      const url = buildViewerSocketUrl(this.ports.socketUrl, this.ports.token);
      socket = open(url);
    } catch (error) {
      log.warning(Modules.APP, 'control: could not open the socket:', error);
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      this.reconnectDelayMs = CONTROL_RECONNECT_BASE_MS;
      log.info(Modules.APP, `control: attached to ${this.ports.socketUrl}`);
    };
    socket.onmessage = (event) => {
      if (typeof event.data === 'string') this.handleFrame(event.data);
    };
    socket.onerror = () => {
      // `close` always follows, and that is where reconnect is scheduled.
      log.warning(Modules.APP, 'control: socket error');
    };
    socket.onclose = (event) => {
      this.socket = null;
      if (this.disposed) return;
      const delay = this.reconnectDelayMs;
      if (event.code !== 1000) {
        const reason = event.reason ? `: ${event.reason}` : '';
        log.warning(
          Modules.APP,
          `control: socket closed (${event.code ?? 'unknown'}${reason}); retrying in ${delay} ms`
        );
      }
      this.scheduleReconnect();
    };
  }

  /** Stop forwarding, close the socket, cancel any pending reconnect. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.forwarding.clear();
    if (this.reconnectTimer !== undefined) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = undefined;
    const socket = this.socket;
    this.socket = null;
    try {
      socket?.close();
    } catch {
      // A socket that is already gone is exactly the state we want.
    }
  }

  private scheduleReconnect(): void {
    if (this.disposed || this.reconnectTimer !== undefined) return;
    const delay = this.reconnectDelayMs;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined;
      this.connect();
    }, delay);
    this.reconnectDelayMs = Math.min(delay * 2, CONTROL_RECONNECT_MAX_MS);
  }

  private send(text: string): void {
    try {
      this.socket?.send(text);
    } catch (error) {
      log.warning(Modules.APP, 'control: send failed:', error);
    }
  }

  // ── inbound ──────────────────────────────────────────────────────────────

  private handleFrame(raw: string): void {
    const frame = decodeFrame(raw);
    if (frame.kind === 'malformed') {
      this.send(errorFrame(frame.id, frame.code, frame.message));
      return;
    }
    // A controller never asks the viewer a question, so a response arriving
    // here is a confused peer. Ignoring it is the whole handling.
    if (frame.kind === 'response') return;
    const id = frame.kind === 'request' ? frame.id : null;
    void this.invokeMethod(frame.method, frame.params, id);
  }

  private async invokeMethod(
    method: string,
    params: unknown[],
    id: JsonRpcId | null
  ): Promise<void> {
    if (method === CONTROL_SUBSCRIBE || method === CONTROL_UNSUBSCRIBE) {
      this.applySubscription(method, params, id);
      return;
    }
    if (!isControlMethodAllowed(method)) {
      this.reply(id, null, {
        code: JSON_RPC_METHOD_NOT_FOUND,
        message: controlRefusalReason(method),
      });
      return;
    }
    const guarded = guardParams(method, params);
    if (guarded.error !== undefined) {
      this.reply(id, null, { code: JSON_RPC_INVALID_PARAMS, message: guarded.error });
      return;
    }
    try {
      const result = await this.ports.invoke(method, guarded.params);
      const wire = isBlobLike(result) ? await encodeBlobForWire(result) : sanitizeForWire(result);
      this.reply(id, wire);
    } catch (error) {
      // The app's own message is the useful one ("...called before init()",
      // "unknown layer '/x'"), and a throw must never reach the socket's
      // handler, where it would look like a transport failure.
      this.reply(id, null, {
        code: JSON_RPC_INTERNAL_ERROR,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private applySubscription(method: string, params: unknown[], id: JsonRpcId | null): void {
    const name = params[0];
    if (typeof name !== 'string' || !CONTROL_FORWARDED_EVENTS.includes(name)) {
      this.reply(id, null, {
        code: JSON_RPC_INVALID_PARAMS,
        message: `not a forwardable event: ${JSON.stringify(name)}`,
      });
      return;
    }
    if (method === CONTROL_SUBSCRIBE) this.forwarding.add(name);
    else this.forwarding.delete(name);
    this.reply(id, true);
  }

  private reply(
    id: JsonRpcId | null,
    result: unknown,
    failure?: { code: number; message: string }
  ): void {
    // A notification asked for no answer; an error still deserves a log so a
    // fire-and-forget mistake is not invisible.
    if (id === null) {
      if (failure !== undefined) log.warning(Modules.APP, `control: ${failure.message}`);
      return;
    }
    this.send(
      failure === undefined
        ? successFrame(id, result)
        : errorFrame(id, failure.code, failure.message)
    );
  }

  // ── outbound events ──────────────────────────────────────────────────────

  private attachEventListeners(): void {
    for (const name of CONTROL_FORWARDED_EVENTS) {
      const unsubscribe = this.ports.subscribe(name, (payload) => this.forwardEvent(name, payload));
      this.ports.events.add(unsubscribe);
    }
  }

  private forwardEvent(name: string, payload: unknown): void {
    if (this.disposed || !this.forwarding.has(name)) return;
    if (name === 'camera-changed' && !this.cameraEventDue()) return;
    this.send(notificationFrame('event', [name, sanitizeForWire(payload)]));
  }

  private cameraEventDue(): boolean {
    const now = (this.ports.now ?? Date.now)();
    if (now - this.lastCameraEventAt < CONTROL_CAMERA_EVENT_MIN_INTERVAL_MS) return false;
    this.lastCameraEventAt = now;
    return true;
  }
}

/**
 * Per-method argument checks that belong at the wire boundary.
 *
 * Only one method needs one today, and it needs it badly:
 * `LuxarApp.switchDataset` validates nothing, so without this a LAN peer could
 * hand the display a `file:` or `javascript:` URL. The in-process caller is the
 * standalone bootstrap, which has already run every `?src` through
 * `normalizeDataSourceUrl`; a controller has not, so the wire does it here.
 */
function guardParams(method: string, params: unknown[]): { params: unknown[]; error?: string } {
  if (method !== 'switchDataset') return { params };
  const raw = params[0];
  if (typeof raw !== 'string') return { params, error: 'switchDataset needs a URL string' };
  const src = normalizeDataSourceUrl(raw);
  if (src === null) return { params, error: `refused dataset URL: ${raw}` };
  return { params: [src, ...params.slice(1)] };
}
