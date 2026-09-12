/**
 * The controller half of the remote-control channel.
 *
 * `core/app/control/control-client.ts` is the party being *driven*; this is the
 * party doing the driving. Same wire format, same framing module, opposite
 * role: it sends requests and receives replies and events.
 *
 * Kept separate from the viewer's client rather than generalised into one
 * class, because the two have almost nothing in common at runtime. The client
 * dispatches onto a live app and never asks anything; the controller tracks
 * outstanding promises and never answers anything. A merged class would be two
 * implementations behind one name.
 */

import { NO_VIEWER } from '../../config/control-contract';
import {
  decodeFrame,
  notificationFrame,
  requestFrame,
  type JsonRpcError,
  type JsonRpcId,
} from '../../utils/json-rpc';
import { log, Modules } from '../../utils/log';
import { CLOSE_POLICY_VIOLATION } from '../../config/control-contract';
import type { ControlSocketLike } from '../app/control/control-client';

/** First reconnect delay after an unexpected close. */
export const CONTROLLER_RECONNECT_BASE_MS = 500;
/** Ceiling for the doubling reconnect delay. */
export const CONTROLLER_RECONNECT_MAX_MS = 15_000;
/**
 * How long a request waits for its reply.
 *
 * Generous because the answer can be behind real work — `flyTo` resolves when
 * the flight lands — but bounded, because a promise that never settles is a
 * panel tile that stays greyed out forever with no explanation.
 */
export const CONTROLLER_CALL_TIMEOUT_MS = 30_000;

/**
 * Connection state, for a status line the visitor can act on.
 *
 * `refused` is terminal and separate from `closed` on purpose: the hub accepts
 * a handshake it means to refuse and then closes it with
 * `CLOSE_POLICY_VIOLATION`, so a wrong token, an unknown role or a foreign
 * origin arrive here as a close like any other. Folding them into `closed`
 * reported a mistyped token as "waiting for the display" and retried it
 * forever — against a hub that will refuse every attempt for the same reason.
 */
export type ControllerStatus = 'connecting' | 'open' | 'closed' | 'refused';

/** A rejected call, carrying the JSON-RPC code so a caller can branch on it. */
export class ControllerCallError extends Error {
  readonly code: number;

  constructor(error: JsonRpcError) {
    super(error.message);
    this.name = 'ControllerCallError';
    this.code = error.code;
  }

  /** The hub's "nothing to drive" answer: wait for it, do not fail on it. */
  get noViewerAttached(): boolean {
    // From the generated wire contract, not a literal: the Python hub and the
    // Go relay both emit this code, and a panel that stopped recognising it
    // would show "waiting for the display" forever.
    return this.code === NO_VIEWER;
  }
}

export interface ControllerSocketPorts {
  /** Hub URL, already validated by `normalizeControlSocketUrl`. */
  url: string;
  /** Shared secret to present as `?token=`, or null for an open hub. */
  token: string | null;
  /** An event pushed by the viewer. */
  onEvent?: (name: string, payload: unknown) => void;
  /** Connection state changed. */
  onStatus?: (status: ControllerStatus) => void;
  /** Socket factory, injectable for tests. */
  openSocket?: (url: string) => ControlSocketLike;
}

/** Build the URL dialled: the hub plus the two query parameters it reads. */
export function buildControllerSocketUrl(url: string, token: string | null): string {
  const target = new URL(url);
  target.searchParams.set('role', 'controller');
  if (token !== null && token.length > 0) target.searchParams.set('token', token);
  return target.href;
}

export class ControllerSocket {
  private readonly ports: ControllerSocketPorts;
  private socket: ControlSocketLike | null = null;
  private nextId = 1;
  private readonly pending = new Map<
    JsonRpcId,
    {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
      timer: ReturnType<typeof setTimeout>;
    }
  >();
  private reconnectDelayMs = CONTROLLER_RECONNECT_BASE_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  private disposed = false;

  constructor(ports: ControllerSocketPorts) {
    this.ports = ports;
  }

  /** Dial the hub, and keep dialling if it goes away. */
  connect(): void {
    if (this.disposed || this.socket !== null) return;
    this.ports.onStatus?.('connecting');
    const url = buildControllerSocketUrl(this.ports.url, this.ports.token);
    const open =
      this.ports.openSocket ??
      ((target: string) => new WebSocket(target) as unknown as ControlSocketLike);
    let socket: ControlSocketLike;
    try {
      socket = open(url);
    } catch (error) {
      log.warning(Modules.APP, 'control panel: could not open the socket:', error);
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      this.reconnectDelayMs = CONTROLLER_RECONNECT_BASE_MS;
      this.ports.onStatus?.('open');
    };
    socket.onmessage = (event) => {
      if (typeof event.data === 'string') this.handleFrame(event.data);
    };
    socket.onerror = () => log.warning(Modules.APP, 'control panel: socket error');
    socket.onclose = (event) => {
      this.socket = null;
      const refused = event?.code === CLOSE_POLICY_VIOLATION;
      this.ports.onStatus?.(refused ? 'refused' : 'closed');
      // Every outstanding promise is now unanswerable; settling them is the
      // difference between a panel that says "disconnected" and one that hangs.
      this.rejectAll(
        new Error(refused ? 'the hub refused this panel' : 'the control socket closed')
      );
      // A refusal is about who we are, not about the network, so retrying it
      // cannot succeed and only keeps the hub busy refusing us.
      if (!this.disposed && !refused) this.scheduleReconnect();
    };
  }

  /** Invoke a viewer method and wait for its result. */
  call(method: string, params: unknown[] = []): Promise<unknown> {
    if (this.socket === null) {
      return Promise.reject(new Error(`not connected; cannot call ${method}`));
    }
    const id = this.nextId;
    this.nextId += 1;
    return new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method} did not answer within ${CONTROLLER_CALL_TIMEOUT_MS}ms`));
      }, CONTROLLER_CALL_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      this.send(requestFrame(id, method, params));
    });
  }

  /**
   * Invoke a method without waiting.
   *
   * What a tile tap uses: the visitor should see the display move, not wait for
   * a round trip, and the authoritative position arrives as an event anyway.
   */
  notify(method: string, params: unknown[] = []): void {
    this.send(notificationFrame(method, params));
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.reconnectTimer !== undefined) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = undefined;
    this.rejectAll(new Error('the controller was disposed'));
    const socket = this.socket;
    this.socket = null;
    try {
      socket?.close();
    } catch {
      // Already gone is the state we wanted.
    }
  }

  private send(text: string): void {
    try {
      this.socket?.send(text);
    } catch (error) {
      log.warning(Modules.APP, 'control panel: send failed:', error);
    }
  }

  private rejectAll(error: Error): void {
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.reject(error);
    }
    this.pending.clear();
  }

  private scheduleReconnect(): void {
    if (this.disposed || this.reconnectTimer !== undefined) return;
    const delay = this.reconnectDelayMs;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined;
      this.connect();
    }, delay);
    this.reconnectDelayMs = Math.min(delay * 2, CONTROLLER_RECONNECT_MAX_MS);
  }

  private handleFrame(raw: string): void {
    const frame = decodeFrame(raw);
    if (frame.kind === 'response') {
      this.settle(frame.id, frame.result, frame.error);
      return;
    }
    if (frame.kind === 'notification' && frame.method === 'event') {
      // Positional, matching the rest of the wire: ['name', payload].
      const [name, payload] = frame.params;
      if (typeof name === 'string') this.ports.onEvent?.(name, payload);
      return;
    }
    if (frame.kind === 'malformed') {
      log.warning(Modules.APP, `control panel: unreadable frame (${frame.message})`);
    }
  }

  private settle(id: JsonRpcId, result: unknown, error: JsonRpcError | undefined): void {
    const entry = this.pending.get(id);
    if (entry === undefined) return; // a late reply to a timed-out call
    this.pending.delete(id);
    clearTimeout(entry.timer);
    if (error !== undefined) entry.reject(new ControllerCallError(error));
    else entry.resolve(result);
  }
}
