/**
 * JSON-RPC 2.0 framing — pure, transport-free, side-effect-free.
 *
 * The remote-control channel (`core/app/control/control-client.ts`) speaks
 * JSON-RPC 2.0 over a WebSocket, and the hub that relays it
 * (`luxar/cli/control_hub.py`) speaks the same. This module owns the *shape* of
 * those frames and nothing else: no socket, no dispatch, no knowledge of the
 * embedder API. That split is what lets the decode rules be exhaustively
 * unit-tested without a fake socket.
 *
 * Two deliberate narrowings of the spec, both because the wire is ours and a
 * smaller surface is a smaller thing to get wrong:
 *
 * - **`params` is positional only.** JSON-RPC permits either an array or an
 *   object; we take the array. A named-parameter form would need a table
 *   mapping each method's parameter names, maintained on both sides of the
 *   wire, and that table drifts the first time a TypeScript signature changes.
 *   Positional params follow mechanically from the signature instead.
 * - **No batching.** A batch frame decodes as malformed. Nothing in the control
 *   surface needs it, and it would make reply routing a fan-in problem.
 *
 * @see docs/guides/specs/REMOTE_CONTROL_SPEC.md §3.2
 */

import {
  INTERNAL_ERROR,
  INVALID_REQUEST,
  JSONRPC_VERSION,
  METHOD_NOT_FOUND,
  PARSE_ERROR,
} from '../config/control-contract';

// Re-exported from the GENERATED wire contract rather than written here.
// Three implementations have to agree on these codes — this client, the Python
// hub, and the Go launcher's relay — and a drift gate
// (`hatch run check-control-contract`) is what makes the agreement checkable.
// The names keep their `JSON_RPC_` prefix because that is what this module's
// callers already import.

/** The only protocol version accepted or emitted. */
export const JSON_RPC_VERSION = JSONRPC_VERSION;

/** Invalid JSON was received. */
export const JSON_RPC_PARSE_ERROR = PARSE_ERROR;
/** Well-formed JSON that is not a valid request object. */
export const JSON_RPC_INVALID_REQUEST = INVALID_REQUEST;
/** The method does not exist, or policy refuses to expose it. */
export const JSON_RPC_METHOD_NOT_FOUND = METHOD_NOT_FOUND;
/**
 * The method exists but the params do not fit it.
 *
 * NOT in the shared contract: nothing in this system emits it. It is here so a
 * reader of a foreign peer's error frame can name the code, and putting it in
 * the contract would imply the hub and relay handle it.
 */
export const JSON_RPC_INVALID_PARAMS = -32602;
/** The method threw. */
export const JSON_RPC_INTERNAL_ERROR = INTERNAL_ERROR;

/**
 * A request identifier. `null` is legal in the spec but we never emit it as a
 * request id — a frame without an id is a notification, which is the clearer
 * way to say "no reply wanted".
 */
export type JsonRpcId = string | number;

/** The error member of a failure response. */
export interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

/** A decoded frame, discriminated so a caller cannot forget a case. */
export type DecodedFrame =
  /** A call expecting a reply on `id`. */
  | { kind: 'request'; id: JsonRpcId; method: string; params: unknown[] }
  /** A call expecting no reply. */
  | { kind: 'notification'; method: string; params: unknown[] }
  /** A reply to something we sent. Exactly one of `result` / `error` is set. */
  | { kind: 'response'; id: JsonRpcId; result?: unknown; error?: JsonRpcError }
  /** Undecodable. Echo `id` when the offending request identified itself. */
  | { kind: 'malformed'; id: JsonRpcId | null; code: number; message: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isId(value: unknown): value is JsonRpcId {
  // Integers and strings only. A float id round-trips badly through languages
  // that distinguish int from float, and the hub remaps ids to its own anyway.
  return typeof value === 'string' || (typeof value === 'number' && Number.isInteger(value));
}

/**
 * Normalize a frame's `params` to a positional array.
 *
 * Absent params mean "no arguments". An object is refused rather than coerced:
 * silently treating `{index: 0}` as no arguments would make a mis-shaped call
 * look like a successful one.
 */
function readParams(raw: unknown): unknown[] | null {
  if (raw === undefined || raw === null) return [];
  return Array.isArray(raw) ? raw : null;
}

/**
 * Decode one text frame.
 *
 * Never throws: every rejection comes back as `kind: 'malformed'` carrying the
 * code a caller should answer with. A control endpoint that threw on garbage
 * would tear down the socket over a single bad frame.
 */
export function decodeFrame(raw: string): DecodedFrame {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { kind: 'malformed', id: null, code: JSON_RPC_PARSE_ERROR, message: 'parse error' };
  }
  if (!isRecord(parsed)) {
    return {
      kind: 'malformed',
      id: null,
      code: JSON_RPC_INVALID_REQUEST,
      message: 'frame must be an object',
    };
  }
  if (parsed.jsonrpc !== JSON_RPC_VERSION) {
    return {
      kind: 'malformed',
      id: isId(parsed.id) ? parsed.id : null,
      code: JSON_RPC_INVALID_REQUEST,
      message: `frame must declare jsonrpc "${JSON_RPC_VERSION}"`,
    };
  }
  if ('result' in parsed || 'error' in parsed) return decodeResponse(parsed);
  return decodeCall(parsed);
}

function decodeResponse(frame: Record<string, unknown>): DecodedFrame {
  if (!isId(frame.id)) {
    return {
      kind: 'malformed',
      id: null,
      code: JSON_RPC_INVALID_REQUEST,
      message: 'response needs an id',
    };
  }
  if ('error' in frame) {
    const error = frame.error;
    if (!isRecord(error) || typeof error.code !== 'number' || typeof error.message !== 'string') {
      return {
        kind: 'malformed',
        id: frame.id,
        code: JSON_RPC_INVALID_REQUEST,
        message: 'error member must carry a numeric code and a message',
      };
    }
    return {
      kind: 'response',
      id: frame.id,
      error: { code: error.code, message: error.message, data: error.data },
    };
  }
  return { kind: 'response', id: frame.id, result: frame.result };
}

function decodeCall(frame: Record<string, unknown>): DecodedFrame {
  if (typeof frame.method !== 'string' || frame.method.length === 0) {
    return {
      kind: 'malformed',
      id: isId(frame.id) ? frame.id : null,
      code: JSON_RPC_INVALID_REQUEST,
      message: 'method must be a non-empty string',
    };
  }
  const params = readParams(frame.params);
  if (params === null) {
    return {
      kind: 'malformed',
      id: isId(frame.id) ? frame.id : null,
      code: JSON_RPC_INVALID_PARAMS,
      message: 'params must be an array (positional)',
    };
  }
  // No id at all, or an explicit null, both mean "no reply wanted".
  if (frame.id === undefined || frame.id === null) {
    return { kind: 'notification', method: frame.method, params };
  }
  if (!isId(frame.id)) {
    return {
      kind: 'malformed',
      id: null,
      code: JSON_RPC_INVALID_REQUEST,
      message: 'id must be a string or an integer',
    };
  }
  return { kind: 'request', id: frame.id, method: frame.method, params };
}

/** Encode a request expecting a reply on `id`. */
export function requestFrame(id: JsonRpcId, method: string, params: unknown[] = []): string {
  return JSON.stringify({ jsonrpc: JSON_RPC_VERSION, id, method, params });
}

/**
 * Encode a call expecting no reply.
 *
 * `params` is positional here too, including for the `event` notification the
 * viewer pushes — `['dimensions-changed', payload]`, not
 * `{name, payload}` — so encode and decode obey one rule rather than two.
 */
export function notificationFrame(method: string, params: unknown[] = []): string {
  return JSON.stringify({ jsonrpc: JSON_RPC_VERSION, method, params });
}

/** Encode a successful reply. `undefined` results serialize as `null`. */
export function successFrame(id: JsonRpcId, result: unknown): string {
  return JSON.stringify({ jsonrpc: JSON_RPC_VERSION, id, result: result ?? null });
}

/**
 * Encode a failure reply. `id` may be `null` when the offending frame's id
 * could not be read (a parse error) — the one place a null id is correct.
 */
export function errorFrame(
  id: JsonRpcId | null,
  code: number,
  message: string,
  data?: unknown
): string {
  const error: JsonRpcError = { code, message };
  if (data !== undefined) error.data = data;
  return JSON.stringify({ jsonrpc: JSON_RPC_VERSION, id, error });
}
