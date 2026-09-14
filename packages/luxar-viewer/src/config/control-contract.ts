/**
 * GENERATED FILE — DO NOT EDIT.
 *
 * Projected from control-contract/contract.yaml by
 * scripts/gen_control_contract.py. Edit the YAML and regenerate; a drift
 * gate (`hatch run check-control-contract`) fails the build if this file
 * and the contract disagree.
 */

/** Roles a socket may declare as `?role=`. */
export const CONTROL_ROLES = {
  viewer: 'viewer',
  controller: 'controller',
} as const;

export type ControlRole = (typeof CONTROL_ROLES)[keyof typeof CONTROL_ROLES];

/** Query-parameter names on the socket URL. */
export const CONTROL_PARAMS = {
  role: 'role',
  token: 'token',
} as const;

/** WebSocket close codes. */
export const CLOSE_POLICY_VIOLATION = 1008;

/** JSON-RPC. */
export const JSONRPC_VERSION = '2.0';
export const EVENT_METHOD = 'event';
export const PARSE_ERROR = -32700;
export const INVALID_REQUEST = -32600;
export const METHOD_NOT_FOUND = -32601;
export const INTERNAL_ERROR = -32603;
export const NO_VIEWER = -32001;

/** Limits bounding what an untrusted peer can make us hold. */
export const MAX_PENDING_PER_CONTROLLER = 64;
export const MAX_BUFFERED_EVENTS = 1024;
export const MAX_FRAME_BYTES = 16777216;
