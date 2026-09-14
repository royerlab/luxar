"""GENERATED FILE — DO NOT EDIT.

Projected from control-contract/contract.yaml by
scripts/gen_control_contract.py. Edit the YAML and regenerate; a drift
gate (`hatch run check-control-contract`) fails the build if this file
and the contract disagree."""

from __future__ import annotations

# Roles a socket may declare as ?role=.
ROLE_VIEWER = "viewer"
ROLE_CONTROLLER = "controller"
ROLES = frozenset({ROLE_VIEWER, ROLE_CONTROLLER})

# Query-parameter names on the socket URL.
PARAM_ROLE = "role"
PARAM_TOKEN = "token"

# WebSocket close codes.
CLOSE_POLICY_VIOLATION = 1008

# JSON-RPC.
JSONRPC_VERSION = "2.0"
EVENT_METHOD = "event"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603
NO_VIEWER = -32001

# Limits bounding what an untrusted peer can make us hold.
MAX_PENDING_PER_CONTROLLER = 64
MAX_BUFFERED_EVENTS = 1024
MAX_FRAME_BYTES = 16777216
