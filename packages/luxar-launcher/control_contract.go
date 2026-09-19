// GENERATED FILE — DO NOT EDIT.
//
// Projected from control-contract/contract.yaml by
// scripts/gen_control_contract.py. Edit the YAML and regenerate; a drift
// gate (`hatch run check-control-contract`) fails the build if this file
// and the contract disagree.

package main

// Roles a socket may declare as ?role=.
const (
	RoleViewer     = "viewer"
	RoleController = "controller"
)

// Query-parameter names on the socket URL.
const (
	ParamRole  = "role"
	ParamToken = "token"
)

// WebSocket close codes.
const (
	ClosePolicyViolation = 1008
)

// JSON-RPC.
const (
	JSONRPCVersion     = "2.0"
	EventMethod        = "event"
	CodeParseError     = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInternalError  = -32603
	CodeNoViewer       = -32001
)

// Limits bounding what an untrusted peer can make us hold.
const (
	MaxPendingPerController = 64
	MaxBufferedEvents       = 1024
	MaxFrameBytes           = 16777216
)
