# Utils Package

Cross-cutting utility functions and helpers used throughout the Luxar viewer. This is the foundation layer: every other layer (data, rendering, controls, UI) may import from here, and nothing here may import upward. The package collects domain-specific helpers (HDR color, geometry-buffer accounting, camera-type helpers) alongside cross-layer plumbing (typed event bus, notifier facade, EventGroup) and small primitives (`clamp`, `Result`, `escapeHtml`).

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Modules](#modules)
- [Console Interception](#console-interception)
- [HDR Detection](#hdr-detection)
- [Structured Logging](#structured-logging)
- [Cross-Layer Plumbing](#cross-layer-plumbing)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)

## Overview

`utils/` is the bottom of the viewer's layer order. Lower layers (`data`, `scene`, `input`) that need to reach into the UI (toasts, panel toggles, FPS readouts) do so through the dependency-inversion targets defined here — `notifier` (single backend) and `eventBus` (typed pub/sub) — instead of importing UI modules directly. Pure helpers (clamp, camera math, HDR conversion, etc.) live here as well so any layer can use them without crossing layer boundaries.

**Core Philosophy**: Stable, side-effect-free helpers and well-typed cross-layer seams. Module load must not patch globals; explicit calls (e.g. `consoleInterceptor.patch()`, `setNotifierBackend(...)`) opt in to runtime effects.

## Key Features

- **Console Interception**: Ring buffer system for capturing all browser console output (opt-in patching)
- **HDR Display Detection**: CSS-media-query probes plus pure decision logic over an `HDRCapabilities` snapshot
- **HDR Color Conversion**: Linear sRGB to BT.2020 PQ I420P10 pipeline for 10-bit WebCodecs video encoding
- **Structured Logging**: `[emoji] [Module] message` format with a fixed `Modules` registry
- **Typed Event Bus**: Cross-layer pub/sub with per-event payload typing and optional last-payload replay
- **Notifier Facade**: Dependency-inverted UI notification surface (toast, error, help overlay, loading indicator)
- **EventGroup**: Group-scoped DOM event listener registration with one-call teardown
- **Result<T, E>**: Discriminated-union return type for fallible operations
- **WebGPU/WebGL2 Availability Probe**: Page-load-time backend detection (async + sync variants, cached)
- **Camera Type Helpers**: Unified `LuxarCamera` union and type guards for perspective vs orthographic
- **Effective Visibility**: `isEffectivelyVisible` — the single parent-chain walk answering "does this node actually render?" (`visible` is a LOCAL flag, so a hidden layer or a hidden LOD level leaves its descendants' flags true). Shared by the LOD load gate, LOD eviction, the pick pass, and the depth-sort scheduler
- **Platform Detection**: Single `isMacPlatform()` helper for OS-conditional defaults
- **HTML Escaping**: XSS prevention for safe HTML rendering
- **Storage Keys**: Single registry of `luxar.*` localStorage keys

## Architecture

```
utils/
├── camera-utils.ts          # LuxarCamera union, type guards, FOV/aspect helpers
├── clamp.ts                 # Generic numeric clamp (optional bounds)
├── console-interceptor.ts   # Ring-buffer console capture (Proxy singleton, opt-in patch)
├── escape-html.ts           # HTML entity escaping for safe rendering
├── format-error.ts          # Unknown thrown value → message / name: message / stack
├── log.ts                   # log object, Modules registry, LogEmoji, createModuleLogger
├── object-visibility.ts     # isEffectivelyVisible (ancestor-aware scene-graph visibility)
├── platform.ts              # isMacPlatform()
├── result.ts                # Result<T, E> + ok/err/isOk/isErr/match/mapOk/mapErr/unwrap/tryAsync
├── storage-keys.ts          # luxar.* localStorage key registry
├── viewer-container.ts      # mount-root registry (get/set/resetViewerContainer) + containing-block promotion
├── webgpu-availability.ts   # getRendererAPI (async) + getRendererAPISync
├── cross-layer/             # Cross-layer plumbing (typed bus, notifier facade, listener group)
│   ├── event-bus.ts         # Typed cross-layer pub/sub (LuxarEventMap, eventBus singleton)
│   ├── event-group.ts       # DOM-listener group with single dispose() teardown
│   └── notifier.ts          # Notifier facade + setNotifierBackend dependency inversion
└── hdr/                     # HDR display + color-conversion pipeline
    ├── hdr-color-conversion.ts  # Linear sRGB → BT.2020 PQ I420P10 (Uint16 planar)
    └── hdr-detection.ts         # CSS-media-query HDR/gamut probes + decision helpers
```

Geometry-byte accounting (`estimateGeometryBytes` / `invalidateCachedByteSize`) used to live here as `geometry-utils.ts`; it has moved to its only consumer at `rendering/gpu-buffer-pool/geometry-bytes.ts` (re-exported by `rendering/gpu-buffer-pool.ts` for the existing test import path).

Each module is focused on a specific domain with minimal dependencies. The only intra-`utils/` imports are `cross-layer/event-group.ts`, `cross-layer/notifier.ts`, and `hdr/hdr-detection.ts` → `log.ts`, and `hdr/hdr-color-conversion.ts` → `clamp.ts`.

## Modules

### console-interceptor.ts - Console Debugging

Ring buffer system for capturing and managing console output:

**Core Features**:

- **Lazy Proxy Singleton**: Importing `consoleInterceptor` is side-effect-free; patching is explicit via `.patch()`
- **Ring Buffer**: Circular buffer that wraps cleanly on the fill boundary. Capacity defaults to `DEFAULT_MAX_BUFFER_SIZE` (10,000) and is reconfigurable via `setMaxBufferSize(n)` so the bootstrap can push the canonical `config.ui.debugConsole.interceptor.maxBufferSize` value without a load-time config import (`getMaxBufferSize()` reads it back)
- **Listener Set**: Real-time callbacks (`addListener` / `removeListener`) for live UI consumption
- **Reversible**: `patch()` is idempotent (`isPatched` reports state); `dispose()` restores the original console methods; `disposeInstance()` resets the singleton between tests

### log.ts - Structured Logging

Consistent logging with emoji prefixes and module identification:

**Core Exports**:

- `log` — Quick logging object (`log.info`, `log.success`, `log.error`, `log.warning`, `log.load`, `log.update`, `log.query`, `log.data`, `log.custom`, `log.raw`)
- `Modules` — Fixed registry of module name constants (LUXAR, APP, MAIN, SCENE_LOADER, WORKER_POOL, WASM, RENDERER, HDR, EVENT_GROUP, …)
- `LogEmoji` — Standard emoji constants for log categories (status, actions, data, rendering, controls, UI, debug)
- `formatLog(emoji, module, message)` — Format a log message with the standard `[emoji] [Module] message` shape
- `createModuleLogger(module)` — Build a module-scoped logger that closes over `module`

### hdr-detection.ts - HDR Display Detection

Display-side capability detection via CSS media queries plus pure decision logic over an `HDRCapabilities` snapshot. Renderer-side probes (`gl.getExtension`, color-buffer bit depth) live in `rendering/renderer-capabilities.ts`; this module's defaults for `floatTextures` / `colorDepth` are placeholders that `createRendererCapabilities` overwrites.

**Core Functions**:

- `detectDisplayCapabilities()` — Build an `HDRCapabilities` from `window.matchMedia` (P3, Rec2020, dynamic-range, 10-bit color)
- `configureHDRRenderer(_renderer, capabilities)` — Logs detected capabilities only. It deliberately does NOT touch `outputColorSpace` or `toneMapping` — `PostProcessingManager` owns those (mega-shader applies tone mapping internally; host pins `outputColorSpace = SRGB`, `toneMapping = NoToneMapping`)
- `logHDRCapabilities(capabilities)` — Pretty-print a capabilities snapshot
- `isHDRDisplay(capabilities)` — True iff HDR + deep color + float textures + (P3 or Rec2020)
- `getOptimalRenderTargetType(capabilities)` — Returns `THREE.HalfFloatType` when float textures exist, else `UnsignedByteType`

### hdr-color-conversion.ts - HDR Color Conversion

Converts linear sRGB float RGBA pixels (from WebGL `readPixels`) to BT.2020 PQ YCbCr I420P10 for WebCodecs `VideoFrame` with HDR metadata. Pipeline: sRGB→BT.2020 linear (3×3 matrix), linear→PQ (SMPTE ST 2084), RGB→YCbCr (BT.2020 NCL), 10-bit limited-range quantization, 4:2:0 chroma subsampling, planar `Uint16Array` layout.

**Core Function**:

- `rgbaFloatToI420P10(rgba, width, height)` — Linear sRGB RGBA float to BT.2020 PQ YCbCr I420P10 `Uint16Array`

### format-error.ts - Error Formatting

Turning an unknown thrown value into something readable. `catch (error)` yields
`unknown`, and both obvious approaches fail in the same place: `String(error)`
loses a `DOMException`'s name, and `JSON.stringify(error)` yields `"{}"` because
`name` / `message` / `stack` are non-enumerable. The debug console's Error branch
uses these so a `log.*(…, error)` call keeps its message.

All three never throw — they run inside `catch` blocks and the patched
`console.*` methods, so a hostile accessor or Proxy trap (or the null-prototype
`String()` TypeError) must not break the very logging path reporting the error.

- `getErrorMessage(error)` — the message
- `formatErrorForDisplay(error)` — one-line `name: message`, name alone when the
  message is empty
- `getErrorStack(error)` — the stack, including duck-typed carriers, else
  `undefined`

### escape-html.ts - HTML Escaping

Simple XSS prevention utility:

- `escapeHtml(str)` — Escape HTML special characters (`&`, `<`, `>`, `"`, and `'` as `&#39;` for single-quoted-attribute defense-in-depth)

### camera-utils.ts - Camera Type Helpers

Union type and helpers so the codebase can treat perspective and orthographic cameras uniformly without scattering `instanceof` checks:

- `LuxarCamera` — `THREE.PerspectiveCamera | THREE.OrthographicCamera`
- `isPerspectiveCamera(camera)` / `isOrthographicCamera(camera)` — Type guards
- `getCameraFovRadians(camera)` — Vertical FOV in radians; perspective converts degrees → radians; orthographic returns 0
- `updateCameraAspect(camera, width, height)` — Resize the projection. Perspective updates `aspect`; orthographic rescales `left`/`right` to preserve the vertical extent
- `getOrthoFrustumHeight(camera)` — `(top − bottom) / zoom`

### clamp.ts - Generic Numeric Clamp

`clamp(value, min?, max?)` — Either bound may be omitted; both omitted is a pass-through. The canonical home for this primitive; `ui/gui/format/value-formatting.ts` re-exports it for back-compat (the original lived there and was unreachable from `rendering` under the dependency-cruiser layer order).

### event-bus.ts - Typed Cross-Layer Pub/Sub

Dependency-inversion target for cross-layer signals that would otherwise require lower layers to import upward.

**Core Exports**:

- `LuxarEventMap` — Catalog of cross-layer events with payload types:
  - Publisher events: `frame-start`, `frame-end`, `loading-progress`
  - Command events: `panel-toggle`, `panel-cycle`, `panel-hide`
- `TypedEventBus<EventMap>` — Surface: `on(type, listener, { replayLast? })`, `emit(type, payload)`, `clear(type?)`
- `eventBus` — Singleton bus typed against `LuxarEventMap`
- `createEventBus<EventMap>()` — Fresh bus, useful for test isolation or per-app-instance buses
- `Unsubscribe` — Function returned by `on(...)` that removes the subscription (idempotent)

The bus caches the last emitted payload per event so `on(..., { replayLast: true })` can fire synchronously for late-binding subscribers. Listener iteration uses a snapshot so subscribe/unsubscribe during a callback can't reorder the active loop.

### event-group.ts - Listener Group with Single-Call Teardown

`EventGroup` collects DOM event-listener registrations (and arbitrary cleanup callbacks) and tears them all down in LIFO order on `dispose()`. Replaces the error-prone "store a bound handler in a field, remember to remove it later" pattern.

- `on(target, type, listener, options?)` — Register a listener (overloads for `Window`, `Document`, `HTMLElement`, `EventTarget`). Returns an early-cancel function; if invoked it self-removes from the group's cleanup list so long-lived groups don't accumulate no-op closures.
- `add(cleanup)` — Register an arbitrary teardown callback (e.g. `observer.disconnect()`, `cancelAnimationFrame(handle)`)
- `dispose()` — Run every registered cleanup in reverse order; idempotent. A throwing cleanup logs via `log.error(Modules.EVENT_GROUP, ...)` and never stops siblings from running.
- `size` — Pending-cleanup count (for tests)

### notifier.ts - Cross-Layer Notification Facade

Dependency-inverted UI notification surface so lower layers can surface user-visible messages without importing the `ui/` helper modules directly.

- `NotifierBackend` — Interface a concrete backend implements (`showError`, `showToast`, `showHelpOverlay`, `hideHelpOverlay`, `showLoadingIndicator`, `hideLoadingIndicator`, `clearError`)
- `notifier` — Stable call surface: `error`, `toast`, `showHelp`, `hideHelp`, `showLoading`, `hideLoading`, `clearError`. Drops calls silently (with a single warn) when no backend is registered, so unit tests and early-startup paths don't crash.
- `setNotifierBackend(b)` — Called once by the UI bootstrap to plug in the concrete `ui/` helpers; subsequent calls replace the backend (useful for tests)
- `clearNotifierBackend()` — Tear down the backend; also resets the once-only missing-backend warning flag

### platform.ts - Platform Detection

- `isMacPlatform()` — `navigator.platform.startsWith('Mac')`; returns `false` in non-browser contexts. Centralized so tests can stub one export.

### result.ts - Result<T, E>

Discriminated-union return type for fallible operations. Disambiguates "missing optional thing" from "real failure" — `T | undefined` can't.

- `Result<T, E = string>` = `Ok<T> | Err<E>`
- `ok(value)` / `err(error)` — Constructors
- `isOk(r)` / `isErr(r)` — Type guards
- `match(r, { ok, err })` — Pattern-match to a value
- `mapOk(r, f)` / `mapErr(r, f)` — Functorial maps
- `unwrap(r)` — Throw on err (code-smell signal; prefer `match`)
- `unwrapOr(r, fallback)` — Default value on err
- `tryAsync(fn, mapError)` — Wrap a throw-based async function in `Result`

### storage-keys.ts - localStorage Key Registry

Single source of truth for every `localStorage` key the viewer touches. Keys are dot-namespaced under `luxar.*` so they can never collide with a host page's storage.

- `StorageKeys.theme` — Active theme id (`'dark' | 'light' | 'frosted-glass' | 'liquid-glass'`)
- `StorageKeys.debug` — Persisted debug-mode toggle (mirrors `?debug` URL param)
- `StorageKeys.settings` — Global viewer preferences (`'luxar.settings'`, the Settings popover; see `config/user-settings.ts`)
- `StorageKeys.rendering(sceneId)` — Per-scene rendering settings; segment is sanitized to `[a-zA-Z0-9-_]`

### viewer-container.ts - Mount-Root Registry

The single DOM element the viewer mounts all overlays, panels, toasts, dialogs, and injected SVG filters into. Defaults to `document.body`; an embedder points it at a host-owned element via `LuxarApp.init({ container })`. Page-level singleton (like `eventBus`/`ThemeManager`) — set at the start of `init()`, reset by the dispose pipeline — so the supported contract stays **one viewer per page**.

- `getViewerContainer()` — The current mount root (falls back to `document.body`).
- `setViewerContainer(el)` — Adopt `el`; a non-`body` element is promoted to a containing block (`contain: layout`, plus `position: relative` when statically positioned) so the viewer's `position: fixed`/`absolute` overlays scope to it. Saves the element's prior inline `position`/`contain`.
- `resetViewerContainer()` — Revert to `document.body` and restore exactly the inline styles `setViewerContainer` mutated.

### webgpu-availability.ts - Backend Availability Probe

Page-load-time graphics-API probe answering "what's available?" (vs `RendererCapabilities.apiSurface`, which answers "what did we pick?"). Used for diagnostics badges and feature flags. See `BROWSER_SUPPORT_POLICY.md` for the WebGPU/WebGL2 policy.

- `RendererAPI` = `'webgpu' | 'webgl2' | 'unsupported'`
- `getRendererAPI()` — Async, definitive. Requests a WebGPU adapter to confirm; result is cached after the first call.
- `getRendererAPISync()` — Synchronous fast path. Returns `'webgpu'` if `navigator.gpu` exists at all (no adapter probe); else falls back to `'webgl2'` / `'unsupported'`.
- `_resetCachedAPI()` — Test-only cache reset

## Console Interception

### Ring Buffer Implementation

```typescript
class ConsoleInterceptor {
  private messageBuffer: BufferedMessage[] = [];
  private bufferIndex = 0;
  private maxBufferSize = DEFAULT_MAX_BUFFER_SIZE; // 10000; reconfigurable via setMaxBufferSize()
  private hasWrapped = false;
}
```

**Key Features**:

- **Memory Efficient**: Bounded circular buffer (default 10,000, `setMaxBufferSize`-tunable) prevents memory leaks
- **Early Capture**: Starts before any other code executes
- **Original Preservation**: Maintains original console.\* functionality
- **Stack Traces**: Automatic stack trace extraction for errors

### Message Structure

```typescript
interface BufferedMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  stack?: string; // For errors
}
```

### Usage Patterns

```typescript
// Import at the very top of main.ts to ensure early capture
import { consoleInterceptor } from '../utils/console-interceptor';

// Access buffered messages
const messages = consoleInterceptor.getBufferedMessages();

// Add real-time listener
consoleInterceptor.addListener((message) => {
  // Handle new console output in real-time
});
```

## HDR Detection

### Capability Snapshot

```typescript
interface HDRCapabilities {
  p3Gamut: boolean; // Display P3 wide color gamut
  rec2020Gamut: boolean; // Rec2020 gamut support
  hdr: boolean; // High dynamic range
  deepColor: boolean; // 10-bit+ color depth
  floatTextures: boolean; // WebGL float texture support (filled by RendererCapabilities)
  colorDepth: {
    // Actual color buffer depth (filled by RendererCapabilities)
    red: number;
    green: number;
    blue: number;
  };
  recommendedColorSpace: 'srgb' | 'display-p3' | 'rec2020';
}
```

### Detection Methods

```typescript
// CSS media query detection (this module)
const hdr = window.matchMedia('(dynamic-range: high)').matches;
const p3Gamut = window.matchMedia('(color-gamut: p3)').matches;
const rec2020Gamut = window.matchMedia('(color-gamut: rec2020)').matches;
// Deep-color probe is belt-and-braces: per-component AND total bit-depth thresholds
const deepColor =
  window.matchMedia('(color: 48)').matches || window.matchMedia('(color: 30)').matches;
```

`floatTextures` and `colorDepth` are populated by `rendering/renderer-capabilities.ts` (the single seam where raw GL is allowed) — `detectDisplayCapabilities()` returns placeholder defaults for those two fields.

### HDR Capability Logging

```typescript
export function configureHDRRenderer(_renderer: unknown, capabilities: HDRCapabilities): void {
  // Log detected capabilities (informational only)
  // Does NOT configure renderer - PostProcessingManager handles that
}
```

**Note**: `configureHDRRenderer()` only logs HDR capabilities. It does NOT configure the renderer — actual HDR configuration (`outputColorSpace`, `toneMapping`) is owned by `PostProcessingManager` (the mega-shader pipeline applies tone mapping internally; the host pins `outputColorSpace = SRGB` / `toneMapping = NoToneMapping`).

## Cross-Layer Plumbing

Three utilities exist specifically to let lower layers reach the UI without violating layer order (see `CONVENTIONS.md` §10):

- **`notifier`** — Single backend, fixed method dictionary. The UI bootstrap calls `setNotifierBackend(...)` once with concrete implementations from `ui/` helper modules; lower layers call `notifier.toast(...)`, `notifier.error(...)`, etc. Pre-registration calls drop silently with a single warn.
- **`eventBus`** — Open subscriber sets typed against `LuxarEventMap`. Panels can subscribe late without bootstrap-order coupling. Events with no listener drop silently — that's the design.
- **`EventGroup`** — Per-component listener-collection so a panel's entire DOM-listener set tears down in one `dispose()` call.

The bus and notifier are deliberately separate: the notifier has one backend and is a method dictionary; the bus has many subscribers per event and is a typed catalog.

## Usage Examples

### Console Debugging Setup

```typescript
// At the very top of main.ts (or LuxarApp.init({ debug: true }))
import { consoleInterceptor } from '../utils/console-interceptor';
consoleInterceptor.patch(); // explicit opt-in; module load alone does not patch console

// Later, in the debug-console component
const messages = consoleInterceptor.getBufferedMessages();
consoleInterceptor.addListener((message) => displayMessage(message));
```

### HDR Display Detection

```typescript
import {
  detectDisplayCapabilities,
  configureHDRRenderer,
  logHDRCapabilities,
  isHDRDisplay,
} from '../utils/hdr/hdr-detection';

// Detect display-side capabilities (renderer probes fill float/depth later)
const hdrCapabilities = detectDisplayCapabilities();
logHDRCapabilities(hdrCapabilities);
configureHDRRenderer(renderer, hdrCapabilities); // logs only — does not configure renderer

if (isHDRDisplay(hdrCapabilities)) {
  // Enable advanced HDR features
}
```

### Cross-Layer Event

```typescript
import { eventBus } from '../utils/cross-layer/event-bus';

// Publisher (animation loop)
eventBus.emit('frame-start', {});

// Subscriber (performance panel) — late-binding-safe via replayLast
const off = eventBus.on('loading-progress', (p) => updateBar(p), { replayLast: true });
// ... later
off();
```

### Listener Group

```typescript
import { EventGroup } from '../utils/cross-layer/event-group';

class Panel {
  private events = new EventGroup();
  attach() {
    this.events.on(window, 'focus', this.handleFocus);
    this.events.on(document, 'visibilitychange', this.handleVisibility);
    this.events.add(() => this.observer.disconnect());
  }
  dispose() {
    this.events.dispose(); // removes everything in reverse order
  }
}
```

## Performance Considerations

### Console Buffer Management

- **Bounded Size**: Ring buffer (default 10,000 messages, `setMaxBufferSize`-tunable) prevents unbounded memory growth; shrinking trims oldest-first to preserve chronological order
- **Boundary-Safe Wrap**: When `length === maxBufferSize`, the next write goes to index 0 (not `maxBufferSize`, which would have grown the array and stranded the oldest entry)
- **Listener Set**: `Set<callback>` for O(1) add/remove and snapshot iteration on emit

### Event Bus

- **Listener Snapshot on Emit**: `emit` iterates `[...set]` so a callback that subscribes / unsubscribes mid-emit doesn't reorder the active loop.
- **Last-Payload Cache**: `lastPayload` is set on every emit so `on(..., { replayLast: true })` is O(1).

---

This package is the cross-cutting foundation layer. Anything that touches global state (console, localStorage, notifier backend) is opt-in via an explicit call; module import alone never patches the host page.
