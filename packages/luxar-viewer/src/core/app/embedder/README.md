# core/app/embedder — Programmatic embedder API

**Public Surface**: Events, value types, and headless screenshot utilities for embedding Luxar programmatically in other applications.

Unlike the rest of `app/` (private orchestrator support), this subpackage is re-exported from the package barrel (`src/index.ts`) — embedders import `LuxarEmbedderEventMap`, `EmbedderDimensions`, `ScreenshotOptions`, etc. from `'@royerlab/luxar-viewer'` directly.

## Modules

| File            | Status       | Description                                                                                                                                                                          |
| --------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `events.ts`     | **Public**   | Value and event type catalog: `LuxarEmbedderEventMap`, `EmbedderDimensions`, `ScreenshotOptions`, `SelectionPayload`. Re-exported from `src/index.ts`.                               |
| `screenshot.ts` | **Internal** | `captureScreenshot(sceneManager, overlayManager, opts)` — headless frame → Blob encoding. Composes the Recording panel's pure screenshot helpers without state-machine dependencies. |

## Events

`LuxarApp.on(event, listener)` subscribes to these app-scoped events (emitted on a per-app `EventEmitter`, not the cross-layer singleton):

| Event                  | Payload                         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `'dataset-loaded'`     | `{ src: string }`               | Fires when a dataset finishes loading (initial `init()`, browser selection, or `switchDataset()`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `'dataset-error'`      | `{ src: string; error: Error }` | Fires when a dataset fails to load.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `'dataset-fault'`      | `DatasetFaultPayload`           | Fires when the loaded dataset becomes unable to update, while preserving its last complete frame. An explicit retry clears the current fault; if the archive fails again, the event fires again. Subscribe before `init()` / `switchDataset()`, or call `getDatasetFault()` to inspect a fault that occurred before subscription.                                                                                                                                                                                                                                                                                                                                                                                         |
| `'dimensions-changed'` | `EmbedderDimensions`            | Fires on any slice-position change (slider, keyboard, or `setDimensionValue()`). Includes current step, displayed dims, metadata, and navigable ranges (all **cloned** copies).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `'selection'`          | `SelectionPayload \| null`      | Fires when the hover-pick changes (`nodeName` + `elementIndex` + `hitNodeName` under the cursor, or `null` when cleared). Subscribe **before** `init()` to provision picking for all datasets. What `elementIndex` counts is per-node — see `SelectionPayload`: a node declaring `has_labels` / `has_image_labels` / `has_keys` reports an on-disk index — flat, or, for **points**, an additive-LOD ladder, whose per-level maps are composed into the parent's union CSR space (#1439); such a **lines** node reports the picked segment's start-vertex row. A node declaring none of those channels — or an additive-LOD **lines** ladder, which publishes no map across its levels — reports the visible-buffer slot. |

## Usage

```typescript
import { LuxarApp } from '@royerlab/luxar-viewer';
import type { EmbedderDimensions } from '@royerlab/luxar-viewer';

const app = new LuxarApp();
await app.init({ canvas, container });

// Subscribe to events (returns an unsubscribe callback)
const unsubscribe = app.on('dimensions-changed', (dims: EmbedderDimensions) => {
  console.log('Slice position:', dims.currentStep);
  console.log('Displayed dims:', dims.displayed);
  console.log('Navigable ranges:', dims.ranges);
});

// Get current dimension state (all arrays are cloned — safe to read)
const dims = app.getDimensions();

// Set a slice position programmatically
app.setDimensionValue(3, 10); // Set dimension 3 (e.g., time) to step 10
await app.awaitDimensionUpdate(); // Wait for the requested slice to load

// Capture a screenshot
const blob = await app.screenshot({ format: 'webp', quality: 0.92 });
const url = URL.createObjectURL(blob);
const link = document.createElement('a');
link.href = url;
link.download = 'screenshot.webp';
link.click();

// Clean up
unsubscribe();
```

## Lifecycle Invariants

- Events are emitted **synchronously** when the state change completes (not deferred).
- All dimension arrays (`currentStep`, `displayed`, `ranges`, `metadata`) are **cloned** before the event fires — an embedder may read them freely without mutating viewer internals.
- The `selection` event fires on **hover** changes (the element under the cursor), not click-to-select.
- Picking is provisioned when a `selection` listener exists at dataset load time — subscribe **before** `init()` / `switchDataset()` to enable picking on all datasets.
- `SelectionPayload.nodeName` is the user-facing layer (the outermost `kind=partition` wrapper when the hit sits under one), so it is **not** the node `elementIndex` is local to. Resolve the element against `hitNodeName`, which is the leaf that was hit and equals `nodeName` when the node is not partitioned.

## See Also

- [`../README.md`](../README.md) — `app/` package overview: orchestrator support tree, lifecycle, and the `LuxarApp` public API.
- [`../../app.ts`](../../app.ts) — The `LuxarApp` class that calls these helpers.
- [`../snapshot/`](../snapshot/) — Separate JSON capture/restore of camera + slice position (for tests, share-view links, regression harnesses).
- [`../../../index.ts`](../../../index.ts) — Package barrel re-exporting these public types.
