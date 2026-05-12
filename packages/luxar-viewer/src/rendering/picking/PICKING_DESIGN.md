# Picking strategy — design decision for the WebGPU port

## Context

The WebGPU port (planned for a future release) changes the readback
contract: `gl.readPixels` is synchronous; the WebGPU equivalent is
`buffer.mapAsync` + `getMappedRange`, which costs at minimum one frame
of latency on Apple Silicon and Linux/WebGPU drivers (~16 ms at 60 Hz,
~33 ms at 30 Hz).

The current picking system in `picking-system.ts` reads a 5×5 RGBA32F
window from a cached half-res pick buffer on every `mousemove` (after
a 10 ms debounce and a 60 Hz throttle). The readback itself is sync
(`renderer.readRenderTargetPixels`) and the result drives:

- hover-tooltip text (showing node/element id)
- cursor style changes
- click-time interaction (if mouse-down lands on a picked element)

A pre-existing CPU pre-cull narrows the work to nodes whose bounding
box intersects the cursor ray, but the **per-pixel decision is still
GPU-side** — necessary for translucent splats and overlapping line
segments where rasterised brightness picks the winner.

This document records the architecture decision for what picking
becomes after the renderer port. The decision needs to land before
the port begins so we don't stall on a product question mid-migration.

## Options evaluated

### Option A — Accept 1-frame latency

Make the readback async. The pick result for cursor position `(x, y)`
arrives one frame after the mouse moves there.

**Pros**:
- Minimal code change. `performPick` becomes async; `readbackAndVote`
  becomes async; the orchestrating callback at `app.ts:782` is
  already `async`.
- No new data structures to keep in sync.
- The `_dirty` flag still prevents re-render during pan/orbit;
  this option preserves that.

**Cons**:
- Tooltip text lags by one frame. On a 60 Hz device, that's 16 ms —
  imperceptible for most users.
- Cursor-style changes (e.g. "pointer over splat → crosshair") lag
  by one frame too. Marginal on 60 Hz, noticeable on 30 Hz.
- A fast mouse motion that crosses a small element entirely within
  one frame may produce a stale pick.
- *Important*: the existing 16 ms throttle on the hover-fast-path
  (`picking-system.ts:290`) does **not** absorb async-readback
  latency. The throttle and the per-frame readback both run at
  the display refresh rate — they're parallel, not serial. The
  cursor still updates at refresh rate; the *pick result* is
  what's delayed by one frame. Mitigation lives in the
  stale-tooltip suppression below, not in any existing buffering.

**Implementation cost**: ~20-line change in `picking-system.ts`
(see implementation outline section).

### Option B — CPU spatial index

Maintain an RBush (2D) or k-d-tree of geometry bounding boxes in
**screen space**, refreshed on every camera change. Pick by
intersecting the cursor with the index; no GPU readback at all on
hover. Reserve GPU readback for click events only (where one frame
of latency is invisible).

**Pros**:
- Zero hover latency, even on WebGPU.
- GPU readback frequency drops dramatically — only on `click`,
  not on `mousemove`.
- Battery / power benefit on laptops (less GPU work on hover).

**Cons**:
- Must keep the index in sync with every camera change (mat4
  update, resize, ortho/perspective switch, FOV change). Stale
  index → wrong pick.
- For GSplats specifically, screen-space bounds depend on per-splat
  Cholesky factors projected through the view matrix. Computing the
  2D bbox for a splat is the *expensive half* of the visual shader —
  doing it on the CPU for millions of splats is a non-starter.
- Brightness-weighted voting (the whole point of the current
  approach, for translucent splats and overlapping line segments)
  cannot be done CPU-side without effectively re-rasterising on
  the CPU.
- Maintenance cost is high: any change to vertex-shader projection
  logic (lens distortion, ortho near-cull) must be mirrored in the
  CPU index.

**Implementation cost**: 2–3 weeks of standalone work. Not gated
on WebGPU — it's a different architecture that could land
independently.

### Option C — GPU compute shader

Replace the rasterised pick pass with a compute shader that writes
a single (nodeId, elementId, brightness) result for the cursor
neighbourhood.

**Pros**:
- Scales better with element count.

**Cons**:
- Full rewrite of the picking shaders + readback path.
- WebGL2 has no compute shaders, so this option **requires** WebGPU
  to land first — reversed dependency, makes prep work harder.
- Doesn't actually solve the async-readback problem; it just
  reorganises the GPU work.

**Verdict**: eliminated. Not a prep candidate.

## Decision

**Adopt Option A — async readback with 1-frame latency.**

Rationale:

1. **The latency is below human perception thresholds** at 60 Hz
   (16 ms). Below 60 Hz, the user has bigger problems than picking
   lag, so optimising it doesn't move the needle.
2. **The *interactive* path is unchanged.** During pan/orbit the
   `_dirty` flag suppresses readback entirely (the debounce at
   `picking-system.ts:294-300` only kicks in on the dirty branch,
   coalescing the eventual single readback to when the camera
   settles). Async readback adds 1 frame on top of that
   already-debounced fire, which is invisible against the much
   larger camera-motion debounce window. The *hover-on-static-scene*
   path is where the 1-frame latency lands — there the only existing
   throttle is the 16 ms refresh-rate gate, which runs in parallel
   with each readback rather than buffering against it. So the
   1 frame is real, but it is bounded.
3. **Option B's complexity is not justified** until we have user
   reports of perceived hover lag. The maintenance cost of keeping
   screen-space bounds in sync with vertex-shader projection logic
   is real (every change to lens distortion / ortho semantics
   would need a parallel CPU update).
4. **Click-time picking remains responsive** because click events
   are not latency-sensitive in the way hover is — a click is a
   discrete event the user expects to take "a moment".

## Mitigations for the perceived-lag edge cases

- **Stale-tooltip suppression**: while a readback is in flight, the
  tooltip stays on the *previous* result rather than blanking. The
  result drifts at most one frame behind cursor position — fast
  motion across a small element shows the wrong tooltip for one
  frame, then the right one. We're betting this is invisible.
- **Cursor-style debounce**: keep the cursor on the last-known shape
  until the new readback settles. Avoids cursor flicker when the
  user scrubs across element boundaries.
- **Click pre-empt**: if a click arrives while a readback is in
  flight, *await* that readback before dispatching the click. This
  guarantees the click sees the pick result that matches the cursor
  position at click time, not one frame earlier.

## Implementation outline (for the port PR, not this design doc)

The real `performPick` (around `picking-system.ts:338`) does the
work of setting `this._lastReadX`/`_lastReadY` from the cursor
coordinates, then calls `readbackAndVote()` with no arguments —
`readbackAndVote` reads its window position from those fields
(see `picking-system.ts:519-527`). So the async edit is two
methods, no signature changes on the caller-facing API:

```ts
// picking-system.ts — diff sketch, NOT to be implemented yet.
// Signatures match the existing methods; only the keyword `async`
// and one `await` are added.

private async performPick(screenX: number, screenY: number): Promise<void> {
  // … existing cursor → _lastReadX/_lastReadY math …
  // … existing _dirty re-render branch …
  // … existing ray-bbox early-out …

  // The line that changes:
  const result = await this.readbackAndVote();   // was: const result = this.readbackAndVote();
  this.onPickResult(result);
}

private async readbackAndVote(): Promise<PickResult | null> {
  // WebGL2: renderer.readRenderTargetPixelsAsync exists in r184+;
  // it returns Promise<void> after the GPU finishes the readback.
  // Under WebGPU the same call dispatches buffer.mapAsync.
  await this.renderer.readRenderTargetPixelsAsync(
    this.pickTarget,
    this._lastReadX, this._lastReadY,
    PICK_SIZE, PICK_SIZE,
    this.readBuffer
  );
  // … existing vote logic over this.readBuffer …
}
```

The orchestrating callback at `app.ts:782` is **already** declared
`async (result: PickResult | null) => { … }` (verified during the
audit-fix planning). No call-site change needed there.

Mouse-move dispatch in `onMouseMove` (around `picking-system.ts:270`)
needs no change either: `performPick` is fired in a fire-and-forget
shape from both the clean-buffer branch (line 292) and the
debounce branch (line 295-300); both already discard the return
value, so awaiting nothing changes.

## Future revision criteria

Open a follow-up if any of the following are observed:

- User reports of perceived hover lag on **60 Hz hardware**
  (anything below 60 Hz is the rendering bottleneck, not picking).
- Profiling shows >5% GPU time spent re-rendering the pick buffer
  on hover (would suggest the dirty-flag isn't doing its job).
- A frequent user workflow involves rapid hover across many small
  elements (e.g. dense annotation work), and Option A's stale-tooltip
  edge case becomes a real complaint.

If any of these fire, revisit Option B with a narrower scope: keep
the GPU-rasterised pick for splats / overlapping lines, but use a
CPU index for point selection where bounding boxes are trivially
computable (just per-point spheres in world space).

## Out of scope for this decision

- The pick buffer encoding (`vec4(nodeId, elementId, brightness, 1)`)
  stays. The decision is about *async readback*, not encoding.
- Visual parity of pick / render shaders stays. The current parity
  rules (50 % truncation for points, 1.5σ for splats, full width for
  lines, brightness-as-depth) are preserved.
- The 5×5 sample window and majority-vote logic stay. Voting works
  identically whether the readback is sync or async.

## References

- `picking-system.ts` — current sync implementation
- `app.ts:782` — sole orchestrator (callback already async)
- WebGPU spec: `GPUBuffer.mapAsync()` — async readback contract
- Plan file: `~/.claude/plans/fixed-check-again-iridescent-pearl.md`
  (Item 4 of the WebGPU prep plan)
