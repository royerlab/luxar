/**
 * Adaptive GPU-geometry byte budget — the single VRAM budget shared by
 * the GPU buffer pool (pooled-buffer eviction) and the LOD-group
 * registry (resident-level eviction). One authority, not two.
 *
 * Browsers deliberately do NOT expose total/available VRAM (it's a
 * fingerprinting vector), so we can't read an "absolute max" and set to
 * it. The only portable memory signal is ``navigator.deviceMemory``
 * (system RAM in GB, privacy-rounded and capped at 8 in most browsers).
 * We derive a conservative budget from it, clamped to a safe range, with
 * an explicit override for power users. As a safety net, the budget is
 * halved on a WebGL context-loss event (a strong OOM signal) so an
 * over-estimate self-corrects instead of repeatedly crashing the context.
 *
 * **Single-viewer-per-page assumption.** The budget lives in module-global
 * state (``budgetBytes`` below), so it is shared by every pool and registry
 * in the page — including the context-loss backoff, which would halve the
 * one shared budget for all of them. This is correct for the supported mode
 * (one viewer per page). Embedding two independent viewers in the same tab
 * would make them fight over (and back off) a single budget; that would
 * require promoting this state onto a per-viewer instance (e.g. owned by
 * ``SceneLoader`` / ``SceneManager``) and threading it through the budget
 * getters wired in ``init/pipeline.ts`` and ``scene-loader.ts``.
 *
 * **Per-element cost note (texture storage).** A pooled gsplat
 * allocation costs ≈68 B/splat: 64 B in the RGBA32F splat texture
 * (4 texels × 16 B) + 4 B for the `aSortedIndex` ordering attribute —
 * vs 52 B/splat in the pre-texture interleaved era (≈ +31% VRAM).
 * A pooled points allocation costs ≈52 B/point: 48 B in the RGBA32F
 * point texture (3 texels × 16 B) + 4 B `aSortedIndex` — vs 32 B/point
 * (36 with scalars) in the interleaved era. EVERY node adds a further
 * 4 B/element for the second ordering buffer the atomic swap needs —
 * ≈72 B/splat, ≈56 B/point — charged from attach whether the node ever
 * depth-sorts or not: materialising that buffer lazily would change the
 * attribute set behind a cached native-WebGPU vertex layout and render
 * the scene black (`element-storage.ts`).
 * `estimateGeometryBytes` counts texture and attributes, deduplicating
 * by attribute identity, so eviction pressure reflects the true
 * footprint. The planned RGBA16F
 * narrowing (spec §8) roughly halves the texture share.
 *
 * @module rendering/gpu-byte-budget
 */

import { computeWorkingSetBudgetBytes } from '../cache/heap-budget';
import { log, Modules } from '../utils/log';

/** Fraction of system RAM to devote to GPU geometry. */
const DEVICE_MEMORY_FRACTION = 0.25;
/**
 * Budget used when NO memory signal is available at all — neither
 * ``deviceMemory`` nor a measurable heap nor an explicit cache-pool override.
 *
 * NOT a floor under a budget that WAS derived from a signal. It used to be one,
 * and that was the bug: a tab told it had a small pool still got 512 MB, which
 * is far above the stranded-pair total the pool needs to evict, so the LRU
 * never fired and nothing was ever reclaimed. A budget that cannot bind is not
 * a budget. See {@link computeAutoBudget}.
 */
const NO_SIGNAL_BUDGET_BYTES = 512_000_000; // 512 MB
/** Upper clamp — keeps discrete-GPU machines (system RAM ≫ VRAM) safe. */
const MAX_BUDGET_BYTES = 2_000_000_000; // 2 GB
/** Floor that context-loss backoff will not reduce below. */
const BACKOFF_FLOOR_BYTES = 256_000_000; // 256 MB

let budgetBytes = NO_SIGNAL_BUDGET_BYTES;

/**
 * Memory signals the auto budget folds in alongside `navigator.deviceMemory`.
 *
 * `cachePoolOverrideBytes` is the `?cacheBudgetMB=` value (and the native
 * launcher's equivalent). It MUST reach this budget, not just the caches:
 * `deviceMemory` is Chromium-only and spec-capped at 8 GB, so on a large
 * machine it pins the budget at the 2 GB ceiling and the pool can never be
 * exercised under pressure at all. The override is the only way to reproduce
 * constrained-device behaviour on a roomy box.
 */
export interface GpuBudgetMemorySignals {
  /** `?cacheBudgetMB=` in bytes, if set. */
  cachePoolOverrideBytes?: number;
  /** Device-class pool size, used when the heap is unmeasurable (WebKit). */
  fallbackPoolBytes?: number;
}

/**
 * Compute the auto budget from ``navigator.deviceMemory``. Undefined
 * (Safari/Firefox) ⇒ the conservative floor.
 */
function computeAutoBudget(memory?: GpuBudgetMemorySignals): number {
  const gb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const fromDeviceMemory =
    typeof gb === 'number' && gb > 0 ? gb * DEVICE_MEMORY_FRACTION * 1_000_000_000 : undefined;

  // The heap-derived working set, from the SAME helper the refinement residency
  // cap uses — so the two agree on what the device can hold instead of the pool
  // carrying an independent constant that never binds.
  //
  // ONLY when there is a REAL heap signal. `computeWorkingSetBudgetBytes`
  // returns a fixed 256 MB fallback when it has nothing to go on, and that
  // value is indistinguishable from a derived one — folding it in would cap a
  // 32 GB Firefox/Safari machine at 256 MB purely because it exposes no
  // `performance.memory`, making those browsers dramatically worse than Chrome
  // at scenes they hold comfortably. An absent measurement must never read as
  // a small one. (The two tests that caught this are the ones asserting the
  // deviceMemory path and the no-signal path.)
  const heapMeasurable =
    typeof performance !== 'undefined' &&
    typeof (performance as Performance & { memory?: { jsHeapSizeLimit?: number } }).memory
      ?.jsHeapSizeLimit === 'number';
  const fromHeap =
    memory?.cachePoolOverrideBytes != null || heapMeasurable
      ? computeWorkingSetBudgetBytes(
          undefined,
          memory?.cachePoolOverrideBytes,
          memory?.fallbackPoolBytes
        )
      : undefined;

  const candidates = [fromDeviceMemory, fromHeap && fromHeap > 0 ? fromHeap : undefined].filter(
    (v): v is number => v !== undefined
  );
  if (candidates.length === 0) return NO_SIGNAL_BUDGET_BYTES;

  // The MINIMUM of the signals, because a pooled allocation costs on BOTH
  // sides: VRAM for the element texture, and JS heap for the CPU-side image
  // that backs it. It has to fit in whichever is scarcer. (Mixing a VRAM signal
  // and a heap signal in one number IS a category blend; it is deliberate. The
  // failure we are fixing is a heap OOM caused by GPU-pool retention, so a
  // budget that only tracked VRAM could not see it coming.)
  //
  // Clamped ABOVE only. There is no floor: a floor is exactly what stopped this
  // budget from ever binding, and a budget that cannot bind cannot evict.
  return Math.min(Math.min(...candidates), MAX_BUDGET_BYTES);
}

/**
 * Configure the budget once at startup from the resolved config value
 * (or the ``?gpuBudgetMB`` URL param, which the caller passes in):
 *
 * - ``null`` / ``undefined`` → **auto-size** from device memory.
 * - ``0`` → disable byte-budget eviction (unbounded resident geometry).
 * - a positive number → pin the budget to exactly that many bytes.
 */
export function configureGpuByteBudget(
  overrideBytes?: number | null,
  memory?: GpuBudgetMemorySignals
): void {
  if (overrideBytes == null) {
    budgetBytes = computeAutoBudget(memory);
    const dm = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    const via =
      memory?.cachePoolOverrideBytes != null
        ? `cacheBudgetMB override + deviceMemory=${dm ?? 'n/a'} GB`
        : `deviceMemory=${dm ?? 'n/a'} GB + heap`;
    log.info(Modules.PERFORMANCE, `GPU byte budget: ${mb(budgetBytes)} MB (auto from ${via})`);
    return;
  }
  budgetBytes = Math.max(0, overrideBytes);
  log.info(
    Modules.PERFORMANCE,
    budgetBytes === 0
      ? 'GPU byte budget: disabled (0 — unbounded resident geometry)'
      : `GPU byte budget: ${mb(budgetBytes)} MB (explicit)`
  );
}

/** Current GPU-geometry byte budget. Read dynamically so backoff applies live. */
export function getGpuByteBudget(): number {
  return budgetBytes;
}

/**
 * Halve the budget (down to a floor) in response to a WebGL context-loss
 * event — a strong sign the previous budget over-committed VRAM. Returns
 * the new budget. The LOD registry reads ``getGpuByteBudget()`` each
 * frame, so the tighter budget takes effect immediately (it will evict
 * resident levels back under the new ceiling).
 */
export function reduceGpuByteBudgetForContextLoss(): number {
  const next = Math.max(BACKOFF_FLOOR_BYTES, Math.floor(budgetBytes * 0.5));
  if (next < budgetBytes) {
    log.warning(
      Modules.PERFORMANCE,
      `WebGL context lost — halving GPU byte budget ${mb(budgetBytes)} → ${mb(next)} MB`
    );
    budgetBytes = next;
  }
  return budgetBytes;
}

function mb(bytes: number): number {
  return Math.round(bytes / 1_000_000);
}
