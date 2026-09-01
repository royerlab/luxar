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

import { computeWorkingSetBudgetBytes, readHeapLimitBytes } from '../cache/heap-budget';
import { log, Modules } from '../utils/log';

/** Fraction of system RAM to devote to GPU geometry. */
const DEVICE_MEMORY_FRACTION = 0.25;
/**
 * Budget used when NO memory signal is available at all — neither
 * ``deviceMemory`` nor an explicit cache-pool override.
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
 * Explicit memory signals the auto budget folds in alongside `navigator.deviceMemory`.
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
}

/**
 * Compute the auto budget from ``navigator.deviceMemory``, the measured JS
 * heap, and an explicit cache-pool override — whichever is scarcest.
 *
 * THE HEAP TERM IS THE ONE THAT MATTERS, and it was briefly dropped from this
 * path on the reasoning that a coarse heap tier is not a GPU-memory signal.
 * That is true of VRAM and wrong for the failure this budget exists to prevent.
 * The crash is `RangeError: Array buffer allocation failed` — a JS HEAP
 * exhaustion — and a stranded pooled pair holds a CPU-side ArrayBuffer as well
 * as a texture, so it is heap the pool must be bounded against.
 * ``deviceMemory`` is structurally blind to that: it reports system RAM, so a
 * 32 GB machine whose browser caps the tab heap near 4 GB reads as "roomy",
 * yields the 2 GB ceiling, never binds, and evicts nothing right up until the
 * tab dies. Measured: with the heap term the budget resolved to 537 MB and the
 * pool reclaimed during churn; without it, 2000 MB and zero evictions on the
 * same scene.
 *
 * It reuses ``computeWorkingSetBudgetBytes`` deliberately, rather than deriving
 * a second heap fraction of its own. That helper is the project's existing
 * answer to "how much CPU-side working set can this device spare", the
 * refinement residency cap already sizes from it, and a pooled buffer is
 * exactly such a working set. Two independent heap fractions would be two
 * authorities free to disagree — which is the arrangement this replaced.
 */
function computeAutoBudget(memory?: GpuBudgetMemorySignals): { bytes: number; sources: string[] } {
  const gb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const fromDeviceMemory =
    typeof gb === 'number' && gb > 0 ? gb * DEVICE_MEMORY_FRACTION * 1_000_000_000 : undefined;
  const cachePoolOverrideBytes = memory?.cachePoolOverrideBytes;
  const fromCachePool =
    typeof cachePoolOverrideBytes === 'number' &&
    Number.isFinite(cachePoolOverrideBytes) &&
    cachePoolOverrideBytes > 0
      ? computeWorkingSetBudgetBytes(undefined, cachePoolOverrideBytes)
      : undefined;
  // Only when the heap is actually MEASURABLE. `computeWorkingSetBudgetBytes`
  // answers with a fixed fallback when it has nothing to go on, and that value
  // is indistinguishable from a derived one — folding it in unguarded would cap
  // a 32 GB Firefox/Safari box at the fallback purely for exposing no
  // `performance.memory`. An absent measurement is not a small one.
  const fromHeap = readHeapLimitBytes() !== undefined ? computeWorkingSetBudgetBytes() : undefined;

  const candidates = [fromDeviceMemory, fromHeap, fromCachePool].filter(
    (value): value is number => value !== undefined
  );
  if (candidates.length === 0) {
    return {
      bytes: NO_SIGNAL_BUDGET_BYTES,
      sources: [`no memory signal -> ${mb(NO_SIGNAL_BUDGET_BYTES)} MB fallback`],
    };
  }

  // The MINIMUM of the signals, because a pooled allocation costs on BOTH
  // sides: VRAM for the element texture, and JS heap for the CPU-side image
  // that backs it. It has to fit in whichever is scarcer.
  //
  // Clamped ABOVE only. There is no floor: a floor is exactly what stopped this
  // budget from ever binding, and a budget that cannot bind cannot evict.
  const sources: string[] = [];
  if (fromDeviceMemory !== undefined) {
    sources.push(`deviceMemory=${gb} GB -> ${mb(Math.min(fromDeviceMemory, MAX_BUDGET_BYTES))} MB`);
  }
  if (fromHeap !== undefined) {
    sources.push(`heap -> ${mb(fromHeap)} MB`);
  }
  if (fromCachePool !== undefined) {
    sources.push(`cacheBudgetMB=${mb(cachePoolOverrideBytes!)} -> ${mb(fromCachePool)} MB`);
  }
  return { bytes: Math.min(Math.min(...candidates), MAX_BUDGET_BYTES), sources };
}

/**
 * Configure the budget once at startup from the resolved config value
 * (or the ``?gpuBudgetMB`` URL param, which the caller passes in):
 *
 * - ``null`` / ``undefined`` → **auto-size** from device memory and any
 *   explicit cache-pool override.
 * - ``0`` → disable byte-budget eviction (unbounded resident geometry).
 * - a positive number → pin the budget to exactly that many bytes.
 */
export function configureGpuByteBudget(
  overrideBytes?: number | null,
  memory?: GpuBudgetMemorySignals
): void {
  if (overrideBytes == null) {
    const auto = computeAutoBudget(memory);
    budgetBytes = auto.bytes;
    const resolution =
      auto.sources.length === 1 && auto.sources[0].startsWith('no memory signal')
        ? auto.sources[0]
        : `${auto.sources.join(', ')}; min=${mb(auto.bytes)} MB`;
    log.info(Modules.PERFORMANCE, `GPU byte budget: ${mb(budgetBytes)} MB (auto: ${resolution})`);
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
