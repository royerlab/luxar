/**
 * Adaptive GPU-geometry byte budget — the single VRAM budget shared by
 * the GPU buffer pool (pooled-buffer eviction) and the LOD-group
 * registry (resident-level eviction). One authority, not two.
 *
 * Browsers deliberately do NOT expose total/available VRAM (it's a
 * fingerprinting vector), so we can't read an "absolute max" and set to
 * it. Auto-sizing takes the lower of ``navigator.deviceMemory`` and the shared
 * non-cache remainder. An explicit total cache-pool override replaces the
 * heap-derived remainder in either direction; a mobile device-class fallback
 * supplies it when the heap is unavailable, and an independent 128 MiB mobile
 * cap remains a peer minimum. The result has a 2 GB ceiling but no lower clamp,
 * with 512 MB used only when no signal exists. As a safety net, the budget is
 * halved on a WebGL context-loss event (a strong OOM signal) so an over-estimate
 * self-corrects instead of repeatedly crashing the context.
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

import {
  computeNonCacheRemainderBytes,
  DEVICE_CLASS_POOL_BYTES,
  readHeapLimitBytes,
} from '../cache/heap-budget';
import { getInputProfile } from '../utils/input-capabilities';
import { log, Modules } from '../utils/log';

/** Fraction of system RAM to devote to GPU geometry. */
const DEVICE_MEMORY_FRACTION = 0.25;
const MIB = 1024 * 1024;
/** Mobile GPU residency cap established by the device-runtime budget pass (#2588). */
const MOBILE_BUDGET_BYTES = 128 * MIB;
/**
 * Budget used when NO memory signal is available at all — no device memory,
 * measured heap, explicit cache-pool override or mobile device class.
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
 * Compute the auto budget from ``navigator.deviceMemory`` and the shared
 * non-cache remainder. An explicit cache-pool override deliberately replaces
 * the measured-heap remainder in either direction, while remaining a peer
 * minimum against device memory and the independent mobile cap.
 *
 * THE HEAP TERM IS THE ONE THAT MATTERS, and it has now been removed twice on
 * the reasoning that a coarse heap tier is not a GPU-memory signal. That is
 * true of VRAM and wrong for the failure this budget exists to prevent. The
 * crash is `RangeError: Array buffer allocation failed` — JS heap exhaustion —
 * and a stranded pooled pair holds a CPU-side ArrayBuffer as well as a
 * texture, so heap is what must bound the pool. ``deviceMemory`` is
 * structurally blind to it: it reports SYSTEM RAM, so a 32 GB machine whose
 * tab heap caps near 4 GB reads as roomy, takes the 2 GB ceiling, never binds,
 * and evicts nothing right up until the tab dies.
 *
 * MEASURED, not argued (host-demos, `jsHeapSizeLimit` spoofed in an init
 * script with real used/total passing through): without the heap term the auto
 * budget stayed at 2000 MB at a spoofed 256 MB heap and reclaimed nothing —
 * not one MB of movement. The pre-#2439 half-share tracked the spoof at 43 / 86
 * / 172 MB for 256 / 512 / 1024 MiB and closed the auto-binding regression
 * (multi6 real-heap evict=5); the full-remainder derivation now resolves those
 * same heap limits to 86 / 172 / 344 MB.
 *
 * Heap, explicit-pool and device-class inputs all go through
 * ``computeNonCacheRemainderBytes``: the project's single answer to "how much
 * can this device spare outside the cache pool". GPU geometry may use that
 * remainder up to this module's own 2 GB ceiling. The eager loader and
 * refinement residency gate deliberately keep their separate capped half-share;
 * mobile devices also retain their independent 128 MiB safety cap. The
 * populations overlap, so these independent ceilings are not additive
 * reservations.
 */
function computeAutoBudget(memory?: GpuBudgetMemorySignals): { bytes: number; sources: string[] } {
  const gb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const fromDeviceMemory =
    typeof gb === 'number' && gb > 0 ? gb * DEVICE_MEMORY_FRACTION * 1_000_000_000 : undefined;
  const cachePoolOverrideBytes = memory?.cachePoolOverrideBytes;
  const hasCachePoolOverride =
    typeof cachePoolOverrideBytes === 'number' &&
    Number.isFinite(cachePoolOverrideBytes) &&
    cachePoolOverrideBytes > 0;
  const heapLimitBytes = readHeapLimitBytes();
  const isMobile = getInputProfile().deviceClass === 'mobile';
  // A phone or tablet is a memory signal in itself. WebKit exposes neither
  // `deviceMemory` nor `performance.memory`, so without this an iPhone took the
  // 512 MB no-signal fallback — ≈7 M resident splats, each also holding a
  // CPU-side ArrayBuffer — which is the `RangeError: Array buffer allocation
  // failed` this module exists to prevent. The mobile cache pool the cache
  // tiers already size from (`DEVICE_CLASS_POOL_BYTES.mobile`) implies the same
  // non-cache remainder as an explicit `?cacheBudgetMB=` would. Laptop and
  // desktop contribute nothing here, so their resolution is unchanged.
  const fallbackPoolBytes = isMobile ? DEVICE_CLASS_POOL_BYTES.mobile : undefined;
  const nonCacheRemainder = computeNonCacheRemainderBytes(
    heapLimitBytes ?? 0,
    cachePoolOverrideBytes,
    fallbackPoolBytes
  );
  const fromWorkingSet =
    nonCacheRemainder === undefined
      ? undefined
      : Math.floor(Math.min(nonCacheRemainder, MAX_BUDGET_BYTES));
  // The mobile pool implies a 256 MiB remainder, so the independently measured
  // 128 MiB safety cap dominates that candidate by construction.
  const fromMobileCap = isMobile ? MOBILE_BUDGET_BYTES : undefined;

  const candidates = [fromDeviceMemory, fromWorkingSet, fromMobileCap].filter(
    (value): value is number => value !== undefined
  );
  if (candidates.length === 0) {
    return {
      bytes: NO_SIGNAL_BUDGET_BYTES,
      sources: [`no memory signal -> ${mb(NO_SIGNAL_BUDGET_BYTES)} MB fallback`],
    };
  }

  // The MINIMUM of the available signals. The shared non-cache remainder keeps
  // the heap signal live on roomy Chromium tiers instead of inheriting the eager
  // loader's unrelated 512 MiB burst cap.
  //
  // Clamped ABOVE only. There is no floor: a floor is exactly what stopped this
  // budget from ever binding, and a budget that cannot bind cannot evict.
  const sources: string[] = [];
  if (fromDeviceMemory !== undefined) {
    sources.push(`deviceMemory=${gb} GB -> ${mb(Math.min(fromDeviceMemory, MAX_BUDGET_BYTES))} MB`);
  }
  if (fromWorkingSet !== undefined && hasCachePoolOverride) {
    sources.push(
      `cacheBudgetMB=${Math.round(cachePoolOverrideBytes / MIB)} -> ${mb(fromWorkingSet)} MB`
    );
  } else if (fromWorkingSet !== undefined && heapLimitBytes !== undefined) {
    sources.push(`heap -> ${mb(fromWorkingSet)} MB`);
  } else if (fromWorkingSet !== undefined && fallbackPoolBytes !== undefined) {
    sources.push(`mobile device class remainder -> ${mb(fromWorkingSet)} MB`);
  }
  if (fromMobileCap !== undefined) {
    sources.push(`mobile safety cap -> ${mb(fromMobileCap)} MB`);
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
