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
 * (36 with scalars) in the interleaved era. `estimateGeometryBytes`
 * counts both texture and attribute, so eviction pressure reflects the
 * true footprint. The planned RGBA16F narrowing (spec §8) roughly
 * halves the texture share.
 *
 * @module rendering/gpu-byte-budget
 */

import { log, Modules } from '../utils/log';

/** Fraction of system RAM to devote to GPU geometry. */
const DEVICE_MEMORY_FRACTION = 0.25;
/** Lower clamp — also the fallback when ``deviceMemory`` is unavailable. */
const MIN_BUDGET_BYTES = 512_000_000; // 512 MB
/** Upper clamp — keeps discrete-GPU machines (system RAM ≫ VRAM) safe. */
const MAX_BUDGET_BYTES = 2_000_000_000; // 2 GB
/** Floor that context-loss backoff will not reduce below. */
const BACKOFF_FLOOR_BYTES = 256_000_000; // 256 MB

let budgetBytes = MIN_BUDGET_BYTES;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Compute the auto budget from ``navigator.deviceMemory``. Undefined
 * (Safari/Firefox) ⇒ the conservative floor.
 */
function computeAutoBudget(): number {
  const gb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (typeof gb === 'number' && gb > 0) {
    return clamp(gb * DEVICE_MEMORY_FRACTION * 1_000_000_000, MIN_BUDGET_BYTES, MAX_BUDGET_BYTES);
  }
  return MIN_BUDGET_BYTES;
}

/**
 * Configure the budget once at startup from the resolved config value
 * (or the ``?gpuBudgetMB`` URL param, which the caller passes in):
 *
 * - ``null`` / ``undefined`` → **auto-size** from device memory.
 * - ``0`` → disable byte-budget eviction (unbounded resident geometry).
 * - a positive number → pin the budget to exactly that many bytes.
 */
export function configureGpuByteBudget(overrideBytes?: number | null): void {
  if (overrideBytes == null) {
    budgetBytes = computeAutoBudget();
    const dm = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    log.info(
      Modules.PERFORMANCE,
      `GPU byte budget: ${mb(budgetBytes)} MB (auto from deviceMemory=${dm ?? 'n/a'} GB)`
    );
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
