/**
 * Per-frame `renderer.info` sampler for the debug/perf surface.
 *
 * Three resets `renderer.info.render` at the start of every `render()` call
 * (with the default `autoReset`), so by the time a probe reads it between
 * frames it holds only the LAST pass of the post-processing chain (a single
 * full-screen quad: `calls: 1, triangles: 1`). Reading it at `frame-end` —
 * emitted by the animation loop right after `postProcessing.render()` — sees
 * the whole frame's counters before the next reset. On the WebGPU backend
 * the counters may lag one frame; they are diagnostics, not assertions.
 *
 * The two renderers' `info` objects disagree on field meaning, and the
 * snapshot normalizes them:
 *
 * | snapshot   | `WebGLRenderer`             | `WebGPURenderer` (three `common/Info`)   |
 * | ---------- | --------------------------- | ---------------------------------------- |
 * | `calls`    | `render.calls` (draw calls) | `render.drawCalls` (`render.calls` there |
 * |            |                             | counts `render()` passes since startup)  |
 * | `frame`    | `render.frame`              | `render.calls`                           |
 * | `programs` | `programs.length`           | `memory.programs` (shader stages)        |
 *
 * Allocation-free on the hot path: one snapshot object is mutated in place
 * per frame and copied only when read.
 *
 * @module core/app/debug/renderer-info-sampler
 */

import { eventBus, type Unsubscribe } from '../../../utils/cross-layer/event-bus';

/** Counters copied from `renderer.info` at the last `frame-end`. */
export interface RendererInfoSnapshot {
  /** Three's render-pass counter (increments per `render()` call, not per frame). */
  frame: number;
  /** Draw calls in the sampled frame, on either backend. */
  calls: number;
  triangles: number;
  points: number;
  lines: number;
  geometries: number;
  textures: number;
  /**
   * Live shader programs, or null when the backend exposes none. NOT comparable
   * across backends: WebGL counts linked programs, WebGPU counts shader STAGES
   * (a render pipeline's vertex and fragment modules each count once).
   */
  programs: number | null;
  /** `performance.now()` of the sampled frame-end. */
  sampledAt: number;
  /** Number of frame-end samples taken since install. */
  samples: number;
}

/** Duck-typed subset of `THREE.WebGLRenderer` / `WebGPURenderer` we read. */
export interface RendererInfoSource {
  info?: {
    /** Three resets the counters at every `render()` when true (the default). */
    autoReset?: boolean;
    reset?: () => void;
    render?: {
      /** WebGL only: render-pass counter. */
      frame?: number;
      /** WebGL: draw calls this frame. WebGPU: `render()` passes since startup. */
      calls?: number;
      /** WebGPU only: draw calls this frame. */
      drawCalls?: number;
      /** WebGPU only: `render()` passes this frame. */
      frameCalls?: number;
      triangles?: number;
      points?: number;
      lines?: number;
    };
    /** `programs` is WebGPU only (a count). */
    memory?: { geometries?: number; textures?: number; programs?: number };
    /** WebGL only: the program cache. */
    programs?: unknown[] | null;
  };
}

const snapshot: RendererInfoSnapshot = {
  frame: 0,
  calls: 0,
  triangles: 0,
  points: 0,
  lines: 0,
  geometries: 0,
  textures: 0,
  programs: null,
  sampledAt: 0,
  samples: 0,
};
let unsubscribe: Unsubscribe | null = null;
/** The `renderer.info` whose `autoReset` this sampler switched off. */
let autoResetOwned: NonNullable<RendererInfoSource['info']> | null = null;

const EMPTY: Record<string, number | undefined> = {};

function num(value: unknown): number {
  return typeof value === 'number' ? value : 0;
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

type RendererInfo = NonNullable<RendererInfoSource['info']>;

/** WebGPU's `Info` is the one with a per-frame `drawCalls` counter. */
function isWebGPUInfo(render: RendererInfo['render'] | undefined): boolean {
  return typeof render?.drawCalls === 'number';
}

function programCount(info: RendererInfo): number | null {
  if (Array.isArray(info.programs)) return info.programs.length;
  const programs = info.memory?.programs;
  return typeof programs === 'number' ? programs : null;
}

function sample(renderer: RendererInfoSource | null | undefined): void {
  const info = renderer?.info;
  if (!info) return;
  const render = info.render ?? EMPTY;
  const memory = info.memory ?? EMPTY;
  const webgpu = isWebGPUInfo(info.render);
  snapshot.frame = num(webgpu ? render.calls : render.frame);
  snapshot.calls = num(webgpu ? render.drawCalls : render.calls);
  snapshot.triangles = num(render.triangles);
  snapshot.points = num(render.points);
  snapshot.lines = num(render.lines);
  snapshot.geometries = num(memory.geometries);
  snapshot.textures = num(memory.textures);
  snapshot.programs = programCount(info);
  snapshot.sampledAt = nowMs();
  snapshot.samples += 1;
}

/**
 * The per-frame counters `info.reset()` zeroes, saved at frame-start. WebGL
 * zeroes `calls`; WebGPU zeroes `drawCalls` / `frameCalls` and leaves its
 * cumulative `calls` alone (restoring that one is then a no-op).
 */
const RESET_COUNTERS = [
  'calls',
  'drawCalls',
  'frameCalls',
  'triangles',
  'points',
  'lines',
] as const;
type ResetCounter = (typeof RESET_COUNTERS)[number];
const savedRender: Record<ResetCounter, number | undefined> = {
  calls: undefined,
  drawCalls: undefined,
  frameCalls: undefined,
  triangles: undefined,
  points: undefined,
  lines: undefined,
};

type RenderCounters = NonNullable<RendererInfo['render']>;

function saveRenderCounters(render: RenderCounters | undefined): void {
  for (const key of RESET_COUNTERS) savedRender[key] = render?.[key];
}

/** Undo a frame-start reset for a tick that drew nothing. */
function restoreRenderCounters(render: RenderCounters | undefined): void {
  if (!render) return;
  for (const key of RESET_COUNTERS) {
    const saved = savedRender[key];
    // Never invent a field the backend's Info does not carry.
    if (typeof saved === 'number') render[key] = saved;
  }
}

/**
 * Start sampling `renderer.info` at every `frame-end`. Re-installing replaces
 * the previous subscription (an app re-init after dispose). Returns the
 * unsubscribe thunk; {@link uninstallRendererInfoSampler} does the same.
 */
export function installRendererInfoSampler(
  getRenderer: () => RendererInfoSource | null | undefined
): Unsubscribe {
  uninstallRendererInfoSampler();
  // A frame is several `render()` calls (scene → HDR target, bloom mips,
  // the mega-shader quad, FXAA). With Three's default `autoReset` each call
  // wipes the counters first, so frame-end would see only the last quad.
  // Own the reset instead: clear at frame-start, read at frame-end, and hand
  // `autoReset` back on uninstall.
  //
  // Render-on-change: a tick may render nothing, and it is only known at
  // frame-end. So frame-start also saves the counters it resets, and a
  // `rendered: false` frame-end writes them back — `renderer.info` then keeps
  // holding the last RENDERED frame's counters between frames, as it did when
  // every tick rendered, for the snapshot and for anyone reading it directly.
  const offStart = eventBus.on('frame-start', () => {
    const info = getRenderer()?.info;
    if (!info) return;
    if (info.autoReset !== false) {
      info.autoReset = false;
      autoResetOwned = info;
    }
    saveRenderCounters(info.render);
    info.reset?.();
  });
  const offEnd = eventBus.on('frame-end', (frame) => {
    if (frame?.rendered === false) {
      restoreRenderCounters(getRenderer()?.info?.render);
      return;
    }
    sample(getRenderer());
  });
  const off: Unsubscribe = () => {
    offStart();
    offEnd();
    if (autoResetOwned) {
      autoResetOwned.autoReset = true;
      autoResetOwned = null;
    }
  };
  unsubscribe = off;
  return () => {
    if (unsubscribe === off) unsubscribe = null;
    off();
  };
}

/** Stop sampling and reset the counters. Safe to call when not installed. */
export function uninstallRendererInfoSampler(): void {
  unsubscribe?.();
  unsubscribe = null;
  snapshot.frame = 0;
  snapshot.calls = 0;
  snapshot.triangles = 0;
  snapshot.points = 0;
  snapshot.lines = 0;
  snapshot.geometries = 0;
  snapshot.textures = 0;
  snapshot.programs = null;
  snapshot.sampledAt = 0;
  snapshot.samples = 0;
}

/** Copy of the last sampled counters, or `null` before the first frame-end. */
export function getRendererInfoSnapshot(): RendererInfoSnapshot | null {
  return snapshot.samples === 0 ? null : { ...snapshot };
}
