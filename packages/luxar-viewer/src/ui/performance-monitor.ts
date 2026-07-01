// Performance monitoring for the Luxar scene player.
//
// A compact, square readout (rail-button sized, theme-matched) that shows one
// metric at a time and cycles FPS → frame time (ms) → a scrolling graph on
// click — the same information stats.js exposed, restyled to the Luxar tokens
// and driven by the animation loop's `frame-start` / `frame-end` events.
//
// Public API (toggle / show / hide / visible / cycleMode / dispose) and the
// `#luxar-stats` element id are preserved so the InputHandler (P key) and the
// control rail toggle it unchanged.

import { config } from '../config';
import { eventBus, type Unsubscribe } from '../utils/cross-layer/event-bus';

/** DOM refresh cadence (ms) — decoupled from frame rate. */
const RENDER_INTERVAL_MS = 200;
/** Rolling window for the FPS average (ms). */
const FPS_WINDOW_MS = 500;
/** Samples kept for the scrolling graph. */
const HISTORY = 48;

type PerfMode = 'fps' | 'ms' | 'graph';
const MODES: PerfMode[] = ['fps', 'ms', 'graph'];

export interface PerfKeepAlive {
  request: () => void;
  release: () => void;
}

export class PerformanceMonitor {
  private readonly el: HTMLDivElement;
  private readonly numEl: HTMLSpanElement;
  private readonly unitEl: HTMLSpanElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx2d: CanvasRenderingContext2D | null;

  private isVisible = false;
  private mode: PerfMode = 'fps';

  private frameStartUnsubscribe: Unsubscribe | null = null;
  private frameEndUnsubscribe: Unsubscribe | null = null;

  // Timing state.
  private frameT0 = 0;
  private msEma = 0;
  private frameCount = 0;
  private windowStart = 0;
  private fps = 0;
  private lastRender = 0;
  private readonly fpsHistory: number[] = [];

  constructor(private readonly keepAlive?: PerfKeepAlive) {
    const el = document.createElement('div');
    el.id = 'luxar-stats';
    el.className = 'luxar-perf';
    el.dataset.mode = this.mode;
    el.setAttribute('role', 'status');
    el.setAttribute('aria-label', 'Performance metrics — click to cycle FPS, frame time, graph');
    el.title = 'Click to cycle: FPS · ms · graph';
    el.style.zIndex = String(config.ui.zIndex.statsMonitor);
    el.classList.add('is-hidden');

    const value = document.createElement('div');
    value.className = 'luxar-perf__value';
    this.numEl = document.createElement('span');
    this.numEl.className = 'luxar-perf__num';
    this.unitEl = document.createElement('span');
    this.unitEl.className = 'luxar-perf__unit';
    this.numEl.textContent = '––';
    this.unitEl.textContent = 'fps';
    value.append(this.numEl, this.unitEl);

    this.canvas = document.createElement('canvas');
    this.canvas.className = 'luxar-perf__graph';
    this.ctx2d = this.canvas.getContext('2d');

    el.append(value, this.canvas);
    el.addEventListener('click', () => this.cycleMode());

    this.el = el;
  }

  /** The widget element — the caller mounts it (e.g. docked in the control rail). */
  get element(): HTMLElement {
    return this.el;
  }

  private readonly onFrameStart = (): void => {
    this.frameT0 = performance.now();
  };

  private readonly onFrameEnd = (): void => {
    const now = performance.now();
    const dt = now - this.frameT0;
    this.msEma = this.msEma ? this.msEma * 0.9 + dt * 0.1 : dt;

    if (!this.windowStart) this.windowStart = now;
    this.frameCount++;
    const elapsed = now - this.windowStart;
    if (elapsed >= FPS_WINDOW_MS) {
      this.fps = (this.frameCount * 1000) / elapsed;
      this.frameCount = 0;
      this.windowStart = now;
      this.fpsHistory.push(this.fps);
      if (this.fpsHistory.length > HISTORY) this.fpsHistory.shift();
    }

    if (now - this.lastRender >= RENDER_INTERVAL_MS) {
      this.render();
      this.lastRender = now;
    }
  };

  private render(): void {
    if (this.mode === 'graph') {
      this.drawGraph();
      return;
    }
    if (this.mode === 'fps') {
      const fps = Math.round(this.fps);
      this.numEl.textContent = `${fps}`;
      this.numEl.dataset.level = fps >= 50 ? 'good' : fps >= 30 ? 'ok' : 'bad';
      this.unitEl.textContent = 'fps';
    } else {
      this.numEl.textContent = this.msEma.toFixed(1);
      delete this.numEl.dataset.level;
      this.unitEl.textContent = 'ms';
    }
  }

  private drawGraph(): void {
    const ctx = this.ctx2d;
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth || 42;
    const h = this.canvas.clientHeight || 42;
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const hist = this.fpsHistory;
    if (hist.length === 0) return;
    const max = Math.max(60, ...hist);
    const style = getComputedStyle(this.el);
    const accent = style.getPropertyValue('--luxar-highlight').trim() || '#00a0ff';
    const barW = w / HISTORY;
    ctx.fillStyle = accent;
    for (let i = 0; i < hist.length; i++) {
      const v = hist[i] / max;
      const bh = Math.max(1, v * (h - 2));
      const x = (HISTORY - hist.length + i) * barW;
      ctx.fillRect(x, h - bh, Math.max(1, barW - 0.5), bh);
    }
  }

  /** Toggle visibility. Subscribes to frame timing only while visible. */
  toggle(): void {
    this.isVisible = !this.isVisible;
    this.el.classList.toggle('is-hidden', !this.isVisible);
    if (this.isVisible) {
      this.subscribeToFrameTiming();
      this.keepAlive?.request();
    } else {
      this.unsubscribeFromFrameTiming();
      this.keepAlive?.release();
    }
  }

  show(): void {
    if (!this.isVisible) this.toggle();
  }

  hide(): void {
    if (this.isVisible) this.toggle();
  }

  get visible(): boolean {
    return this.isVisible;
  }

  /** Cycle the displayed metric: FPS → ms → graph → FPS. */
  cycleMode(): void {
    const next = (MODES.indexOf(this.mode) + 1) % MODES.length;
    this.mode = MODES[next];
    this.el.dataset.mode = this.mode;
    this.render();
  }

  dispose(): void {
    this.unsubscribeFromFrameTiming();
    if (this.isVisible) this.keepAlive?.release();
    this.el.remove();
  }

  private subscribeToFrameTiming(): void {
    if (this.frameStartUnsubscribe) return;
    this.windowStart = 0;
    this.frameCount = 0;
    this.lastRender = 0;
    this.frameStartUnsubscribe = eventBus.on('frame-start', this.onFrameStart);
    this.frameEndUnsubscribe = eventBus.on('frame-end', this.onFrameEnd);
  }

  private unsubscribeFromFrameTiming(): void {
    this.frameStartUnsubscribe?.();
    this.frameEndUnsubscribe?.();
    this.frameStartUnsubscribe = null;
    this.frameEndUnsubscribe = null;
  }
}
