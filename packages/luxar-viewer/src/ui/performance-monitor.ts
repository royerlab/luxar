// Performance monitoring for the Luxar scene player.
//
// A compact, theme-matched readout of FPS + frame time (and optionally JS heap
// memory), driven by the animation loop's `frame-start` / `frame-end` events.
// It replaces the previous stats.js canvas widget so the readout matches the
// Luxar design tokens and can dock cleanly next to the control rail.
//
// Public API (toggle / show / hide / visible / cyclePanels / dispose) and the
// `#luxar-stats` element id are preserved so the InputHandler (P key) and the
// control rail toggle it unchanged.

import { config } from '../config';
import { eventBus, type Unsubscribe } from '../utils/cross-layer/event-bus';
import { getViewerContainer } from '../utils/viewer-container';

/** How often the DOM text is refreshed (ms) — decoupled from frame rate. */
const RENDER_INTERVAL_MS = 250;
/** Rolling window for the FPS average (ms). */
const FPS_WINDOW_MS = 500;

interface PerfMemory {
  usedJSHeapSize: number;
}

/**
 * Optional hook to keep the render loop ticking while the readout is open, so
 * FPS stays live even when the scene would otherwise idle. `request` is called
 * on show, `release` on hide.
 */
export interface PerfKeepAlive {
  request: () => void;
  release: () => void;
}

export class PerformanceMonitor {
  private readonly el: HTMLDivElement;
  private readonly fpsEl: HTMLSpanElement;
  private readonly msEl: HTMLSpanElement;
  private readonly memEl: HTMLSpanElement;

  private isVisible = false;
  private showMem = false;

  private frameStartUnsubscribe: Unsubscribe | null = null;
  private frameEndUnsubscribe: Unsubscribe | null = null;

  // Timing state.
  private frameT0 = 0;
  private msEma = 0;
  private frameCount = 0;
  private windowStart = 0;
  private fps = 0;
  private lastRender = 0;

  constructor(private readonly keepAlive?: PerfKeepAlive) {
    const el = document.createElement('div');
    el.id = 'luxar-stats';
    el.className = 'luxar-perf';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-label', 'Performance metrics: frames per second and frame time');
    el.style.zIndex = String(config.ui.zIndex.statsMonitor);
    el.style.display = 'none';

    this.fpsEl = document.createElement('span');
    this.fpsEl.className = 'luxar-perf__stat luxar-perf__fps';
    this.msEl = document.createElement('span');
    this.msEl.className = 'luxar-perf__stat luxar-perf__ms';
    this.memEl = document.createElement('span');
    this.memEl.className = 'luxar-perf__stat luxar-perf__mem';
    this.memEl.style.display = 'none';

    this.fpsEl.textContent = '–– fps';
    this.msEl.textContent = '–– ms';
    el.append(this.fpsEl, this.msEl, this.memEl);

    this.el = el;
    getViewerContainer().appendChild(el);
  }

  private readonly onFrameStart = (): void => {
    this.frameT0 = performance.now();
  };

  private readonly onFrameEnd = (): void => {
    const now = performance.now();
    const dt = now - this.frameT0;
    // Exponential moving average smooths the per-frame jitter.
    this.msEma = this.msEma ? this.msEma * 0.9 + dt * 0.1 : dt;

    if (!this.windowStart) this.windowStart = now;
    this.frameCount++;
    const elapsed = now - this.windowStart;
    if (elapsed >= FPS_WINDOW_MS) {
      this.fps = (this.frameCount * 1000) / elapsed;
      this.frameCount = 0;
      this.windowStart = now;
    }

    if (now - this.lastRender >= RENDER_INTERVAL_MS) {
      this.render();
      this.lastRender = now;
    }
  };

  private render(): void {
    const fps = Math.round(this.fps);
    this.fpsEl.textContent = `${fps} fps`;
    this.fpsEl.dataset.level = fps >= 50 ? 'good' : fps >= 30 ? 'ok' : 'bad';
    this.msEl.textContent = `${this.msEma.toFixed(1)} ms`;
    if (this.showMem) {
      const mem = (performance as unknown as { memory?: PerfMemory }).memory;
      this.memEl.textContent = mem ? `${Math.round(mem.usedJSHeapSize / 1048576)} MB` : 'n/a';
    }
  }

  /** Toggle visibility. Subscribes to frame timing only while visible. */
  toggle(): void {
    this.isVisible = !this.isVisible;
    this.el.style.display = this.isVisible ? 'flex' : 'none';
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

  /** Cycle extra detail — toggles the JS-heap memory readout on/off. */
  cyclePanels(): void {
    if (!this.isVisible) return;
    this.showMem = !this.showMem;
    this.memEl.style.display = this.showMem ? '' : 'none';
    this.render();
  }

  dispose(): void {
    this.unsubscribeFromFrameTiming();
    if (this.isVisible) this.keepAlive?.release();
    this.el.remove();
  }

  private subscribeToFrameTiming(): void {
    if (this.frameStartUnsubscribe) return;
    // Reset the rolling counters so a re-open starts fresh.
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
