/**
 * WebGPU device-loss recovery, the counterpart of `webgl-context-recovery.ts`.
 *
 * three's `WebGPURenderer` does NOT recreate a lost device: its backend calls
 * `renderer.onDeviceLost`, which only logs and latches `_isDeviceLost`, after
 * which every render returns early. So recovery here means building a NEW
 * renderer on the same canvas (the host's `recreateRenderer`) and handing it to
 * everything that renders (post-processing, environment, picking), then
 * re-uploading the geometry from its CPU mirrors, exactly as a WebGL context
 * restore does.
 *
 * Sequence:
 *
 *   - **Loss**: the current device's `lost` promise resolves. A `'destroyed'`
 *     reason is our own `renderer.dispose()` (three ignores it too) and is not a
 *     loss. Otherwise flip `isLost`, tell the host (`onLost`: event, toast,
 *     smaller GPU budget), and start recovery.
 *   - **Recovery**: `recreateRenderer()` builds a renderer on a fresh device and
 *     rewires the host. Success re-arms the observer on the new device and calls
 *     `onRestored`. A failure is retried once after `retryDelayMs`; a second
 *     failure ends in `onUnrecoverable` (the reload prompt).
 *
 * A device that is lost again after a successful recovery starts the cycle
 * afresh: the attempt budget is per loss, not per session.
 *
 * @module scene/scene-manager/render-pipeline/webgpu-device-recovery
 */

import { log, Modules } from '../../../utils/log';

/** `GPUDeviceLostInfo`, structurally (no `@webgpu/types` dependency). */
export interface DeviceLostInfo {
  reason?: string;
  message?: string;
}

/** The slice of `GPUDevice` the observer needs. */
export interface GpuDeviceLike {
  lost: Promise<DeviceLostInfo>;
}

/** Host side-effects, passed once at construction. */
export interface WebGPUDeviceRecoveryDeps {
  /** The device the CURRENT renderer draws with; undefined on the WebGL2 bridge. */
  getDevice(): GpuDeviceLike | undefined;
  /**
   * Build a renderer on a fresh native device and rewire the host to it.
   * Resolves false (or rejects) when no native device could be obtained.
   */
  recreateRenderer(): Promise<boolean>;
  /** The device was lost (not destroyed by us); recovery starts right after. */
  onLost(info: DeviceLostInfo): void;
  /** A new renderer is live and the scene has been rewired to it. */
  onRestored(): void;
  /** Every attempt failed; the page has to reload. */
  onUnrecoverable(info: DeviceLostInfo): void;
  /** Wait before the second attempt. Injected for tests; defaults to `setTimeout`. */
  delay?: (ms: number) => Promise<void>;
  /** Pause between the two attempts (ms). */
  retryDelayMs?: number;
}

const ATTEMPTS_PER_LOSS = 2;
const DEFAULT_RETRY_DELAY_MS = 1000;

/**
 * Watches the active device and rebuilds the renderer when it is lost.
 * `attach()` arms the observer; `dispose()` disarms it (a later resolution of
 * an old `lost` promise is ignored).
 */
export class WebGPUDeviceRecovery {
  private lost = false;
  private disposed = false;
  /** Bumped per armed device, so a stale `lost` resolution is ignored. */
  private generation = 0;

  constructor(private readonly deps: WebGPUDeviceRecoveryDeps) {}

  /** Whether the device is currently lost (recovery pending or failed). */
  isLost(): boolean {
    return this.lost;
  }

  /** Observe the current renderer's device. Returns false when there is none. */
  attach(): boolean {
    const device = this.deps.getDevice();
    if (!device || typeof device.lost?.then !== 'function') return false;
    const generation = ++this.generation;
    void device.lost.then((info) => this.handleLost(info ?? {}, generation));
    return true;
  }

  /** Stop observing. Idempotent. */
  dispose(): void {
    this.disposed = true;
    this.generation++;
  }

  /**
   * Run the loss → recovery sequence as if the device had just been lost.
   * The debug surface uses it to exercise recovery in a native session, where
   * the only way to lose a device on demand (`device.destroy()`) reports
   * `'destroyed'`, which the observer deliberately ignores.
   */
  simulateLoss(info: DeviceLostInfo = { reason: 'unknown', message: 'simulated' }): Promise<void> {
    return this.handleLost(info, this.generation);
  }

  private async handleLost(info: DeviceLostInfo, generation: number): Promise<void> {
    if (this.disposed || generation !== this.generation) return;
    if (info.reason === 'destroyed') return;
    this.generation++; // this device is done: ignore anything else it reports
    this.lost = true;
    log.error(
      Modules.RENDERER,
      `WebGPU device lost (reason=${info.reason ?? 'unknown'}): ${info.message ?? '(no message)'}. ` +
        'Rebuilding the renderer on a new device.'
    );
    this.deps.onLost(info);
    const attempt = await this.recover();
    if (this.disposed) return;
    if (attempt === null) {
      log.error(Modules.RENDERER, 'WebGPU device could not be recovered; a reload is required.');
      this.deps.onUnrecoverable(info);
      return;
    }
    this.lost = false;
    this.attach();
    this.deps.onRestored();
    log.success(Modules.RENDERER, `WebGPU renderer rebuilt after device loss (attempt ${attempt})`);
  }

  /** The attempt that rebuilt the renderer, or null when none did (or we were disposed). */
  private async recover(): Promise<number | null> {
    for (let attempt = 1; attempt <= ATTEMPTS_PER_LOSS; attempt++) {
      if (attempt > 1) {
        await (this.deps.delay ?? defaultDelay)(this.deps.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS);
      }
      if (this.disposed) return null;
      if (await this.tryRecreate(attempt)) return attempt;
    }
    return null;
  }

  private async tryRecreate(attempt: number): Promise<boolean> {
    try {
      return await this.deps.recreateRenderer();
    } catch (error) {
      log.warning(Modules.RENDERER, `WebGPU renderer rebuild attempt ${attempt} failed`, error);
      return false;
    }
  }
}

function defaultDelay(ms: number): Promise<void> {
  return new Promise((resolve) => globalThis.setTimeout(resolve, ms));
}
