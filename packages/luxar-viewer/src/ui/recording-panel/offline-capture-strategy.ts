/**
 * Offline (frame-by-frame) capture strategy.
 *
 * Used for:
 * - EXR sequence recording (ZIP of EXR frames, full float precision)
 * - Turntable smooth mode (renders each frame individually for perfectly
 *   smooth output regardless of GPU FPS)
 *
 * Unlike real-time MediaRecorder capture, this loop is decoupled from
 * the browser's animation frame RATE — but not from the animation loop
 * itself: step 1 runs as a per-frame callback, so the rAF loop has to
 * be running (see the wake-up in `runOfflineCaptureLoop`). Each frame
 * is:
 * 1. Camera orbited by one step (quaternion rotation, same as auto-rotate)
 * 2. One mandatory rAF (so the LOD selector, which runs BEFORE the orbit
 *    callback in the same frame, has evaluated the new pose) plus a
 *    bounded wait for it to settle (`hooks.isLODSettled`) — the rAF loop
 *    is live for the whole sweep, so a tile that left the frustum
 *    mid-orbit can be mid-reload when it swings back, and capturing
 *    immediately bakes a coarse-level pop into the sequence (#1695).
 *    Skipped entirely — that mandatory rAF included — when the hook
 *    answers `null`, i.e. this scene has no lod_group to wait for
 * 3. Scene rendered (full pipeline, into the capture's own target)
 * 4. Pixels read back asynchronously (PBO fence on WebGL2, mapAsync on
 *    WebGPU) — the rAF loop keeps ticking through the await, which is
 *    why the loop's own render is suppressed for the whole capture
 *    (see the render-skip predicate wired in `core/app/init/pipeline`)
 * 5. Frame stored / encoded
 * 6. Brief yield to keep the browser responsive
 *
 * The loop owns shared scaffolding (state save/restore via Session,
 * modal overlay, animation pump, progress display, error tolerance);
 * per-mode capture (PNG/WebP/JPEG sequence, video container, EXR
 * sequence) is delegated to a driver implementing
 * {@link OfflineCaptureDriver}. Each driver runs its own setup,
 * captures one frame at a time, and finalizes (download/save).
 *
 * Critical correctness invariants:
 * 1. AbortController is assigned BEFORE any state mutation — dispose()
 *    during the early state-save / rAF window must abort the session.
 * 2. The body from driver.setup onward is wrapped in try/finally so an
 *    exception from driver.setup, driver.captureFrame, driver.finalize,
 *    or any DOM/state mutation cannot leave the panel with a stuck
 *    overlay, hidden panels, scaled renderer, or stale recording flags.
 *    The earlier window — panel hide / saveRecordingState through the
 *    overlay construction — is NOT covered (the finally closes over
 *    bindings that window creates), which is why
 *    isLoopRenderSuppressed is raised inside the try: a stuck value
 *    there blanks the whole viewport, where the other flags only lock
 *    further captures or leave the panel looking wrong. That window is
 *    synchronous DOM construction with no production-reachable throw —
 *    showConfirmationDialog above it already assigns innerHTML, so an
 *    environment that forbids it fails before any state is mutated.
 * 3. The finally block is idempotent — every removal/restore handles
 *    the "wasn't set" case gracefully.
 * 4. The LOD settle drain (step 2) runs AFTER the orbit callback has
 *    been removed. The camera pose is fixed by then, so the extra
 *    frames only let pending loads land; draining before the removal
 *    would keep orbiting the camera while waiting and smear the sweep.
 */

import { log, Modules } from '../../utils/log';
import { showToast } from '../toast';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import {
  buildCaptureContext,
  createCaptureDriver,
  type OfflineMode,
} from './offline-capture-context';
import { runFrameLoop } from './offline-capture-frame-loop';
import { createOfflineCaptureOverlay } from './offline-capture-overlay';
import { runCapturePreflight } from './offline-capture-preflight';
import { runCaptureTeardown } from './offline-capture-teardown';
import { LodSettleDrain } from './offline-lod-settle';
import type { CaptureStrategy, SessionState } from './capture-strategy';
import type { RecordingSession } from './session';
import type { RecordingMode, RecordingOptions } from './types';

export interface OfflineCaptureStrategyHooks {
  hideAllPanels(): void;
  renderFrameToCanvas(): Promise<HTMLCanvasElement>;
  downloadBlob(blob: Blob, filename: string): void;
  generateFilename(ext: string): string;
  /**
   * Whether every in-frame LOD group or partition part is at final committed
   * quality (`LODGroupRegistry.isCaptureQuiescent()`). The capture drains on
   * this before grabbing each frame, so an asynchronous fine-level reload or
   * partition resync kicked when a tile swings back into the frustum mid-orbit
   * cannot be filmed at a coarse fallback, stale slice, or empty state (#1695,
   * #2633).
   *
   * TRI-STATE, and the third state is what keeps a plain points/lines scene
   * free:
   * - `true` / `false` — this scene HAS level-of-detail or partition groups, so the loop
   *   drains. It then always spends at least one extra rAF per frame even when
   *   the answer is already `true`, because the selector runs before the orbit
   *   callback within a frame and so is a pose behind until it ticks once more.
   * - `null` — this scene has no lod_group or partition to wait for (no
   *   registry, or a registry with neither). The loop skips the drain entirely, INCLUDING
   *   that mandatory tick, which is the pre-#1695 behaviour to the frame. Note
   *   the narrowness: `null` is NOT "nothing here could ever be mid-load". A
   *   `--recipe stream` scene — one leaf with an additive ladder and no
   *   lod_group — answers `null` while its progressive refinement is still
   *   climbing the ladder, so its early frames can be exported at a partial
   *   prefix. Same artifact class, not covered by this drain.
   *
   * Absent ⇒ identical to `null`. The panel wires this unconditionally and the
   * pipeline's provider is the one that answers `null`, so "the hook exists"
   * must never be read as "this scene needs draining".
   */
  isLODSettled?(): boolean | null;
}

export class OfflineCaptureStrategy implements CaptureStrategy {
  readonly kind = 'offline' as const;

  /**
   * AbortController for the offline-capture session. Set immediately
   * after the confirmation check, BEFORE any state mutation, so
   * dispose() during the early state-save / rAF window can abort the
   * in-flight session.
   */
  sessionAbort: AbortController | null = null;
  overlayCleanup: (() => void) | null = null;

  /** Offline-capture callback IDs — static so dispose() can remove them
   *  unconditionally even if the loop is parked on an `await` and
   *  hasn't reached its finally yet. */
  static readonly CAPTURE_CALLBACK_ID = 'recording-offline-capture';
  static readonly KEEPALIVE_CALLBACK_ID = 'recording-offline-keepalive';

  constructor(
    private readonly sceneManager: SceneManager,
    private readonly animationController: AnimationController,
    private readonly hooks: OfflineCaptureStrategyHooks
  ) {}

  canRun(state: SessionState): boolean {
    return !state.isRecording && !state.isOfflineCaptureActive;
  }

  async run(opts: RecordingOptions, mode: RecordingMode, session: RecordingSession): Promise<void> {
    return this.runOfflineCaptureLoop(opts.outputFormat as OfflineMode, opts, session, mode);
  }

  /** Synchronously stop the loop. The loop's await checkpoints
   *  observe `sessionAbort.signal` and break out cleanly. */
  abort(): void {
    this.sessionAbort?.abort('user-stop');
  }

  dispose(): void {
    this.sessionAbort?.abort('disposed');
    this.overlayCleanup?.();
    // The normal loop path removes these in its finally; this covers
    // the dispose-while-awaiting case where the loop hasn't reached
    // its finally yet.
    this.animationController.removePerFrameCallback(OfflineCaptureStrategy.CAPTURE_CALLBACK_ID);
    this.animationController.removePerFrameCallback(OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID);
  }

  // ── Test-only access (Panel proxies forward to these) ─────────
  cleanupOfflineOverlayForTests(): void {
    this.overlayCleanup?.();
  }

  private async runOfflineCaptureLoop(
    mode: OfflineMode,
    opts: RecordingOptions,
    session: RecordingSession,
    recordingMode: RecordingMode
  ): Promise<void> {
    const plan = await runCapturePreflight(mode, opts, recordingMode, {
      session,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      hideAllPanels: () => this.hooks.hideAllPanels(),
      claimSession: (abort) => {
        this.sessionAbort = abort;
      },
      releaseSession: (abort) => {
        if (this.sessionAbort === abort) {
          this.sessionAbort = null;
        }
      },
    });
    if (!plan) return;
    // The rotation/dolly schedule stays inside `plan` — `runFrameLoop` is the
    // only thing that plays it, and it takes the plan whole.
    const { sessionAbort, fps, totalFrames } = plan;

    const driver = createCaptureDriver(mode, () => {
      session.isEXRSequenceRecording = false;
    });

    if (mode === 'exr') {
      session.isEXRSequenceRecording = true;
    }
    session.isRecording = true;
    session.isOfflineCaptureActive = true;
    session.recordingStartTime = Date.now();
    session.showRecordingIndicator();

    const handleCancel = (): void => {
      session.isRecording = false;
      sessionAbort.abort('user-cancel');
    };
    const overlayUi = createOfflineCaptureOverlay(totalFrames, handleCancel, (self) => {
      if (this.overlayCleanup === self) {
        this.overlayCleanup = null;
      }
    });
    const cleanupOfflineOverlay = overlayUi.cleanup;
    this.overlayCleanup = cleanupOfflineOverlay;

    const ctx = buildCaptureContext({
      sceneManager: this.sceneManager,
      opts,
      recordingMode,
      fps,
      signal: sessionAbort.signal,
      captureBase: this.hooks.generateFilename('zip').replace(/\.zip$/, ''),
      renderFrameToCanvas: () => this.hooks.renderFrameToCanvas(),
      downloadBlob: (blob, filename) => this.hooks.downloadBlob(blob, filename),
    });

    const progress = overlayUi.progress;

    const captureCallbackId = OfflineCaptureStrategy.CAPTURE_CALLBACK_ID;
    const keepAliveId = OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID;
    let setupCompleted = false;
    let finalizeSucceeded = false;

    try {
      // The loop's own render is redundant from here on — the capture
      // renders its own pipeline pass per frame (see the render-skip
      // predicate wired in `core/app/init/pipeline`). Set INSIDE the try so
      // the finally below always clears it: the flag suppresses the loop's
      // render globally, so escaping with it stuck true blanks the viewport
      // until a page reload. Nothing between the recording flags above and
      // this point can paint a frame — the REC indicator, the overlay and the
      // driver context are all built synchronously, and the last yield is the
      // rAF well above them.
      session.isLoopRenderSuppressed = true;
      const setupOk = await driver.setup(ctx);
      if (!setupOk) {
        return;
      }
      setupCompleted = true;
      if (sessionAbort.signal.aborted) return;

      // The per-frame LOD settle wait (#1695) and its run-scoped
      // bookkeeping. See offline-lod-settle.ts -- the tri-state probe, the
      // latch/re-arm and the end-of-run report all live there now.
      const lodSettle = new LodSettleDrain({
        isLODSettled: this.hooks.isLODSettled?.bind(this.hooks),
        isRecording: () => session.isRecording,
        isAborted: () => sessionAbort.signal.aborted,
        nextFrame: () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())),
      });

      // Wake the rAF loop, exactly as the real-time strategy does. The
      // turntable's rotation is applied from a per-frame callback, and
      // those only run while the loop is animating — but the loop
      // idle-stops after ~2s of no interaction, which is the normal
      // state by the time the user has read the panel and confirmed the
      // dialog. Registering a `continuous` callback only KEEPS a running
      // loop alive; it never restarts a stopped one. Without this call
      // the camera never rotates and the capture silently emits N
      // identical frames — the capture path renders its own pipeline
      // pass (`renderToImageData`), so frames are still produced, just
      // all from the same pose.
      //
      // Don't "optimize" this away by orbiting the camera inline and
      // leaving the loop stopped: the depth-sort scheduler and the LOD
      // group selector are per-frame callbacks too, so a stopped loop
      // would freeze depth order and LOD level at the opening pose
      // while the camera swings a full turn.
      this.animationController.startAnimation();
      this.animationController.addPerFrameCallback(keepAliveId, () => {}, { continuous: true });

      const { capturedFrames, attemptedFrames } = await runFrameLoop({
        plan,
        session,
        driver,
        ctx,
        progress,
        lodSettle,
        animationController: this.animationController,
        captureCallbackId,
        setFrameCount: overlayUi.setFrameCount,
      });

      if (sessionAbort.signal.aborted) {
        return;
      }

      const lodReport = lodSettle.report(capturedFrames, attemptedFrames);
      if (lodReport) showToast(lodReport);

      try {
        await driver.finalize(ctx, capturedFrames, progress);
        finalizeSucceeded = true;
      } catch (err) {
        log.error(Modules.RECORDING, `Offline ${mode} finalize failed: ${err}`);
        showToast('Recording finalize failed');
      }
    } catch (err) {
      log.error(Modules.RECORDING, `Offline ${mode} capture failed: ${err}`);
      showToast('Recording failed');
    } finally {
      await runCaptureTeardown({
        session,
        sessionAbort,
        driver,
        ctx,
        setupCompleted,
        finalizeSucceeded,
        animationController: this.animationController,
        callbackIds: [captureCallbackId, keepAliveId],
        cleanupOverlay: cleanupOfflineOverlay,
        releaseSession: (abort) => {
          if (this.sessionAbort === abort) {
            this.sessionAbort = null;
          }
        },
      });
    }
  }
}
