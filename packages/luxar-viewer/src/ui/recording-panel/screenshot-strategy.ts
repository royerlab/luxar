/**
 * Screenshot capture strategy.
 *
 * Single capture path: hide panels → save state → optionally null the
 * scene background for transparent output → capture → encode → download.
 * The transparent-background restore and `restoreRecordingState()`
 * always run in a finally so the panel never gets stuck in a half-captured
 * state if encoding throws.
 *
 * The `inProgress` flag is a debounce — the G keyboard shortcut and
 * the Capture button can fire in rapid succession and a second
 * captureScreenshot() entering while the first is still encoding would
 * race the save/restore-state pair (corrupting the renderer).
 */

import * as THREE from 'three';
import { log, Modules } from '../../utils/log';
import { showToast } from '../toast';
import type { SceneManager } from '../../scene/scene-manager';
import { getMaxPixelRatio } from '../../rendering/pixel-ratio-cap';
import {
  renderFrameToCanvas as renderFrameToCanvasHelper,
  encodeScreenshotBlob,
  normalizeScreenshotFormat,
} from './screenshot-exporter';
import type { CaptureStrategy, SessionState } from './capture-strategy';
import type { RecordingSession } from './session';
import type { RecordingMode, RecordingOptions } from './types';

/**
 * Hooks the Panel must expose so the strategy can:
 * - `hideAllPanels`: hide the recording panel + others before capture
 * - `downloadBlob`: trigger a file download (routed through Panel's
 *   wrapper so existing test spies still see the call)
 * - `generateFilename`: format the timestamped filename (kept on Panel
 *   so the format stays consistent across capture paths)
 *
 * Kept as a callback object (not a Panel reference) so the strategy has
 * zero structural coupling to the orchestrator.
 */
export interface ScreenshotStrategyHooks {
  hideAllPanels(): void;
  downloadBlob(blob: Blob, filename: string): void;
  generateFilename(ext: string): string;
}

export class ScreenshotStrategy implements CaptureStrategy {
  readonly kind = 'screenshot' as const;

  private inProgress: boolean = false;

  constructor(
    private readonly sceneManager: SceneManager,
    private readonly hooks: ScreenshotStrategyHooks
  ) {}

  canRun(state: SessionState): boolean {
    if (state.isRecording || state.isOfflineCaptureActive) return false;
    if (this.inProgress) return false;
    return true;
  }

  async run(
    opts: RecordingOptions,
    _mode: RecordingMode,
    session: RecordingSession
  ): Promise<void> {
    // Refuse during active recording. captureScreenshot() and the
    // recording paths share `savedRecordingState` — without this guard,
    // a screenshot during recording would clobber the active session's
    // saved DPR/resize/renderer snapshot and the recording's eventual
    // restoreRecordingState() would no-op, leaving DPR disabled and
    // resize locked after recording ends.
    if (session.isRecording || session.isOfflineCaptureActive) {
      showToast('Stop recording before taking a screenshot');
      return;
    }
    // MED-49 (audit-ack): the `inProgress` lock MUST be acquired here
    // BEFORE the first `await` (the `requestAnimationFrame` below).
    // The "debounces concurrent screenshot requests" test in
    // `tests/unit/ui/recording-panel/screenshot-strategy.test.ts`
    // depends on this ordering: two synchronous `run()` invocations
    // (G keypress + Capture-button click) reach this guard before
    // either has yielded, so the second short-circuits via `return`.
    // If the lock acquisition ever moves below an `await`, the second
    // call would slip past the guard and clobber the saved
    // recording/DPR state. Keep the lock here; do not lazily acquire
    // it inside the try-block.
    if (this.inProgress) return;
    this.inProgress = true;

    let savedBackground: THREE.Color | THREE.Texture | null = null;

    try {
      log.info(Modules.RECORDING, 'Capturing screenshot...');

      this.hooks.hideAllPanels();
      await new Promise((r) => requestAnimationFrame(r));

      // Save state (always — ensures restoreRecordingState restores panels).
      //
      // The capture is pinned to `captureDPR` whenever there is a manager
      // to freeze adaptation with. That is normally the on-screen ceiling
      // (so the file matches the viewport), but pinning it explicitly
      // still matters even then: it stops the adaptive loop moving the
      // resolution mid-capture.
      const pinDPR = session.adaptiveDPRManager
        ? (opts.captureDPR ?? getMaxPixelRatio())
        : undefined;
      session.saveRecordingState({ captureDPR: pinDPR });
      if (pinDPR !== undefined) {
        await new Promise((r) => requestAnimationFrame(r));
      }

      // Set transparent background
      if (opts.transparentBackground) {
        savedBackground = this.sceneManager.scene.background as THREE.Color | THREE.Texture | null;
        this.sceneManager.scene.background = null;
      }

      const format = opts.outputFormat;

      if (format === 'exr') {
        const exrData = await this.sceneManager.postProcessing.captureHDRAsEXR();
        const blob = new Blob([exrData as BlobPart], { type: 'application/octet-stream' });
        this.hooks.downloadBlob(blob, this.hooks.generateFilename('exr'));
        showToast('HDR screenshot saved (EXR)');
      } else {
        const captureCanvas = await renderFrameToCanvasHelper(
          this.sceneManager.postProcessing,
          opts.includeOverlays,
          session.overlayManager,
          this.sceneManager.renderer.domElement
        );

        const { format: effectiveFormat, warning } = normalizeScreenshotFormat(
          format,
          opts.transparentBackground
        );
        if (warning === 'video-fallback') {
          log.warning(
            Modules.RECORDING,
            `Screenshot format '${format}' is a video format, falling back to PNG`
          );
        } else if (warning === 'jpeg-no-alpha') {
          showToast('Switched to PNG (JPEG has no alpha)');
        }

        const blob = await encodeScreenshotBlob(captureCanvas, effectiveFormat, opts.imageQuality);

        if (blob) {
          this.hooks.downloadBlob(blob, this.hooks.generateFilename(effectiveFormat));
          showToast('Screenshot saved');
        } else {
          showToast('Screenshot failed');
        }
      }
    } finally {
      if (savedBackground !== null) {
        this.sceneManager.scene.background = savedBackground;
      }
      session.restoreRecordingState();
      this.inProgress = false;
    }
  }

  abort(): void {
    // No-op — screenshot captures are short and synchronous from the
    // user's standpoint. The browser may still be encoding the blob,
    // but there's nothing meaningful to cancel.
  }

  dispose(): void {
    this.inProgress = false;
  }

  /** Test-facing: lets the in-progress flag be probed/forced from tests. */
  isInProgress(): boolean {
    return this.inProgress;
  }
}
