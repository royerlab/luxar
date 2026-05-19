/**
 * Screenshot capture flow — extracted from `recording-panel.ts`.
 *
 * The flow coordinates: refusal during active recording, capture-in-
 * progress debounce, panel hide, DPR override, transparent-background
 * toggle, EXR vs raster path selection, and state restore. The panel
 * itself remains the owner of mode state; this function only operates
 * via the narrow `ScreenshotCtx` it receives.
 */

import * as THREE from 'three';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import { log, Modules } from '../../../utils/log';
import { showToast } from '../../toast';
import { normalizeScreenshotFormat, encodeScreenshotBlob } from '../screenshot-exporter';
import type { RecordingOptions } from '../types';

export interface ScreenshotCtx {
  readonly isRecording: boolean;
  readonly isOfflineCaptureActive: boolean;
  isCaptureInProgress: boolean;
  readonly options: RecordingOptions;
  readonly sceneManager: SceneManager;
  readonly adaptiveDPRManager: AdaptiveDPRManager | null;
  hideAllPanels(): void;
  saveRecordingState(opts: { disableDPR: boolean }): void;
  restoreRecordingState(): void;
  downloadBlob(blob: Blob, filename: string): void;
  generateFilename(ext: string): string;
  renderFrameToCanvas(): Promise<HTMLCanvasElement>;
}

export async function captureScreenshot(ctx: ScreenshotCtx): Promise<void> {
  // Refuse during active recording. captureScreenshot() and the
  // recording paths share `savedRecordingState` — without this guard,
  // a screenshot during recording would clobber the active session's
  // saved DPR/resize/renderer snapshot and the recording's eventual
  // restoreRecordingState() would no-op, leaving DPR disabled and
  // resize locked after recording ends.
  if (ctx.isRecording || ctx.isOfflineCaptureActive) {
    showToast('Stop recording before taking a screenshot');
    return;
  }
  if (ctx.isCaptureInProgress) return;
  ctx.isCaptureInProgress = true;

  let savedBackground: THREE.Color | THREE.Texture | null = null;

  try {
    log.info(Modules.RECORDING, 'Capturing screenshot...');

    ctx.hideAllPanels();
    await new Promise((r) => requestAnimationFrame(r));

    // Save state (always — ensures restoreRecordingState restores panels)
    const wantMaxDPR = ctx.options.maxDPR && !!ctx.adaptiveDPRManager;
    ctx.saveRecordingState({ disableDPR: wantMaxDPR });
    if (wantMaxDPR) {
      await new Promise((r) => requestAnimationFrame(r));
    }

    // Set transparent background
    if (ctx.options.transparentBackground) {
      savedBackground = ctx.sceneManager.scene.background as THREE.Color | THREE.Texture | null;
      ctx.sceneManager.scene.background = null;
    }

    const format = ctx.options.outputFormat;

    if (format === 'exr') {
      const exrData = await ctx.sceneManager.postProcessing.captureHDRAsEXR();
      const blob = new Blob([exrData as BlobPart], { type: 'application/octet-stream' });
      ctx.downloadBlob(blob, ctx.generateFilename('exr'));
      showToast('HDR screenshot saved (EXR)');
    } else {
      const captureCanvas = await ctx.renderFrameToCanvas();

      const { format: effectiveFormat, warning } = normalizeScreenshotFormat(
        format,
        ctx.options.transparentBackground
      );
      if (warning === 'video-fallback') {
        log.warning(
          Modules.RECORDING,
          `Screenshot format '${format}' is a video format, falling back to PNG`
        );
      } else if (warning === 'jpeg-no-alpha') {
        showToast('Switched to PNG (JPEG has no alpha)');
      }

      const blob = await encodeScreenshotBlob(
        captureCanvas,
        effectiveFormat,
        ctx.options.imageQuality
      );

      if (blob) {
        ctx.downloadBlob(blob, ctx.generateFilename(effectiveFormat));
        showToast('Screenshot saved');
      } else {
        showToast('Screenshot failed');
      }
    }
  } finally {
    if (savedBackground !== null) {
      ctx.sceneManager.scene.background = savedBackground;
    }
    ctx.restoreRecordingState();
    ctx.isCaptureInProgress = false;
  }
}
