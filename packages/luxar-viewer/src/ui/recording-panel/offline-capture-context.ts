/**
 * Assembling what a capture driver needs: the driver itself, and the
 * {@link CaptureContext} of collaborators it runs against.
 *
 * Extracted from `OfflineCaptureStrategy.runOfflineCaptureLoop` (audit A2-01).
 * Neither of these decides anything about the capture — one is a mode→class
 * lookup, the other a dependency literal — but between them they carried a
 * third of the method's remaining branch count, so they were the difference
 * between the loop reading as stages and reading as setup.
 */

import * as THREE from 'three';
import { Modules, log } from '../../utils/log';
import { showToast } from '../toast';
import { computeVideoBitrate as computeVideoBitratePure } from './media-utilities';
import {
  generateFfmpegScript as generateFfmpegScriptPure,
  type GradeSettings,
  type ToneMapName,
} from './ffmpeg-script';
import { ImageSequenceDriver } from './drivers/image-sequence-driver';
import { ExrSequenceDriver } from './drivers/exr-sequence-driver';
import { VideoModeDriver } from './drivers/video-mode-driver';
import type { CaptureContext, OfflineCaptureDriver } from './drivers/offline-capture-driver';
import type { SceneManager } from '../../scene/scene-manager';
import type { RecordingMode, RecordingOptions } from './types';

/** Every offline output format, by the driver family that produces it. */
export type OfflineMode = 'exr' | 'webm' | 'mp4' | 'mkv' | 'png' | 'webp' | 'jpeg';

/** THREE tone-mapping constants → the names the ffmpeg script knows. */
const TONE_MAP_BY_THREE_CONSTANT: Record<number, ToneMapName> = {
  [THREE.LinearToneMapping]: 'linear',
  [THREE.ReinhardToneMapping]: 'reinhard',
  [THREE.CineonToneMapping]: 'cineon',
  [THREE.ACESFilmicToneMapping]: 'aces',
  [THREE.AgXToneMapping]: 'agx',
  [THREE.NeutralToneMapping]: 'neutral',
};

/**
 * Read the display transform an EXR capture bypasses, so the bundled
 * ffmpeg script can put it back.
 *
 * Returns `undefined` when the renderer doesn't expose the grade (older
 * mocks in tests), which is what the script's `unknown-grade` chain is
 * for: it still converts the transfer — linear floats must never go out
 * untouched — and says in the header that the curve is missing. Handing
 * it a fabricated neutral grade instead produced identical pixels but a
 * header claiming the viewer's own curve had been written out, while the
 * viewer may well have been on ACES.
 */
export function readGradeSettings(sceneManager: SceneManager): GradeSettings | undefined {
  const grade = sceneManager.postProcessing?.getGradeSettings?.();
  if (!grade) return undefined;
  return {
    toneMapping: TONE_MAP_BY_THREE_CONSTANT[grade.toneMapping] ?? 'neutral',
    exposure: grade.exposure,
    offset: grade.offset,
    gamma: grade.gamma,
  };
}

/**
 * Build the per-mode driver.
 *
 * EXR mode flips the session's `isEXRSequenceRecording` flag in finalize via
 * `onFinalized` so the driver doesn't need to know about that field.
 */
export function createCaptureDriver(
  mode: OfflineMode,
  onExrFinalized: () => void
): OfflineCaptureDriver {
  if (mode === 'png' || mode === 'webp' || mode === 'jpeg') {
    return new ImageSequenceDriver(mode);
  }
  if (mode === 'exr') {
    return new ExrSequenceDriver(onExrFinalized);
  }
  return new VideoModeDriver(mode);
}

export interface CaptureContextDeps {
  sceneManager: SceneManager;
  opts: RecordingOptions;
  recordingMode: RecordingMode;
  fps: number;
  signal: AbortSignal;
  /**
   * One timestamped stem for the whole capture. `generateFilename`
   * stamps `new Date()` on every call, so calling it per artifact —
   * the ZIP's save-dialog name at setup, its fallback download name at
   * finalize, the script's output base — hands out names that disagree
   * whenever a capture crosses a second boundary.
   */
  captureBase: string;
  renderFrameToCanvas: () => Promise<HTMLCanvasElement>;
  downloadBlob: (blob: Blob, filename: string) => void;
}

/** Build the dependency context the driver runs against. */
export function buildCaptureContext(deps: CaptureContextDeps): CaptureContext {
  const { sceneManager, opts, recordingMode, fps, captureBase } = deps;
  return {
    sceneManager,
    fps,
    renderFrameToCanvas: () => deps.renderFrameToCanvas(),
    generateFilename: (ext) => `${captureBase}.${ext}`,
    generateFfmpegScript: (frames, ext) =>
      generateFfmpegScriptPure({
        fps,
        frameCount: frames,
        frameExt: ext,
        mode: recordingMode === 'turntable' ? 'turntable' : 'video',
        outputBase: captureBase,
        // EXR frames are scene-linear and pre-grade, so the script has
        // to re-apply the viewer's display transform. LDR frames are
        // already graded and ignore this.
        grade: ext === 'exr' ? readGradeSettings(sceneManager) : undefined,
      }),
    downloadBlob: (blob, filename) => deps.downloadBlob(blob, filename),
    computeVideoBitrate: (w, h) => computeVideoBitratePure(w, h, opts.videoFPS, opts.videoQuality),
    showToast,
    logWarning: (msg) => log.warning(Modules.RECORDING, msg),
    logError: (msg) => log.error(Modules.RECORDING, msg),
    imageQuality: opts.imageQuality,
    videoCodec: opts.videoCodec,
    env: window as unknown as CaptureContext['env'],
    signal: deps.signal,
  };
}
