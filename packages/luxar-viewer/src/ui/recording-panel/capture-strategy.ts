/**
 * Capture-strategy contract — the Strategy half of the Strategy + Session
 * decomposition.
 *
 * Each implementation (ScreenshotStrategy, VideoRecordingStrategy,
 * OfflineCaptureStrategy) owns a single capture pipeline (screenshot,
 * real-time MediaRecorder, frame-by-frame offline loop) and runs against
 * a shared `RecordingSession` for cross-cutting scaffolding (state
 * save/restore, dialog, indicator, mutual-exclusion).
 *
 * The interface is intentionally tiny:
 * - `canRun` lets `RecordingPanel`'s dispatch ask whether the strategy
 *   could start in the current session state (informational; the strategy
 *   itself still verifies and reserves inside `run`).
 * - `run` is the capture operation. It returns when the capture finishes
 *   (or is aborted). Strategies are responsible for their own internal
 *   try/finally cleanup, including releasing the session reservation.
 * - `abort` is a synchronous external request to stop. For real-time
 *   video this stops MediaRecorder; for offline it fires an AbortController;
 *   for screenshot it's a no-op (the operation is short).
 * - `dispose` is the panel-shutting-down signal. Strategies use it to
 *   release any long-lived state (track refs, AbortController, …).
 */

import type { RecordingMode, RecordingOptions } from './types';
import type { RecordingSession } from './session';

export type CaptureKind = 'screenshot' | 'video' | 'offline';

/**
 * Read-only view of the session state passed to `canRun`. Strategies use
 * this to refuse to start when another capture is active.
 */
export interface SessionState {
  isRecording: boolean;
  isOfflineCaptureActive: boolean;
  isCaptureInProgress: boolean;
}

export interface CaptureStrategy {
  readonly kind: CaptureKind;
  canRun(state: SessionState): boolean;
  run(opts: RecordingOptions, mode: RecordingMode, session: RecordingSession): Promise<void>;
  abort(): void;
  dispose(): void;
}
