import type { EventGroup } from '../../../utils/cross-layer/event-group';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { RecordingPanel } from '../../../ui/recording-panel';
import { log, Modules } from '../../../utils/log';

/**
 * Register window-focus and document-visibility listeners that pause /
 * resume the animation loop as the page goes background and back. Both
 * listeners early-return when {@link FocusHandlingPorts.recordingPanel}
 * reports an active capture, so offline recording keeps a stable loop.
 *
 * Routed through the supplied {@link EventGroup} so the listeners are
 * cleaned up on dispose.
 */
export interface FocusHandlingPorts {
  events: EventGroup;
  animationController: AnimationController;
  recordingPanel: RecordingPanel | undefined;
}

export function installFocusHandling(ports: FocusHandlingPorts): void {
  ports.events.on(window, 'focus', () => {
    // Suppress focus-triggered renders during recording — they can interfere
    // with the deterministic capture loop or cause resize side effects
    if (ports.recordingPanel?.isCurrentlyRecording()) return;
    ports.animationController.startAnimation();
    log.info(Modules.LUXAR, 'Window focused - triggering render refresh');
  });
  ports.events.on(document, 'visibilitychange', () => {
    // Don't stop animation during recording (offline capture needs the loop alive)
    if (ports.recordingPanel?.isCurrentlyRecording()) return;
    if (document.hidden) {
      ports.animationController.stopAnimation();
      log.info(Modules.LUXAR, 'Document hidden - stopping animation to save resources');
    } else {
      ports.animationController.startAnimation();
      log.info(Modules.LUXAR, 'Document became visible - resuming animation');
    }
  });
}
