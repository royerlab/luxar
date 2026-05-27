/**
 * Unit tests for core/app/lifecycle/focus-handling.ts.
 *
 * The handler wires two listeners on the supplied EventGroup:
 *   - window 'focus' → startAnimation (unless recording)
 *   - document 'visibilitychange' → start/stop based on document.hidden
 *     (unless recording)
 *
 * The key invariant under test is that `getRecordingPanel()` is a LIVE
 * accessor (not a snapshot). A focus event fired after the orchestrator
 * cleared its recordingPanel field must see `undefined` and the
 * optional-chain must short-circuit — otherwise we'd call
 * `.isCurrentlyRecording()` on a disposed RecordingPanel.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { installFocusHandling } from '../../../../../core/app/lifecycle/focus-handling';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import type { AnimationController } from '../../../../../scene/animation/animation-controller';
import type { RecordingPanel } from '../../../../../ui/recording-panel';

interface AnimationStub {
  startAnimation: ReturnType<typeof vi.fn>;
  stopAnimation: ReturnType<typeof vi.fn>;
}

interface RecordingStub {
  isCurrentlyRecording: ReturnType<typeof vi.fn>;
}

function makeAnimation(): AnimationStub {
  return {
    startAnimation: vi.fn(),
    stopAnimation: vi.fn(),
  };
}

function makeRecording(isRecording = false): RecordingStub {
  return {
    isCurrentlyRecording: vi.fn().mockReturnValue(isRecording),
  };
}

const asAnim = (a: AnimationStub) => a as unknown as AnimationController;
const asRec = (r: RecordingStub) => r as unknown as RecordingPanel;

function setHidden(hidden: boolean): void {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden });
}

describe('installFocusHandling', () => {
  let events: EventGroup;
  let anim: AnimationStub;
  let rec: RecordingStub;

  beforeEach(() => {
    events = new EventGroup();
    anim = makeAnimation();
    rec = makeRecording(false);
    setHidden(false);
  });

  // EventGroup-on-window listeners leak across tests in the same jsdom
  // unless explicitly disposed. The describe-scoped `rec` mutates per
  // beforeEach, so a leaked closure from a prior test would read the
  // CURRENT test's rec and inflate the call counts.
  afterEach(() => {
    events.dispose();
  });

  describe('focus event', () => {
    it('startAnimation when window gains focus and not recording', () => {
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => asRec(rec),
      });

      window.dispatchEvent(new Event('focus'));

      expect(anim.startAnimation).toHaveBeenCalledOnce();
    });

    it('NO startAnimation when window gains focus during a recording', () => {
      rec = makeRecording(true);
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => asRec(rec),
      });

      window.dispatchEvent(new Event('focus'));

      expect(anim.startAnimation).not.toHaveBeenCalled();
    });

    it('startAnimation when recordingPanel is undefined and focus fires (optional-chain short-circuit treats as not-recording)', () => {
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => undefined,
      });

      expect(() => window.dispatchEvent(new Event('focus'))).not.toThrow();
      expect(anim.startAnimation).toHaveBeenCalledOnce();
    });
  });

  describe('visibilitychange event', () => {
    it('document.hidden=true → stopAnimation', () => {
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => asRec(rec),
      });

      setHidden(true);
      document.dispatchEvent(new Event('visibilitychange'));

      expect(anim.stopAnimation).toHaveBeenCalledOnce();
      expect(anim.startAnimation).not.toHaveBeenCalled();
    });

    it('document.hidden=false → startAnimation', () => {
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => asRec(rec),
      });

      setHidden(false);
      document.dispatchEvent(new Event('visibilitychange'));

      expect(anim.startAnimation).toHaveBeenCalledOnce();
      expect(anim.stopAnimation).not.toHaveBeenCalled();
    });

    it('recording active → NO start/stop regardless of hidden state', () => {
      rec = makeRecording(true);
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => asRec(rec),
      });

      setHidden(true);
      document.dispatchEvent(new Event('visibilitychange'));
      setHidden(false);
      document.dispatchEvent(new Event('visibilitychange'));

      expect(anim.startAnimation).not.toHaveBeenCalled();
      expect(anim.stopAnimation).not.toHaveBeenCalled();
    });
  });

  describe('live recordingPanel access (regression guard)', () => {
    it('getRecordingPanel is invoked per event, not captured at install time', () => {
      // Simulate the orchestrator's `() => this.recordingPanel` accessor.
      let current: RecordingStub | undefined = rec;
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => (current ? asRec(current) : undefined),
      });

      // First focus: recording panel returns false (not recording) → startAnimation.
      window.dispatchEvent(new Event('focus'));
      expect(anim.startAnimation).toHaveBeenCalledTimes(1);
      expect(rec.isCurrentlyRecording).toHaveBeenCalledTimes(1);

      // Now mimic dispose clearing the recordingPanel field. A late focus
      // event must NOT call .isCurrentlyRecording() on the cleared panel.
      current = undefined;
      window.dispatchEvent(new Event('focus'));

      // Still only the first invocation against the original panel.
      expect(rec.isCurrentlyRecording).toHaveBeenCalledTimes(1);
      // startAnimation fires again (optional-chain short-circuited).
      expect(anim.startAnimation).toHaveBeenCalledTimes(2);
    });
  });

  describe('EventGroup teardown', () => {
    it('events.dispose() removes both listeners', () => {
      installFocusHandling({
        events,
        animationController: asAnim(anim),
        getRecordingPanel: () => asRec(rec),
      });

      events.dispose();

      window.dispatchEvent(new Event('focus'));
      document.dispatchEvent(new Event('visibilitychange'));

      expect(anim.startAnimation).not.toHaveBeenCalled();
      expect(anim.stopAnimation).not.toHaveBeenCalled();
    });
  });
});
