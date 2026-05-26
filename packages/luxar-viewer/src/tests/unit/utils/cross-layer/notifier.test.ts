/**
 * Unit tests for the cross-layer notifier surface.
 *
 * The notifier is a thin pub/sub-style indirection — lower layers call
 * `notifier.toast(...)` etc. without importing the UI helpers
 * directly. The tests register a real stub-backend (an object literal
 * recording the calls) — no mocking framework needed.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearNotifierBackend,
  notifier,
  setNotifierBackend,
  type NotifierBackend,
} from '../../../../utils/cross-layer/notifier';

interface RecordingBackend extends NotifierBackend {
  calls: Array<{ method: string; args: unknown[] }>;
}

function makeRecordingBackend(): RecordingBackend {
  const calls: RecordingBackend['calls'] = [];
  return {
    calls,
    showError(message) {
      calls.push({ method: 'showError', args: [message] });
    },
    showToast(message, durationMs) {
      calls.push({ method: 'showToast', args: [message, durationMs] });
    },
    showHelpOverlay() {
      calls.push({ method: 'showHelpOverlay', args: [] });
    },
    hideHelpOverlay() {
      calls.push({ method: 'hideHelpOverlay', args: [] });
    },
    showLoadingIndicator() {
      calls.push({ method: 'showLoadingIndicator', args: [] });
    },
    hideLoadingIndicator() {
      calls.push({ method: 'hideLoadingIndicator', args: [] });
    },
    clearError() {
      calls.push({ method: 'clearError', args: [] });
    },
  };
}

describe('notifier — without a backend registered', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    clearNotifierBackend();
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
    clearNotifierBackend();
  });

  // utils.md O7 / Phase E21: previously one `it` bundled 7
  // `.not.toThrow()` assertions across all notifier methods. A regression
  // that made `notifier.showLoading()` throw on missing-backend would
  // surface as a generic "every method returns silently..." failure
  // without naming the offending method. Parametrize via `it.each` so
  // each method is exercised in its own row.
  it.each<{ method: string; invoke: (n: typeof notifier) => void }>([
    { method: 'error', invoke: (n) => n.error('e') },
    { method: 'toast', invoke: (n) => n.toast('t') },
    { method: 'showHelp', invoke: (n) => n.showHelp() },
    { method: 'hideHelp', invoke: (n) => n.hideHelp() },
    { method: 'showLoading', invoke: (n) => n.showLoading() },
    { method: 'hideLoading', invoke: (n) => n.hideLoading() },
    { method: 'clearError', invoke: (n) => n.clearError() },
  ])('notifier.$method returns silently when no backend is registered', ({ invoke }) => {
    expect(() => invoke(notifier)).not.toThrow();
  });

  it('warns exactly once across many missing-backend calls', () => {
    notifier.error('a');
    notifier.toast('b');
    notifier.showHelp();
    notifier.hideHelp();
    notifier.showLoading();
    notifier.hideLoading();
    notifier.clearError();
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it('the warning message names the missing method', () => {
    notifier.toast('x');
    // log.warning(module, message) → console.warn('[!] [module] message')
    // so the message text is the second console.warn arg.
    const args = warnSpy.mock.calls[0];
    const text = args.map(String).join(' ');
    expect(text).toContain('notifier.toast');
    expect(text).toContain('backend registered');
  });
});

describe('notifier — with a backend registered', () => {
  let backend: RecordingBackend;

  beforeEach(() => {
    backend = makeRecordingBackend();
    setNotifierBackend(backend);
  });

  afterEach(() => {
    clearNotifierBackend();
  });

  it('forwards error()', () => {
    notifier.error('boom');
    expect(backend.calls).toEqual([{ method: 'showError', args: ['boom'] }]);
  });

  it('forwards toast() with the default duration of 2000ms when not supplied', () => {
    notifier.toast('hello');
    expect(backend.calls).toEqual([{ method: 'showToast', args: ['hello', 2000] }]);
  });

  it('forwards toast() with an explicit duration', () => {
    notifier.toast('briefly', 500);
    expect(backend.calls).toEqual([{ method: 'showToast', args: ['briefly', 500] }]);
  });

  it('forwards showHelp / hideHelp', () => {
    notifier.showHelp();
    notifier.hideHelp();
    expect(backend.calls.map((c) => c.method)).toEqual(['showHelpOverlay', 'hideHelpOverlay']);
  });

  it('forwards showLoading / hideLoading', () => {
    notifier.showLoading();
    notifier.hideLoading();
    expect(backend.calls.map((c) => c.method)).toEqual([
      'showLoadingIndicator',
      'hideLoadingIndicator',
    ]);
  });

  it('forwards clearError()', () => {
    notifier.clearError();
    expect(backend.calls).toEqual([{ method: 'clearError', args: [] }]);
  });

  it('does NOT warn when the backend is registered', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    notifier.toast('x');
    notifier.showHelp();
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});

describe('notifier — backend lifecycle', () => {
  afterEach(() => clearNotifierBackend());

  it('replacing the backend reroutes subsequent calls to the new one', () => {
    const a = makeRecordingBackend();
    const b = makeRecordingBackend();

    setNotifierBackend(a);
    notifier.toast('one');
    expect(a.calls.length).toBe(1);
    expect(b.calls.length).toBe(0);

    setNotifierBackend(b);
    notifier.toast('two');
    expect(a.calls.length).toBe(1);
    expect(b.calls.length).toBe(1);
    expect(b.calls[0].args[0]).toBe('two');
  });

  it('clearNotifierBackend() makes subsequent calls fall back to no-op + warn', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const backend = makeRecordingBackend();
    setNotifierBackend(backend);
    notifier.toast('with-backend');
    expect(backend.calls.length).toBe(1);

    clearNotifierBackend();
    notifier.toast('without-backend');
    expect(backend.calls.length).toBe(1); // not delivered
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });

  it('re-registering after a missing-backend warning resets the warn-once flag', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    clearNotifierBackend();
    notifier.toast('first'); // triggers warn
    expect(warnSpy).toHaveBeenCalledTimes(1);

    const backend = makeRecordingBackend();
    setNotifierBackend(backend);
    notifier.toast('delivered');
    expect(backend.calls.length).toBe(1);

    clearNotifierBackend();
    notifier.toast('second-missing'); // should warn again because flag was reset on register
    expect(warnSpy).toHaveBeenCalledTimes(2);

    warnSpy.mockRestore();
  });
});
