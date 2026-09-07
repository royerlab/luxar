// @vitest-environment jsdom
/**
 * Unit tests for `installCanvasGestureOwnership`.
 *
 * jsdom does not model `touch-action`, so `getComputedStyle` is stubbed where
 * the test needs a specific computed value; the inline-style writes and the
 * Safari gesture listeners are asserted directly.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import { installCanvasGestureOwnership } from '../../../../../core/app/interaction/canvas-gesture-ownership';

const profile = { touchPoints: 0 };
vi.mock('../../../../../utils/input-capabilities', () => ({
  getInputProfile: () => profile,
}));

function stubComputedTouchAction(value: string): () => void {
  const original = window.getComputedStyle;
  const stub = vi.fn((el: Element) => {
    const real = original.call(window, el);
    return new Proxy(real, {
      get(target, prop) {
        if (prop === 'touchAction') return value;
        return Reflect.get(target, prop);
      },
    });
  });
  Object.defineProperty(window, 'getComputedStyle', { configurable: true, value: stub });
  return () =>
    Object.defineProperty(window, 'getComputedStyle', { configurable: true, value: original });
}

function recordingStyle(): CSSStyleDeclaration {
  const props = new Map<string, string>();
  const style = {
    setProperty: (name: string, value: string) => void props.set(name, value),
    getPropertyValue: (name: string) => props.get(name) ?? '',
  } as Record<string, unknown>;
  for (const [camel, kebab] of [
    ['touchAction', 'touch-action'],
    ['userSelect', 'user-select'],
  ] as const) {
    Object.defineProperty(style, camel, {
      get: () => props.get(kebab) ?? '',
      set: (v: string) => {
        props.set(kebab, v);
      },
    });
  }
  return style as unknown as CSSStyleDeclaration;
}

describe('installCanvasGestureOwnership', () => {
  let canvas: HTMLCanvasElement;
  let events: EventGroup;
  let restore: (() => void) | undefined;

  beforeEach(() => {
    canvas = document.createElement('canvas');
    // jsdom's CSSStyleDeclaration drops properties it does not model
    // (touch-action, -webkit-touch-callout), so give the element a recording
    // style object with the same surface the module writes to.
    Object.defineProperty(canvas, 'style', { configurable: true, value: recordingStyle() });
    document.body.appendChild(canvas);
    events = new EventGroup();
    profile.touchPoints = 0;
  });

  afterEach(() => {
    events.dispose();
    restore?.();
    restore = undefined;
    document.body.innerHTML = '';
  });

  it('stamps touch-action: none (+ callout / selection off) when the computed value is the default', () => {
    restore = stubComputedTouchAction('auto');
    installCanvasGestureOwnership(canvas, events);
    expect(canvas.style.touchAction).toBe('none');
    expect(canvas.style.getPropertyValue('-webkit-touch-callout')).toBe('none');
    expect(canvas.style.userSelect).toBe('none');
  });

  it('treats an unmodelled (empty) computed value as the default too', () => {
    restore = stubComputedTouchAction('');
    installCanvasGestureOwnership(canvas, events);
    expect(canvas.style.touchAction).toBe('none');
  });

  it("respects an embedder's explicit touch-action", () => {
    restore = stubComputedTouchAction('pan-y');
    installCanvasGestureOwnership(canvas, events);
    expect(canvas.style.touchAction).toBe('');
    expect(canvas.style.getPropertyValue('-webkit-touch-callout')).toBe('none');
    expect(canvas.style.userSelect).toBe('none');
    expect(canvas.style.getPropertyValue('-webkit-user-select')).toBe('none');
  });

  it('registers Safari gesture cancellers only on devices with touch points', () => {
    restore = stubComputedTouchAction('auto');
    installCanvasGestureOwnership(canvas, events);
    const evNoTouch = new Event('gesturestart', { cancelable: true });
    canvas.dispatchEvent(evNoTouch);
    expect(evNoTouch.defaultPrevented).toBe(false);
    events.dispose();

    profile.touchPoints = 5;
    const touchEvents = new EventGroup();
    installCanvasGestureOwnership(canvas, touchEvents);
    for (const type of ['gesturestart', 'gesturechange', 'gestureend']) {
      const ev = new Event(type, { cancelable: true });
      canvas.dispatchEvent(ev);
      expect(ev.defaultPrevented, type).toBe(true);
    }

    // Disposing the group removes them.
    touchEvents.dispose();
    const after = new Event('gesturechange', { cancelable: true });
    canvas.dispatchEvent(after);
    expect(after.defaultPrevented).toBe(false);
  });

  it('is idempotent', () => {
    restore = stubComputedTouchAction('auto');
    profile.touchPoints = 5;
    installCanvasGestureOwnership(canvas, events);
    expect(events.size).toBe(4);
    installCanvasGestureOwnership(canvas, events);
    expect(events.size).toBe(4);
    expect(canvas.style.touchAction).toBe('none');
    const ev = new Event('gesturestart', { cancelable: true });
    canvas.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });

  it('restores every inline declaration and permits reinstallation after dispose', () => {
    restore = stubComputedTouchAction('auto');
    canvas.style.setProperty('-webkit-touch-callout', 'default');
    canvas.style.userSelect = 'text';
    canvas.style.setProperty('-webkit-user-select', 'text');

    installCanvasGestureOwnership(canvas, events);
    expect(canvas.style.touchAction).toBe('none');
    expect(canvas.style.userSelect).toBe('none');

    events.dispose();
    expect(canvas.style.touchAction).toBe('');
    expect(canvas.style.getPropertyValue('-webkit-touch-callout')).toBe('default');
    expect(canvas.style.userSelect).toBe('text');
    expect(canvas.style.getPropertyValue('-webkit-user-select')).toBe('text');

    const nextEvents = new EventGroup();
    installCanvasGestureOwnership(canvas, nextEvents);
    expect(canvas.style.touchAction).toBe('none');
    nextEvents.dispose();
  });
});
