/**
 * Unit tests for the pure DOM-focus helpers in input/handlers/focus-utils.
 *
 * These tests run in jsdom against real DOM elements — no mocks. The
 * helpers take an `Element | null` so the test can construct elements
 * directly instead of calling `.focus()` and waiting for activeElement
 * to update (jsdom is sometimes flaky about that).
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { isFocusOnSceneCanvas, isTypingInInput } from '../../../../../input/input-handler/commands/focus-utils';

describe('isTypingInInput', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('returns false for null', () => {
    expect(isTypingInInput(null)).toBe(false);
  });

  it('returns false for body and div', () => {
    const div = document.createElement('div');
    document.body.appendChild(div);
    expect(isTypingInInput(document.body)).toBe(false);
    expect(isTypingInInput(div)).toBe(false);
  });

  it('returns true for a plain text input', () => {
    const input = document.createElement('input');
    input.type = 'text';
    expect(isTypingInInput(input)).toBe(true);
  });

  it('returns true for an input with no type attribute (defaults to text)', () => {
    const input = document.createElement('input');
    expect(isTypingInInput(input)).toBe(true);
  });

  it('returns true for password / number / search inputs', () => {
    for (const type of ['password', 'number', 'search', 'email', 'url']) {
      const input = document.createElement('input');
      input.type = type;
      expect(isTypingInInput(input)).toBe(true);
    }
  });

  it('returns false for non-typing input types: range / checkbox / radio', () => {
    for (const type of ['range', 'checkbox', 'radio']) {
      const input = document.createElement('input');
      input.type = type;
      expect(isTypingInInput(input)).toBe(false);
    }
  });

  it('returns true for textarea and select', () => {
    expect(isTypingInInput(document.createElement('textarea'))).toBe(true);
    expect(isTypingInInput(document.createElement('select'))).toBe(true);
  });

  it('returns true for a contenteditable=true element', () => {
    const div = document.createElement('div');
    div.setAttribute('contenteditable', 'true');
    expect(isTypingInInput(div)).toBe(true);
  });

  it('returns false for contenteditable=false', () => {
    const div = document.createElement('div');
    div.setAttribute('contenteditable', 'false');
    expect(isTypingInInput(div)).toBe(false);
  });

  it('handles uppercase tag names (e.g. when DOM serialiser normalises)', () => {
    const input = document.createElement('INPUT');
    input.setAttribute('type', 'text');
    expect(isTypingInInput(input)).toBe(true);
  });
});

describe('isFocusOnSceneCanvas', () => {
  let canvas: HTMLCanvasElement;

  beforeEach(() => {
    document.body.innerHTML = '';
    canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
  });

  it('returns false for null active element', () => {
    expect(isFocusOnSceneCanvas(null, canvas)).toBe(false);
  });

  it('returns true when active element is document.body', () => {
    expect(isFocusOnSceneCanvas(document.body, canvas)).toBe(true);
  });

  it('returns true when active element is the canvas', () => {
    expect(isFocusOnSceneCanvas(canvas, canvas)).toBe(true);
  });

  it('returns false when active element is some other DOM element', () => {
    const div = document.createElement('div');
    document.body.appendChild(div);
    expect(isFocusOnSceneCanvas(div, canvas)).toBe(false);
  });

  it('returns false when canvas is null and focus is not on body', () => {
    const div = document.createElement('div');
    expect(isFocusOnSceneCanvas(div, null)).toBe(false);
  });

  it('returns true when canvas is null but focus is on body (e.g. before canvas attached)', () => {
    expect(isFocusOnSceneCanvas(document.body, null)).toBe(true);
  });
});
