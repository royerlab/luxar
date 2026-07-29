/**
 * Tests for `utils/format-error` — the shared "unknown thrown value → readable"
 * helpers behind the debug console's Error rendering.
 */

import { describe, it, expect } from 'vitest';
import { getErrorMessage, formatErrorForDisplay, getErrorStack } from '../../../utils/format-error';

describe('getErrorMessage', () => {
  it('reads the message off an Error', () => {
    expect(getErrorMessage(new Error('boom'))).toBe('boom');
  });

  it('reads the message off an Error subclass', () => {
    class Sub extends Error {}
    expect(getErrorMessage(new Sub('sub boom'))).toBe('sub boom');
  });

  it('stringifies non-Error values', () => {
    expect(getErrorMessage('plain string')).toBe('plain string');
    expect(getErrorMessage(42)).toBe('42');
    expect(getErrorMessage(undefined)).toBe('undefined');
    expect(getErrorMessage(null)).toBe('null');
  });

  it('never throws on a null-prototype object', () => {
    // String() raises "Cannot convert object to primitive value" here. This
    // helper runs inside catch blocks, so it must not become the thing that
    // breaks the error handler.
    expect(() => getErrorMessage(Object.create(null))).not.toThrow();
    expect(getErrorMessage(Object.create(null))).toBe('[unprintable error]');
  });
});

describe('formatErrorForDisplay', () => {
  it('renders name: message', () => {
    expect(formatErrorForDisplay(new Error('boom'))).toBe('Error: boom');
  });

  it('uses a subclass name', () => {
    class Sub extends Error {
      constructor(m: string) {
        super(m);
        this.name = 'Sub';
      }
    }
    expect(formatErrorForDisplay(new Sub('boom'))).toBe('Sub: boom');
  });

  it('omits the colon when there is no message', () => {
    expect(formatErrorForDisplay(new Error())).toBe('Error');
  });

  it('handles a DOMException, which is what OPFS throws', () => {
    // DOMException satisfies `instanceof Error` per WebIDL, and stringifies to
    // '{}' — the exact shape behind the reported `metadata save failed {}`.
    const e = new DOMException('A requested file could not be found', 'NotFoundError');
    expect(formatErrorForDisplay(e)).toBe('NotFoundError: A requested file could not be found');
  });
});

describe('getErrorStack', () => {
  it('returns a real Error stack', () => {
    const stack = getErrorStack(new Error('boom'));
    expect(typeof stack).toBe('string');
    expect(stack).toContain('Error');
  });

  it('accepts a duck-typed stack carrier', () => {
    // Some Firefox DOMExceptions carry a stack without reporting as an Error.
    expect(getErrorStack({ stack: 'at somewhere' })).toBe('at somewhere');
  });

  it('returns undefined when there is no stack to be had', () => {
    expect(getErrorStack('just a string')).toBeUndefined();
    expect(getErrorStack({ notAStack: 1 })).toBeUndefined();
    expect(getErrorStack(null)).toBeUndefined();
    // A non-string stack is not a stack.
    expect(getErrorStack({ stack: 42 })).toBeUndefined();
  });
});
