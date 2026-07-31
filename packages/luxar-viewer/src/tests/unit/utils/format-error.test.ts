/**
 * Tests for `utils/format-error` — the shared "unknown thrown value → readable"
 * helpers behind the debug console's Error rendering.
 */

import { describe, it, expect } from 'vitest';
import {
  getErrorMessage,
  formatErrorForDisplay,
  getErrorStack,
  isErrorLike,
} from '../../../utils/format-error';

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

  it('never throws on an Error with a throwing message accessor', () => {
    const err = new Error('boom');
    Object.defineProperty(err, 'message', {
      get() {
        throw new Error('hostile message getter');
      },
    });
    expect(getErrorMessage(err)).toBe('[unprintable error]');
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

  it('never throws on a throwing name/message accessor', () => {
    const err = new Error('boom');
    Object.defineProperty(err, 'name', {
      get() {
        throw new Error('hostile name getter');
      },
    });
    expect(formatErrorForDisplay(err)).toBe('[unprintable error]');
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

  it('never throws on a throwing stack accessor', () => {
    // This runs inside the patched console.warn/error BEFORE the original
    // console call — a throw here would swallow the diagnostic being logged.
    const err = new Error('boom');
    Object.defineProperty(err, 'stack', {
      get() {
        throw new Error('hostile stack getter');
      },
    });
    expect(getErrorStack(err)).toBeUndefined();
  });

  it('never throws on a Proxy with a throwing has trap', () => {
    const trap = new Proxy(
      {},
      {
        has() {
          throw new Error('hostile has trap');
        },
      }
    );
    expect(getErrorStack(trap)).toBeUndefined();
  });
});

describe('isErrorLike', () => {
  it('matches a real Error and its subclasses', () => {
    expect(isErrorLike(new Error('boom'))).toBe(true);
    class Sub extends Error {}
    expect(isErrorLike(new Sub('boom'))).toBe(true);
    expect(isErrorLike(new DOMException('x', 'NotFoundError'))).toBe(true);
  });

  it('matches a cross-realm Error by its realm-proof [[Class]]', () => {
    // A same-realm `instanceof Error` misses this; the internal [[Class]] tag
    // does not. Emulate it: a plain object whose Symbol.toStringTag is 'Error'
    // reports as '[object Error]' without being an Error instance.
    const crossRealm = { [Symbol.toStringTag]: 'Error', name: 'TypeError', message: 'boom' };
    expect(crossRealm instanceof Error).toBe(false);
    expect(isErrorLike(crossRealm)).toBe(true);
  });

  it('matches a plain object carrying the full name+message+stack triple', () => {
    expect(isErrorLike({ name: 'Error', message: 'boom', stack: 'at somewhere' })).toBe(true);
  });

  it('does NOT match an ordinary data object', () => {
    // The false-positive that must not happen: a legitimate `{ name, message }`
    // payload should still render as JSON, not as an Error.
    expect(isErrorLike({ name: 'Widget', message: 'hello' })).toBe(false);
    expect(isErrorLike({ foo: 1 })).toBe(false);
    expect(isErrorLike('a string')).toBe(false);
    expect(isErrorLike(null)).toBe(false);
    expect(isErrorLike(undefined)).toBe(false);
    expect(isErrorLike(42)).toBe(false);
  });
});
