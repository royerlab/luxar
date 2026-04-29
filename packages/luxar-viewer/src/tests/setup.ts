/**
 * Test setup file for Vitest
 *
 * Configures test environment with all necessary mocks:
 * - WebGL rendering context
 * - Browser APIs (matchMedia, ResizeObserver, etc.)
 * - OPFS (Origin Private File System)
 *
 * All mocks are now organized in separate files under ./mocks/
 */

import { installAllMocks } from './mocks';

// Vitest 4 constructs mocks via `Reflect.construct(implementation, ...)`, which
// throws `TypeError: ... is not a constructor` when the implementation is an
// arrow function (arrows have no [[Construct]] internal method). Many existing
// tests — including `vi.mock(modulePath)` auto-mocks and `vi.fn(arrow)` /
// `vi.fn().mockImplementation(arrow)` patterns — pass arrows that are then
// invoked with `new`.
//
// Rather than rewriting every test, intercept Reflect.construct: when called
// on an arrow, wrap it in a regular `function` (which is constructable and
// returns whatever the arrow returns when its return value is an object).
type AnyFn = (...args: unknown[]) => unknown;
const _origConstruct = Reflect.construct;
Reflect.construct = function patchedConstruct<T extends object>(
  target: (...args: unknown[]) => unknown,
  argumentsList: ArrayLike<unknown>,
  newTarget?: (...args: unknown[]) => unknown,
): T {
  if (
    typeof target === 'function' &&
    (target as { prototype?: unknown }).prototype === undefined
  ) {
    const arrow = target as AnyFn;
    const wrapped = function wrapped(this: unknown, ...args: unknown[]) {
      return arrow.apply(this, args);
    };
    return _origConstruct(
      wrapped as unknown as new (...a: unknown[]) => T,
      argumentsList,
      (newTarget ?? wrapped) as unknown as new (...a: unknown[]) => T,
    );
  }
  return _origConstruct(
    target as unknown as new (...a: unknown[]) => T,
    argumentsList,
    newTarget as unknown as new (...a: unknown[]) => T,
  );
} as typeof Reflect.construct;

// Install all mocks
installAllMocks();
