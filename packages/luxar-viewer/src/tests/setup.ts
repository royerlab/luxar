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
  newTarget?: (...args: unknown[]) => unknown
): T {
  if (typeof target === 'function' && (target as { prototype?: unknown }).prototype === undefined) {
    const arrow = target as AnyFn;
    const wrapped = function wrapped(this: unknown, ...args: unknown[]) {
      return arrow.apply(this, args);
    };
    return _origConstruct(
      wrapped as unknown as new (...a: unknown[]) => T,
      argumentsList,
      (newTarget ?? wrapped) as unknown as new (...a: unknown[]) => T
    );
  }
  return _origConstruct(
    target as unknown as new (...a: unknown[]) => T,
    argumentsList,
    newTarget as unknown as new (...a: unknown[]) => T
  );
} as typeof Reflect.construct;

// Install all mocks
installAllMocks();

// Silence the jsdom "Not implemented: navigation (except hash changes)"
// errors emitted whenever production code calls `<a>.click()` to trigger
// a download (recording-panel screenshot/video/EXR export uses this
// pattern). The behaviour is correct in real browsers; jsdom can't
// navigate, so it emits a jsdomError which the default virtualConsole
// pipes to console.error.
//
// Intercept at the virtualConsole `jsdomError` listener level — jsdom's
// own emit path — so the noise is filtered before the runner ever sees
// it. Other jsdomError messages (real DOM violations) still propagate.
{
  type VirtualConsoleLike = {
    on(event: 'jsdomError', cb: (e: Error) => void): void;
    removeAllListeners(event: 'jsdomError'): void;
  };
  const win = (typeof window !== 'undefined' ? window : null) as
    | (Window & { _virtualConsole?: VirtualConsoleLike })
    | null;
  const vc = win?._virtualConsole;
  if (vc) {
    vc.removeAllListeners('jsdomError');
    vc.on('jsdomError', (err: Error) => {
      if (err.message.startsWith('Not implemented: navigation')) return;
      // Anything else: forward verbatim so real DOM violations aren't lost.
      console.error(err);
    });
  }
}
