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

import { afterEach } from 'vitest';
import { installAllMocks } from './mocks';
import { __setMinInstanceCapacityForTesting } from '../rendering/gpu-buffer-pool';

// Production scenes (real-WebGPU dispatch in particular) need a floor
// on instance-capacity to sidestep a WebGPU zero-buffer rebinding bug
// on grow — see the comment in `rendering/gpu-buffer-pool.ts` near
// `DEFAULT_MIN_INSTANCE_CAPACITY` (256). That floor is the right
// value for production but is far too coarse for unit tests that
// legitimately want to exercise the grow path at small counts
// (capacity 50 → 500 etc.).
//
// Lower the floor to zero for unit tests so capacity tracks the
// requested count tightly. The grow-path semantics are unchanged; the
// floor is purely an additional `Math.max` clamp. Tests that
// specifically need to assert production-floor behaviour can call
// `__setMinInstanceCapacityForTesting(null)` in a try/finally.
__setMinInstanceCapacityForTesting(0);

// Vitest constructs mocks via `Reflect.construct(implementation, ...)`, which
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

// Node >= 25 ships an experimental global `localStorage`/`sessionStorage`
// accessor that evaluates to `undefined` unless Node is started with
// --localstorage-file. Vitest's jsdom environment copies window properties
// onto the worker global via `populateGlobal`, but SKIPS any key already
// present on the global unless it's in Vitest's hardcoded KEYS list —
// which does not include the storages. On Node >= 25 the key pre-exists
// (Node's getter), so jsdom's Storage never lands on the global and bare
// `localStorage.clear()` in tests throws "Cannot read properties of
// undefined (reading 'clear')". On Node <= 24 the key doesn't pre-exist
// and everything works — which is why CI (Node 22) is green while newer
// local Node versions fail.
//
// Fix: rebind the globals to the REAL jsdom Storage, reachable via the
// JSDOM instance that Vitest's jsdom env exposes as `globalThis.jsdom`
// (its `dom.window` is the original window object, not the augmented
// global, so its own `localStorage` getter is intact). Node's file-backed
// storage is NOT a substitute — it would persist across runs and be
// shared by every worker.
{
  const dom = (globalThis as { jsdom?: { window?: Window } }).jsdom;
  const realWindow = dom?.window;
  if (realWindow?.localStorage && globalThis.localStorage === undefined) {
    Object.defineProperty(globalThis, 'localStorage', {
      value: realWindow.localStorage,
      configurable: true,
    });
  }
  if (realWindow?.sessionStorage && globalThis.sessionStorage === undefined) {
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: realWindow.sessionStorage,
      configurable: true,
    });
  }
}

// Pin `navigator.platform` so the two test environments agree.
//
// jsdom reports `''`; Node's own global `navigator` reports the REAL host
// ('MacIntel' on a Mac, 'Linux x86_64' in CI). `utils/platform.ts::isMacPlatform`
// reads it, and ~130 tests reach that module transitively — so without this pin,
// moving a file between `node` and `jsdom` (or moving CI between a Mac and a
// Linux runner) silently flips Cmd-vs-Ctrl modifier behaviour underneath tests
// that never mention the platform.
//
// `''` is jsdom's value, chosen so no existing expectation moves: today every
// test already runs as "not a Mac". A test that wants the Mac branch should
// override this explicitly rather than inherit it from the host.
Object.defineProperty(globalThis.navigator, 'platform', {
  value: '',
  configurable: true,
});

// Install all mocks
installAllMocks();

// Drain any animation frames left pending by a test after each test. The
// rAF mock (browser-apis.mock.ts) backs each frame with a real setTimeout;
// frames that are never flushed or cancelled accumulate as live timers and,
// under whole-suite execution, contribute to worker-pool exhaustion and the
// `Timeout waiting for worker to respond` failures. The cleanup helper is
// installed by installAnimationFrameMock(); guard in case a test swapped the
// rAF implementation.
afterEach(() => {
  (globalThis as { __clearAllAnimationFrames?: () => void }).__clearAllAnimationFrames?.();
});

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
    (Window & { _virtualConsole?: VirtualConsoleLike }) | null;
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
