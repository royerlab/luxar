/**
 * Worker-config-defaults: the two Node-runnable facts about worker wiring.
 *
 * Moved from `src/tests/unit/integration/worker-integration.test.ts` per
 * integration.md C1: the file's only assertions are a config-default
 * check and a bare-import smoke, neither of which is an "integration"
 * test. Its proper home is alongside other config-default tests under
 * `src/tests/unit/config/`.
 *
 * Real worker-API integration coverage lives in:
 *   - `src/tests/unit/workers/worker-pool/*.test.ts` (per-helper unit tests)
 *   - `src/tests/e2e/worker-wasm-integration.spec.ts` (Playwright, real browser)
 */

import { describe, it, expect } from 'vitest';
import { config } from '../../../config';

describe('Worker config defaults — Node-side wiring smoke', () => {
  it('worker config defaults route through useWebWorkers=true (production default)', () => {
    // This is the only meaningful contract we can pin in Node: the
    // production config defaults to using web workers. The default came from
    // src/config/sections/data-loading/performance/data.ts; if a future
    // refactor accidentally flips the default, this test catches it. (The
    // *behavior* of workers vs main-thread fallback is verified in the
    // Playwright suite, which runs in a real browser.)
    expect(config.dataLoading.performance.useWebWorkers).toBe(true);
  });

  it('worker pool module imports without throwing under Node (no top-level Worker access)', async () => {
    // Contract: importing the worker-pool module must not crash in
    // environments without a Worker constructor. The pool defers Worker
    // instantiation until `initialize()` is called, so a bare import is
    // safe. If a future refactor moves `new Worker(...)` to module scope,
    // this test catches it.
    await expect(import('../../../workers/worker-pool')).resolves.toBeDefined();
  });
});
