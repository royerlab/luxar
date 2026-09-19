import { expect, it, vi } from 'vitest';

it('keeps performance time frozen under fake timers', () => {
  // Points, lines, and gsplats progressive-loader depth tests rely on this
  // Vitest default so scheduler stalls cannot trigger the production time brake.
  vi.useFakeTimers();
  try {
    const startedAt = performance.now();
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 20);
    expect(performance.now()).toBe(startedAt);
  } finally {
    vi.useRealTimers();
  }
});
