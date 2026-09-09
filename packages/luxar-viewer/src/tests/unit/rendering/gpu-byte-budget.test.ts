import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  configureGpuByteBudget,
  getGpuByteBudget,
  initializeGpuByteBudget,
  reduceGpuByteBudgetForContextLoss,
} from '../../../rendering/gpu-byte-budget';
import { log, Modules } from '../../../utils/log';

const profile = { deviceClass: 'laptop' as 'mobile' | 'laptop' | 'desktop' };
vi.mock('../../../utils/input-capabilities', () => ({
  getInputProfile: () => profile,
}));

/** Temporarily set navigator.deviceMemory (GB) for a test. */
function withDeviceMemory(gb: number | undefined, fn: () => void): void {
  const had = Object.prototype.hasOwnProperty.call(navigator, 'deviceMemory');
  const prev = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  Object.defineProperty(navigator, 'deviceMemory', { value: gb, configurable: true });
  try {
    fn();
  } finally {
    if (had) Object.defineProperty(navigator, 'deviceMemory', { value: prev, configurable: true });
    else delete (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  }
}

/** Temporarily set performance.memory.jsHeapSizeLimit for a test. */
function withHeapLimit(bytes: number | undefined, fn: () => void): void {
  const descriptor = Object.getOwnPropertyDescriptor(performance, 'memory');
  Object.defineProperty(performance, 'memory', {
    configurable: true,
    value: bytes === undefined ? undefined : { jsHeapSizeLimit: bytes },
  });
  try {
    fn();
  } finally {
    if (descriptor) Object.defineProperty(performance, 'memory', descriptor);
    else delete (performance as Performance & { memory?: unknown }).memory;
  }
}

const MB = 1_000_000;
const MiB = 1024 * 1024;

afterEach(() => {
  // Reset to a known state so tests don't leak the singleton budget.
  configureGpuByteBudget(512 * MB);
});

describe('gpu-byte-budget', () => {
  it('initializes a fresh embed budget from the mobile device class', async () => {
    vi.resetModules();
    profile.deviceClass = 'mobile';
    try {
      const freshBudget = await import('../../../rendering/gpu-byte-budget');
      withDeviceMemory(undefined, () => {
        withHeapLimit(undefined, () => {
          freshBudget.initializeGpuByteBudget(null);
          expect(freshBudget.getGpuByteBudget()).toBe(Math.floor((384 * MiB) / 3));
        });
      });
    } finally {
      profile.deviceClass = 'laptop';
    }
  });

  it('resolves an omitted embed override from config', async () => {
    vi.resetModules();
    const { config: freshConfig } = await import('../../../config');
    const previous = freshConfig.dataLoading.performance.gpuPoolMaxBytes;
    try {
      freshConfig.dataLoading.performance.gpuPoolMaxBytes = 321 * MB;
      const freshBudget = await import('../../../rendering/gpu-byte-budget');
      freshBudget.initializeGpuByteBudget();
      expect(freshBudget.getGpuByteBudget()).toBe(321 * MB);
    } finally {
      freshConfig.dataLoading.performance.gpuPoolMaxBytes = previous;
    }
  });

  it('honors an explicit override and skips the heuristic', () => {
    withDeviceMemory(8, () => {
      configureGpuByteBudget(1536 * MB);
      expect(getGpuByteBudget()).toBe(1536 * MB);
    });
  });

  it('keeps the first startup configuration when a shared embed path configures again', () => {
    withDeviceMemory(1, () => {
      configureGpuByteBudget(777 * MB);
      initializeGpuByteBudget();
      expect(getGpuByteBudget()).toBe(777 * MB);
    });
  });

  it.each([
    [null, 'auto-sizing'],
    [0, 'disabling eviction'],
    [123 * MB, '123 MB'],
  ] as const)('warns when a later entry point requests explicit %s', (overrideBytes, requested) => {
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      configureGpuByteBudget(777 * MB);
      initializeGpuByteBudget(overrideBytes);
      expect(getGpuByteBudget()).toBe(777 * MB);
      expect(warning).toHaveBeenCalledWith(
        Modules.PERFORMANCE,
        `GPU byte budget is already configured; ignoring later request for ${requested}`
      );
    } finally {
      warning.mockRestore();
    }
  });

  it('auto-sizes from deviceMemory at 25%, capped at 2GB with NO floor', () => {
    withDeviceMemory(8, () => {
      configureGpuByteBudget(); // 8 GB × 0.25 = 2 GB (== ceiling)
      expect(getGpuByteBudget()).toBe(2_000 * MB);
    });
    withDeviceMemory(4, () => {
      configureGpuByteBudget(); // 4 GB × 0.25 = 1 GB
      expect(getGpuByteBudget()).toBe(1_000 * MB);
    });
    withDeviceMemory(32, () => {
      configureGpuByteBudget(); // clamps to 2 GB ceiling
      expect(getGpuByteBudget()).toBe(2_000 * MB);
    });
    withDeviceMemory(1, () => {
      // A small device gets a SMALL budget — 1 GB × 0.25 = 250 MB, not clamped
      // up. The old 512 MB floor is deliberately gone: the pool's stranded
      // superseded pairs total a few hundred MB, so a floor above them meant
      // the LRU never fired and nothing was ever reclaimed on exactly the
      // devices that needed it. A budget that cannot bind is not a budget.
      configureGpuByteBudget();
      expect(getGpuByteBudget()).toBe(250 * MB);
    });
  });

  it('treats 0 as "disable" (verbatim, not auto-sized)', () => {
    withDeviceMemory(8, () => {
      configureGpuByteBudget(0); // explicit disable must NOT auto-size to 2 GB
      expect(getGpuByteBudget()).toBe(0);
    });
  });

  it('falls back to the no-signal budget when deviceMemory is unavailable', () => {
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    withDeviceMemory(undefined, () => {
      configureGpuByteBudget();
      expect(getGpuByteBudget()).toBe(512 * MB);
    });
    expect(info).toHaveBeenCalledWith(
      Modules.PERFORMANCE,
      'GPU byte budget: 512 MB (auto: no memory signal -> 512 MB fallback)'
    );
    info.mockRestore();
  });

  describe('cache-pool override drives the budget (#2426 pool retention)', () => {
    // `navigator.deviceMemory` is Chromium-only and spec-capped at 8 GB, so on
    // any large machine it pins the budget at the 2 GB ceiling and the pool's
    // eviction path is unreachable. The override is the ONLY way to reproduce
    // constrained-device behaviour — without it the fix is untestable on the
    // hardware that would verify it.
    it('is what makes eviction reachable: budget lands under a few hundred MB', () => {
      // The stranded superseded pairs measured on a 2M-splat 16-rung node are
      // ~285 MB. For the pool LRU to fire at all, the budget must land below
      // active + pooled. This pins that the 1024 MiB test point actually binds
      // — the whole acceptance criterion (evictions > 0) depends on it.
      withDeviceMemory(32, () => {
        configureGpuByteBudget(null, { cachePoolOverrideBytes: 1024 * MiB });
        expect(getGpuByteBudget()).toBe(Math.floor((1024 * MiB) / 3));
      });
    });

    it('an explicit gpuBudgetMB still wins over the override', () => {
      withDeviceMemory(8, () => {
        configureGpuByteBudget(1536 * MB, { cachePoolOverrideBytes: 256 * MB });
        expect(getGpuByteBudget()).toBe(1536 * MB);
      });
    });

    it('keeps large cache overrides proportional instead of applying the eager-loader cap', () => {
      withDeviceMemory(32, () => {
        configureGpuByteBudget(null, { cachePoolOverrideBytes: 4096 * MiB });
        expect(getGpuByteBudget()).toBe(Math.floor((4096 * MiB) / 3));
      });
    });

    it('uses the cache override when deviceMemory is unavailable', () => {
      withDeviceMemory(undefined, () => {
        configureGpuByteBudget(null, { cachePoolOverrideBytes: 2048 * MiB });
        expect(getGpuByteBudget()).toBe(Math.floor((2048 * MiB) / 3));
      });
    });

    it('does not shrink the budget merely because the heap is unmeasurable', () => {
      // Firefox and Safari expose no `performance.memory`. Their absence of a
      // heap measurement must not tighten an otherwise roomy device budget.
      withDeviceMemory(32, () => {
        configureGpuByteBudget(null, {});
        expect(getGpuByteBudget()).toBe(2_000 * MB);
      });
    });

    it('DOES bind a roomy Chromium device to its measured heap tier', () => {
      // This assertion has been the opposite twice, on the reasoning that a
      // coarse heap tier is not a GPU-memory signal. True of VRAM, wrong for
      // the failure this budget prevents: the crash is `RangeError: Array
      // buffer allocation failed` — JS heap exhaustion — and a stranded pooled
      // pair holds a CPU-side ArrayBuffer, so heap is what must bound the
      // pool. `deviceMemory` reports SYSTEM RAM, so a 32 GB box whose tab heap
      // caps near 4 GB reads as roomy, takes the 2 GB ceiling, and evicts
      // nothing right up until the tab dies.
      //
      // Settled by measurement, not argument (host-demos, `jsHeapSizeLimit`
      // spoofed with real used/total passing through): WITHOUT the heap term
      // the auto budget stayed at 2000 MB even at a spoofed 256 MB heap and
      // reclaimed nothing — zero movement. WITH it the budget tracks the spoof
      // at 43 / 86 / 172 MB for 256 / 512 / 1024 MiB, and multi6 returns to
      // evict=5 on the real heap.
      withDeviceMemory(32, () => {
        withHeapLimit(4 * 1024 * 1024 * 1024, () => {
          configureGpuByteBudget();
          expect(getGpuByteBudget()).toBeLessThan(2_000 * MB);
          expect(getGpuByteBudget()).toBeGreaterThan(0);
        });
      });
    });

    it('tracks the heap DOWNWARD, so a smaller heap binds harder', () => {
      // The property the spoof measured, pinned here so it cannot regress
      // silently: the budget must MOVE with the heap, not merely be below the
      // ceiling once. A constant that happens to sit under 2 GB would pass the
      // assertion above while being just as blind.
      let atFourGB = 0;
      let atOneGB = 0;
      withDeviceMemory(32, () => {
        withHeapLimit(4 * 1024 * 1024 * 1024, () => {
          configureGpuByteBudget();
          atFourGB = getGpuByteBudget();
        });
        withHeapLimit(1024 * 1024 * 1024, () => {
          configureGpuByteBudget();
          atOneGB = getGpuByteBudget();
        });
      });
      expect(atOneGB).toBeLessThan(atFourGB);
    });

    it('logs the cache override in the MiB units supplied by the user', () => {
      const info = vi.spyOn(log, 'info').mockImplementation(() => {});
      withDeviceMemory(32, () => {
        configureGpuByteBudget(null, { cachePoolOverrideBytes: 1024 * MiB });
      });
      expect(info).toHaveBeenCalledWith(
        Modules.PERFORMANCE,
        'GPU byte budget: 358 MB (auto: deviceMemory=32 GB -> 2000 MB, cacheBudgetMB=1024 -> 358 MB; min=358 MB)'
      );
      info.mockRestore();
    });
  });

  it('halves the budget on context loss, down to a 256MB floor', () => {
    configureGpuByteBudget(2_000 * MB);
    expect(reduceGpuByteBudgetForContextLoss()).toBe(1_000 * MB);
    expect(reduceGpuByteBudgetForContextLoss()).toBe(500 * MB);
    expect(getGpuByteBudget()).toBe(500 * MB);
    // Keeps halving but never below the 256 MB floor.
    reduceGpuByteBudgetForContextLoss(); // 250M → clamped to 256M floor
    expect(getGpuByteBudget()).toBe(256 * MB);
    reduceGpuByteBudgetForContextLoss(); // already at floor → unchanged
    expect(getGpuByteBudget()).toBe(256 * MB);
  });
});

describe('gpu-byte-budget — mobile device class', () => {
  afterEach(() => {
    profile.deviceClass = 'laptop';
  });

  it('a phone/tablet with NO other signal (WebKit) gets the mobile pool share, not the 512 MB fallback', () => {
    profile.deviceClass = 'mobile';
    withDeviceMemory(undefined, () => {
      withHeapLimit(undefined, () => {
        configureGpuByteBudget();
        expect(getGpuByteBudget()).toBe(Math.floor((384 * MiB) / 3));
      });
    });
  });

  it('the mobile candidate is a peer in the minimum: it binds under a generous deviceMemory', () => {
    profile.deviceClass = 'mobile';
    withDeviceMemory(8, () => {
      withHeapLimit(undefined, () => {
        configureGpuByteBudget(); // 8 GB × 0.25 = 2 GB would otherwise win
        expect(getGpuByteBudget()).toBe(Math.floor((384 * MiB) / 3));
      });
    });
  });

  it('a laptop/desktop contributes no device-class candidate (unchanged resolution)', () => {
    withDeviceMemory(undefined, () => {
      withHeapLimit(undefined, () => {
        configureGpuByteBudget();
        expect(getGpuByteBudget()).toBe(512 * MB);
      });
    });
  });
});
