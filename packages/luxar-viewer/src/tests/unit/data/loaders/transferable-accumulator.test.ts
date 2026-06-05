import { describe, it, expect, beforeEach } from 'vitest';
import {
  TransferableAccumulator,
  createPointsAccumulator,
  createLinesAccumulator,
  createGSplatsAccumulator,
  type PointsBuffers,
  type TransferableBuffers,
  type BufferShape,
} from '../../../../data/loaders';

describe('TransferableAccumulator', () => {
  describe('basic operations', () => {
    let accumulator: TransferableAccumulator<PointsBuffers>;

    beforeEach(() => {
      accumulator = createPointsAccumulator(1024);
    });

    it('should create with initial capacity', () => {
      expect(accumulator.getCapacity()).toBe(1024);
      expect(accumulator.isBuffersDetached()).toBe(false);
    });

    it('should provide access to buffers', () => {
      const buffers = accumulator.getBuffers();

      expect(buffers).not.toBeNull();
      expect(buffers!.positions).toBeInstanceOf(Float32Array);
      // capacity (1024) * elementsPerItem (3 for nD positions) — pins the
      // full allocation, not just the type.
      expect(buffers!.positions.length).toBe(1024 * 3);
      // Optional buffers are NOT allocated until enableBuffer() is called.
      expect(buffers!.colors).toBeUndefined();
      expect(buffers!.radii).toBeUndefined();
    });

    it('should track statistics', () => {
      const stats = accumulator.getStats();

      expect(stats.allocations).toBe(1);
      expect(stats.bytesAllocated).toBeGreaterThan(0);
      expect(stats.reuseCount).toBe(0);
    });
  });

  describe('ensureCapacity', () => {
    it('should grow when capacity exceeded', () => {
      const accumulator = createPointsAccumulator(100);

      const didGrow = accumulator.ensureCapacity(200);

      expect(didGrow).toBe(true);
      expect(accumulator.getCapacity()).toBeGreaterThanOrEqual(200);
    });

    it('should not grow when capacity sufficient', () => {
      const accumulator = createPointsAccumulator(100);

      const didGrow = accumulator.ensureCapacity(50);

      expect(didGrow).toBe(false);
      expect(accumulator.getCapacity()).toBe(100);
    });

    it('should preserve existing data when growing', () => {
      const accumulator = createPointsAccumulator(10);
      const buffers = accumulator.getBuffers()!;

      // Write some test data
      buffers.positions[0] = 1.0;
      buffers.positions[1] = 2.0;
      buffers.positions[2] = 3.0;

      // Grow
      accumulator.ensureCapacity(100);

      // Data should be preserved
      const newBuffers = accumulator.getBuffers()!;
      expect(newBuffers.positions[0]).toBe(1.0);
      expect(newBuffers.positions[1]).toBe(2.0);
      expect(newBuffers.positions[2]).toBe(3.0);
    });

    it('should throw when growing while detached', () => {
      const accumulator = createPointsAccumulator(100);
      accumulator.detach();

      expect(() => accumulator.ensureCapacity(200)).toThrow(/detached/);
    });
  });

  describe('detach/adopt cycle', () => {
    it('should detach buffers', () => {
      const accumulator = createPointsAccumulator(100);

      const detached = accumulator.detach();

      expect(detached.positions).toBeInstanceOf(Float32Array);
      expect(accumulator.isBuffersDetached()).toBe(true);
      expect(accumulator.getBuffers()).toBeNull();
    });

    it('should adopt buffers', () => {
      const accumulator = createPointsAccumulator(100);
      const detached = accumulator.detach();

      // Modify buffers (simulating worker filling them)
      detached.positions[0] = 42.0;

      accumulator.adopt(detached);

      expect(accumulator.isBuffersDetached()).toBe(false);
      expect(accumulator.getBuffers()!.positions[0]).toBe(42.0);
    });

    it('should track reuse count on adopt', () => {
      const accumulator = createPointsAccumulator(100);

      // Multiple detach/adopt cycles
      for (let i = 0; i < 3; i++) {
        const detached = accumulator.detach();
        accumulator.adopt(detached);
      }

      expect(accumulator.getStats().reuseCount).toBe(3);
    });

    it('should throw when detaching already detached', () => {
      const accumulator = createPointsAccumulator(100);
      accumulator.detach();

      expect(() => accumulator.detach()).toThrow(/already detached/);
    });

    it('should throw when adopting incompatible buffers', () => {
      const accumulator = createPointsAccumulator(100);
      accumulator.detach();

      // Try to adopt buffers missing required 'positions'
      const badBuffers = {} as PointsBuffers;

      expect(() => accumulator.adopt(badBuffers)).toThrow(/missing required/);
    });
  });

  describe('getTransferables', () => {
    it('should return ArrayBuffer list for transfer', () => {
      const accumulator = createPointsAccumulator(100);
      const buffers = accumulator.getBuffers()!;

      const transferables = accumulator.getTransferables(buffers);

      expect(transferables.length).toBeGreaterThan(0);
      expect(transferables[0]).toBeInstanceOf(ArrayBuffer);
    });

    it('should include all allocated buffers', () => {
      const accumulator = createPointsAccumulator(100);
      accumulator.enableBuffer('colors');
      accumulator.enableBuffer('radii');
      const buffers = accumulator.getBuffers()!;

      const transferables = accumulator.getTransferables(buffers);

      // positions + colors + radii = 3 buffers
      expect(transferables.length).toBe(3);
    });
  });

  describe('enableBuffer', () => {
    it('should enable optional buffer', () => {
      const accumulator = createPointsAccumulator(100);
      const buffersBefore = accumulator.getBuffers()!;

      expect(buffersBefore.colors).toBeUndefined();

      accumulator.enableBuffer('colors');

      const buffersAfter = accumulator.getBuffers()!;
      expect(buffersAfter.colors).toBeInstanceOf(Float32Array);
      expect(buffersAfter.colors!.length).toBe(100 * 3);
    });

    it('should throw for unknown buffer name', () => {
      const accumulator = createPointsAccumulator(100);

      expect(() => accumulator.enableBuffer('unknown')).toThrow(/Unknown buffer/);
    });

    it('should be idempotent', () => {
      const accumulator = createPointsAccumulator(100);

      accumulator.enableBuffer('colors');
      const buffers1 = accumulator.getBuffers()!.colors;

      accumulator.enableBuffer('colors');
      const buffers2 = accumulator.getBuffers()!.colors;

      expect(buffers1).toBe(buffers2);
    });
  });

  describe('dispose', () => {
    it('should release all buffers', () => {
      const accumulator = createPointsAccumulator(100);

      accumulator.dispose();

      expect(accumulator.getBuffers()).toBeNull();
      expect(accumulator.getCapacity()).toBe(0);
    });
  });
});

describe('createPointsAccumulator', () => {
  it('should create with Float32 colors by default', () => {
    const accumulator = createPointsAccumulator(100);
    accumulator.enableBuffer('colors');

    const buffers = accumulator.getBuffers()!;
    expect(buffers.colors).toBeInstanceOf(Float32Array);
  });

  it('should create with Uint8 colors when specified', () => {
    const accumulator = createPointsAccumulator(100, 'uint8');
    accumulator.enableBuffer('colors');

    const buffers = accumulator.getBuffers()!;
    expect(buffers.colors).toBeInstanceOf(Uint8Array);
  });

  it('should create with Uint16 colors when specified', () => {
    const accumulator = createPointsAccumulator(100, 'uint16');
    accumulator.enableBuffer('colors');

    const buffers = accumulator.getBuffers()!;
    expect(buffers.colors).toBeInstanceOf(Uint16Array);
  });
});

describe('createLinesAccumulator', () => {
  it('should create with correct buffer structure', () => {
    const accumulator = createLinesAccumulator(100);
    const buffers = accumulator.getBuffers()!;

    expect(buffers.startPositions).toBeInstanceOf(Float32Array);
    expect(buffers.startPositions.length).toBe(100 * 3);

    expect(buffers.endPositions).toBeInstanceOf(Float32Array);
    expect(buffers.endPositions.length).toBe(100 * 3);
  });

  it('should support optional widths buffer', () => {
    const accumulator = createLinesAccumulator(100);
    accumulator.enableBuffer('widths');

    const buffers = accumulator.getBuffers()!;
    expect(buffers.widths).toBeInstanceOf(Float32Array);
    expect(buffers.widths!.length).toBe(100);
  });
});

describe('createGSplatsAccumulator', () => {
  it('should create with correct buffer structure', () => {
    const accumulator = createGSplatsAccumulator(100, 6); // 3D: k=6
    const buffers = accumulator.getBuffers()!;

    expect(buffers.positions).toBeInstanceOf(Float32Array);
    expect(buffers.positions.length).toBe(100 * 3);

    expect(buffers.amplitudes).toBeInstanceOf(Float32Array);
    expect(buffers.amplitudes.length).toBe(100);

    expect(buffers.choleskyFactors).toBeInstanceOf(Float32Array);
    expect(buffers.choleskyFactors.length).toBe(100 * 6);
  });

  it('should handle different Cholesky sizes', () => {
    // 4D: k = 4 * 5 / 2 = 10
    const accumulator = createGSplatsAccumulator(100, 10);
    const buffers = accumulator.getBuffers()!;

    expect(buffers.choleskyFactors.length).toBe(100 * 10);
  });
});

describe('Generic TransferableAccumulator', () => {
  it('should work with custom buffer shapes', () => {
    interface CustomBuffers extends TransferableBuffers {
      values: Float32Array;
      indices: Uint16Array;
    }

    const shapes: BufferShape[] = [
      { name: 'values', elementsPerItem: 4, type: 'float32' },
      { name: 'indices', elementsPerItem: 1, type: 'uint16' },
    ];

    const accumulator = new TransferableAccumulator<CustomBuffers>(shapes, 50);
    const buffers = accumulator.getBuffers()!;

    expect(buffers.values).toBeInstanceOf(Float32Array);
    expect(buffers.values.length).toBe(50 * 4);

    expect(buffers.indices).toBeInstanceOf(Uint16Array);
    expect(buffers.indices.length).toBe(50);
  });

  it('should calculate peak memory correctly', () => {
    const shapes: BufferShape[] = [
      { name: 'floats', elementsPerItem: 4, type: 'float32' }, // 4 * 4 = 16 bytes per item
      { name: 'bytes', elementsPerItem: 2, type: 'uint8' }, // 2 bytes per item
    ];

    interface TestBuffers extends TransferableBuffers {
      floats: Float32Array;
      bytes: Uint8Array;
    }

    const accumulator = new TransferableAccumulator<TestBuffers>(shapes, 100);
    const stats = accumulator.getStats();

    // 100 items * (16 + 2) = 1800 bytes
    expect(stats.bytesAllocated).toBe(1800);
    expect(stats.peakMemoryBytes).toBe(1800);
  });
});
