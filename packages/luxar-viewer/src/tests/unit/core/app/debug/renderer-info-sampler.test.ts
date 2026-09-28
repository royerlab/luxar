import { afterEach, describe, expect, it } from 'vitest';

import {
  getRendererInfoSnapshot,
  installRendererInfoSampler,
  uninstallRendererInfoSampler,
} from '../../../../../core/app/debug/renderer-info-sampler';
import { eventBus } from '../../../../../utils/cross-layer/event-bus';

function fakeRenderer() {
  return {
    info: {
      render: { frame: 7, calls: 12, triangles: 3_000_001, points: 0, lines: 4 },
      memory: { geometries: 5, textures: 9 },
      programs: [{}, {}, {}],
    },
  };
}

describe('renderer-info-sampler', () => {
  afterEach(() => uninstallRendererInfoSampler());

  it('is null before the first frame-end and copies renderer.info at frame-end', () => {
    const renderer = fakeRenderer();
    installRendererInfoSampler(() => renderer);
    expect(getRendererInfoSnapshot()).toBeNull();

    eventBus.emit('frame-end', {});
    const snap = getRendererInfoSnapshot();
    expect(snap).toMatchObject({
      frame: 7,
      calls: 12,
      triangles: 3_000_001,
      points: 0,
      lines: 4,
      geometries: 5,
      textures: 9,
      programs: 3,
      samples: 1,
    });
    expect(snap?.sampledAt).toBeGreaterThan(0);

    renderer.info.render.calls = 2;
    eventBus.emit('frame-end', {});
    expect(getRendererInfoSnapshot()).toMatchObject({ calls: 2, samples: 2 });
  });

  it('returns a copy, tolerates a missing renderer, and reports null programs on WebGPU-like info', () => {
    let renderer: ReturnType<typeof fakeRenderer> | null = null;
    installRendererInfoSampler(() => renderer);
    eventBus.emit('frame-end', {}); // no renderer yet → nothing sampled
    expect(getRendererInfoSnapshot()).toBeNull();

    renderer = fakeRenderer();
    (renderer.info as { programs?: unknown }).programs = null;
    eventBus.emit('frame-end', {});
    const a = getRendererInfoSnapshot();
    expect(a?.programs).toBeNull();
    a!.calls = -1;
    expect(getRendererInfoSnapshot()?.calls).toBe(12);
  });

  it('a tick that rendered nothing keeps the last rendered frame in renderer.info and the snapshot', () => {
    const base = fakeRenderer();
    const renderer = {
      info: {
        ...base.info,
        autoReset: true,
        reset(): void {
          this.render.calls = 0;
          this.render.triangles = 0;
          this.render.points = 0;
          this.render.lines = 0;
        },
      },
    };
    installRendererInfoSampler(() => renderer);
    // A rendered frame: reset at start, filled by the render, sampled at end.
    eventBus.emit('frame-start', {});
    renderer.info.render.calls = 12;
    renderer.info.render.triangles = 99;
    eventBus.emit('frame-end', { rendered: true });
    expect(getRendererInfoSnapshot()).toMatchObject({ calls: 12, triangles: 99, samples: 1 });

    // A skipped (render-on-change) tick: the reset is undone, nothing sampled.
    eventBus.emit('frame-start', {});
    expect(renderer.info.render.calls).toBe(0);
    eventBus.emit('frame-end', { rendered: false });
    expect(renderer.info.render.calls).toBe(12);
    expect(renderer.info.render.triangles).toBe(99);
    expect(getRendererInfoSnapshot()).toMatchObject({ calls: 12, samples: 1 });
  });

  it('owns the counter reset: autoReset off + reset() at frame-start, restored on uninstall', () => {
    const renderer = { info: { ...fakeRenderer().info, autoReset: true, reset: () => {} } };
    let resets = 0;
    renderer.info.reset = () => {
      resets += 1;
    };
    installRendererInfoSampler(() => renderer);
    expect(renderer.info.autoReset).toBe(true); // untouched until the first frame

    eventBus.emit('frame-start', {});
    expect(renderer.info.autoReset).toBe(false);
    expect(resets).toBe(1);
    eventBus.emit('frame-start', {});
    expect(resets).toBe(2);

    uninstallRendererInfoSampler();
    expect(renderer.info.autoReset).toBe(true);
    eventBus.emit('frame-start', {});
    expect(resets).toBe(2); // unsubscribed
  });

  it('re-install replaces the previous subscription and uninstall resets', () => {
    const first = fakeRenderer();
    const second = fakeRenderer();
    second.info.render.calls = 99;
    installRendererInfoSampler(() => first);
    installRendererInfoSampler(() => second);
    eventBus.emit('frame-end', {});
    // Only one subscription is live: a single sample, from the second renderer.
    expect(getRendererInfoSnapshot()).toMatchObject({ calls: 99, samples: 1 });

    uninstallRendererInfoSampler();
    eventBus.emit('frame-end', {});
    expect(getRendererInfoSnapshot()).toBeNull();
  });
});
