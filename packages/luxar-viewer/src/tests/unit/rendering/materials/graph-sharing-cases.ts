/**
 * Shared cases for "TSL materials of one configuration share ONE node graph".
 *
 * three's WebGPU renderer caches a node build under a key that hashes every
 * graph node by its `id` (`NodeMaterial.customProgramCacheKey`). A wrapper
 * that builds a private graph per instance therefore pays a full
 * `NodeBuilder.build` (11–28 ms for a gsplat graph) for every new material —
 * 92 builds in a 6 s WebGPU playback pass of a 50-timepoint partition.
 *
 * The observable without a GPU is that cache key: equal for materials whose
 * configuration selects the same code (whatever their per-object VALUES),
 * different whenever a code-selecting flag differs.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

export interface KeyedMaterial {
  customProgramCacheKey(): string;
  clone(): KeyedMaterial;
}

export interface GraphSharingSpec<M extends KeyedMaterial> {
  /** Build a material from a config. */
  readonly make: (config?: Record<string, unknown>) => M;
  /** Configs that differ only in per-object VALUES (must share). */
  readonly valueConfigs: readonly [Record<string, unknown>, Record<string, unknown>];
  /** Configs whose flags select different code (must not share with `{}` or each other). */
  readonly variants: readonly [string, Record<string, unknown>][];
  /** Rebind the element data texture, when the material has one. */
  readonly bindElementTexture?: (m: M, tex: THREE.DataTexture) => void;
  /** Swap the colormap texture, when the material has one. */
  readonly bindColormap?: (m: M, tex: THREE.DataTexture) => void;
  /** A runtime flag flip and its constructor equivalent. */
  readonly runtimeFlag?: {
    readonly config: Record<string, unknown>;
    readonly set: (m: M) => void;
    readonly unset: (m: M) => void;
  };
}

export function floatTexture(width: number): THREE.DataTexture {
  const tex = new THREE.DataTexture(
    new Float32Array(width * 4),
    width,
    1,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  tex.needsUpdate = true;
  return tex;
}

export function colormapTexture(): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Uint8Array(256 * 4), 256, 1, THREE.RGBAFormat);
  tex.needsUpdate = true;
  return tex;
}

const key = (m: KeyedMaterial): string => m.customProgramCacheKey();

/** Register the graph-sharing cases for one TSL material class. */
export function describeGraphSharing<M extends KeyedMaterial>(
  name: string,
  spec: GraphSharingSpec<M>
): void {
  describe(`${name} graph sharing`, () => {
    it('materials of one configuration share one node-graph cache key', () => {
      const a = spec.make(spec.valueConfigs[0]);
      const b = spec.make(spec.valueConfigs[1]);
      expect(key(b)).toBe(key(a));
      if (spec.bindElementTexture) {
        // A different data texture of the same width and format is a VALUE.
        spec.bindElementTexture(a, floatTexture(4096));
        spec.bindElementTexture(b, floatTexture(4096));
        expect(key(b)).toBe(key(a));
      }
      if (spec.bindColormap) {
        // So is a colormap texture, once both carry one.
        spec.bindColormap(a, colormapTexture());
        spec.bindColormap(b, colormapTexture());
        expect(key(b)).toBe(key(a));
      }
    });

    it('a clone shares its source graph', () => {
      const a = spec.make(spec.valueConfigs[1]);
      expect(key(a.clone())).toBe(key(a));
    });

    it('materials whose flags select different code do not share a key', () => {
      const base = key(spec.make({}));
      const keys = new Set([base]);
      for (const [label, config] of spec.variants) {
        const k = key(spec.make(config));
        expect(k, label).not.toBe(base);
        keys.add(k);
      }
      expect(keys.size).toBe(spec.variants.length + 1);
    });

    if (spec.runtimeFlag) {
      const flag = spec.runtimeFlag;
      it('a runtime flag change moves the material to the other configuration and back', () => {
        const plain = key(spec.make({}));
        const flagged = key(spec.make(flag.config));
        const m = spec.make({});
        flag.set(m);
        expect(key(m)).toBe(flagged);
        flag.unset(m);
        expect(key(m)).toBe(plain);
      });
    }

    if (spec.bindElementTexture) {
      const bind = spec.bindElementTexture;
      it('a data texture of a different width selects a different graph (the width is baked)', () => {
        const a = spec.make({});
        const b = spec.make({});
        bind(a, floatTexture(4096));
        bind(b, floatTexture(48));
        expect(key(b)).not.toBe(key(a));
      });
    }
  });
}
