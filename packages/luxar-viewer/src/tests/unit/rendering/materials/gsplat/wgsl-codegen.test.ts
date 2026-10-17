/**
 * The gsplat TSL graph's generated WGSL keeps the GLSL twin's cost structure
 * (#3140). Three properties no pixel test can see, because each one changes
 * only how much work the GPU does:
 *
 *   1. Every per-splat varying is `flat, either` (compat-safe flat sampling).
 *      Only the clip `zw` the glass partition reads stays interpolated.
 *   2. The vertex stage returns early: no texel fetch runs before the density
 *      test, and the covariance projection runs inside the near / coverage
 *      fade test (GLSL twin: the `return`s in main()).
 *   3. The fragment work past the truncation test sits inside an `if`. WGSL
 *      `discard` only demotes the invocation to a helper, which keeps running
 *      to the end of the shader; GLSL's terminates it.
 *
 * The observable without a GPU is the generated WGSL: a real
 * `WGSLNodeBuilder` builds the production material (no device needed).
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { context } from 'three/tsl';
import { WGSLNodeBuilder, type NodeMaterial } from 'three/webgpu';

import type { BlendingMode } from '../../../../../types/blending';
import { GSplatTSLMaterial } from '../../../../../rendering/materials/gsplat/material-tsl';

/** The renderer surface `WGSLNodeBuilder.build()` reads (no GPU device needed). */
function stubRenderer(): unknown {
  return {
    getRenderTarget: () => null,
    getMRT: () => null,
    depth: true,
    reversedDepthBuffer: false,
    logarithmicDepthBuffer: false,
    contextNode: context(),
    library: { fromMaterial: (m: unknown) => m },
    hasFeature: () => false,
    hasCompatibility: () => false,
    backend: {
      compatibilityMode: false,
      capabilities: { getUniformBufferLimit: () => 65536 },
      utils: {
        getTextureSampleData: () => ({ samples: 1, primarySamples: 1, isMSAA: false }),
      },
    },
  };
}

/** The `NodeBuilder` members this test drives (three's typings omit them). */
interface BuilderSurface {
  camera: THREE.Camera;
  scene: THREE.Scene;
  material: NodeMaterial;
  vertexShader: string;
  fragmentShader: string;
  build(): void;
}

function build(blendingMode: BlendingMode): { vertex: string; fragment: string } {
  const material = new GSplatTSLMaterial({ blendingMode });
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
  const builder = new WGSLNodeBuilder(
    mesh,
    stubRenderer() as ConstructorParameters<typeof WGSLNodeBuilder>[1]
  ) as unknown as BuilderSurface;
  builder.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  builder.scene = new THREE.Scene();
  builder.material = material as unknown as NodeMaterial;
  builder.build();
  return { vertex: String(builder.vertexShader), fragment: String(builder.fragmentShader) };
}

/** The `@location` inputs of the fragment entry point. */
function fragmentInputs(fragment: string): string[] {
  const entry = fragment.indexOf('@fragment');
  const signature = fragment.slice(entry, fragment.indexOf('->', entry));
  return signature.match(/@location\( \d+ \)(?: @interpolate\( [^)]*\))?/g) ?? [];
}

/** The text of the first `{ ... }` block opened at or after `from`. */
function blockAt(src: string, from: number): string {
  const open = src.indexOf('{', from);
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) return src.slice(open, i + 1);
  }
  throw new Error('unbalanced block');
}

describe('gsplat TSL WGSL keeps the GLSL cost structure (#3140)', () => {
  it.each(['normal', 'volumetric', 'additive'] as const)('%s', (mode) => {
    const { vertex, fragment } = build(mode);

    // 1. Six varyings: five compat-safe flat values plus the interpolated clip zw.
    const inputs = fragmentInputs(fragment);
    expect(inputs).toHaveLength(6);
    expect(inputs.filter((v) => v.includes('@interpolate( flat, either )'))).toHaveLength(5);

    // 2. No texel fetch before the density test, and the projection Jacobian
    // (the first read of P[0].xy) inside a nested early-out block.
    const body = vertex.slice(vertex.indexOf('@vertex'));
    const firstIf = body.indexOf('if (');
    expect(body.indexOf('textureLoad')).toBeGreaterThan(firstIf);
    const densityBlock = blockAt(body, firstIf);
    expect(densityBlock).toContain('textureLoad');
    const jacobian = body.indexOf('render.cameraProjectionMatrix[ 0 ].xy');
    expect(densityBlock).toContain('render.cameraProjectionMatrix[ 0 ].xy');
    const fadeTest = body.lastIndexOf('< 0.01 )', jacobian);
    expect(fadeTest).toBeGreaterThan(firstIf);
    expect(blockAt(body, body.lastIndexOf('if (', fadeTest))).toContain(
      'render.cameraProjectionMatrix[ 0 ].xy'
    );

    // 3. The falloff exp() runs only in the else of the truncation test,
    // whose then-branch is the discard.
    const frag = fragment.slice(fragment.indexOf('@fragment'));
    const expAt = frag.indexOf('exp(');
    const elseAt = frag.lastIndexOf('} else {', expAt);
    const truncation = frag.lastIndexOf('if (', elseAt);
    expect(truncation).toBeGreaterThan(-1);
    expect(frag.slice(truncation, elseAt)).toMatch(
      /> object\.nodeUniform\d+ \) \) \{\s*discard;\s*$/
    );
    expect(blockAt(frag, elseAt)).toContain('exp(');
  });
});
