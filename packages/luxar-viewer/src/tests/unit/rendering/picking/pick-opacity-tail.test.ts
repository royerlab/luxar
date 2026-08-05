/**
 * Pick-material opacity-tail tripwire (all FOUR geometry types).
 *
 * THREE's NodeMaterial appends `DiffuseColor.w *= material.opacity` to every
 * generated fragment — visible in the codegen snapshots as
 * `DiffuseColor.w = DiffuseColor.w * nodeUniformN`. The sibling
 * `materials/tsl-opacity-tail.test.ts` pins that tail for the VISUAL
 * materials, where a stray opacity would merely mis-dim a pixel.
 *
 * On the PICK materials the stakes are higher: the element index is carried in
 * two 16-bit halves and the HIGH half rides in the alpha channel, so a
 * non-identity multiplier does not dim anything — it decodes a **wrong element
 * id** for any index above 65535. `NoBlending` does not suppress the multiply
 * (it governs the blend stage, not the fragment body), and the hand-written
 * GLSL twins have no such tail, so the corruption would appear on the
 * WebGPU/TSL path only — invisible to a WebGL-only suite.
 *
 * The factories therefore pin `.opacity = 1` explicitly rather than relying on
 * the THREE default, which also covers the injected-`outMaterial` path where a
 * caller could hand in a material carrying someone else's opacity. These pins
 * turn that invariant into a loud failure the day it changes.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  buildPointPickTSLNodesFromUniforms,
  pointPickWebGPUFactory,
} from '../../../../rendering/picking/point/pick.tsl';
import {
  buildLinePickTSLNodesFromUniforms,
  linePickWebGPUFactory,
} from '../../../../rendering/picking/line/pick.tsl';
import {
  buildGSplatPickTSLNodesFromUniforms,
  gsplatPickWebGPUFactory,
} from '../../../../rendering/picking/gsplat/pick.tsl';
import {
  buildMeshPickTSLNodesFromUniforms,
  meshPickWebGPUFactory,
} from '../../../../rendering/picking/mesh/pick.tsl';

/** Build each pick material, optionally reusing a caller-supplied material. */
const build = (out?: () => THREE.NodeMaterial) =>
  [
    ['point', pointPickWebGPUFactory(buildPointPickTSLNodesFromUniforms({}), out?.())],
    ['line', linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms({}), {}, out?.())],
    ['gsplat', gsplatPickWebGPUFactory(buildGSplatPickTSLNodesFromUniforms({}), out?.())],
    // Mesh reaches this tripwire for the same reason and with one extra twist: its
    // element id is a VERTEX ordinal off `gl_VertexID`, bounded only by the vertex
    // count rather than by a texture-layout capacity, so the high half is populated
    // on any mesh past 65,535 vertices — an ordinary size, not an extreme one.
    ['mesh', meshPickWebGPUFactory(buildMeshPickTSLNodesFromUniforms({}), out?.())],
  ] as const;

describe('pick-material opacity tail is identity (element-id high half rides in alpha)', () => {
  it('opacity is exactly 1 on a freshly built pick material', () => {
    for (const [name, mat] of build()) {
      expect(
        mat.opacity,
        `${name} pick: opacity must be exactly 1 or the element-id high half is scaled`
      ).toBe(1);
    }
  });

  it('an injected outMaterial carrying a foreign opacity is forced back to 1', () => {
    // The regression this guards: `outMaterial ?? new NodeMaterial()` means the
    // factory does not always own a fresh material, so inheriting 0.5 here
    // would silently halve every high half.
    const dimmed = () => {
      const m = new THREE.NodeMaterial();
      m.opacity = 0.5;
      return m;
    };
    for (const [name, mat] of build(dimmed)) {
      expect(mat.opacity, `${name} pick: factory leaked the caller's opacity`).toBe(1);
    }
  });

  it('keeps the rest of the opaque-ID-buffer contract intact', () => {
    // Guards against "fixing" the tail by making the material transparent or
    // blended, either of which would smear ids across overlapping picks.
    for (const [name, mat] of build()) {
      expect(mat.transparent, `${name} pick: must stay opaque`).toBe(false);
      expect(mat.blending, `${name} pick: must stay NoBlending`).toBe(THREE.NoBlending);
      expect(mat.toneMapped, `${name} pick: tone mapping would corrupt ids`).toBe(false);
    }
  });
});
