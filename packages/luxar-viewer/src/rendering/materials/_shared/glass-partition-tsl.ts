/**
 * TSL twin of `glass-partition.ts`: the per-fragment depth partition of the emissive
 * data around `refract_data` glass, for the WebGPURenderer graphs (native WebGPU and its
 * WebGL2 fallback).
 *
 * Two things differ from the GLSL side, both forced by TSL:
 *
 * 1. **The fragment's own window depth is a varying.** Every Luxar data graph replaces
 *    the vertex stage (`material.vertexNode`), so three's `positionView`-derived depth
 *    nodes would read a quad-corner attribute, and no TSL node exposes `fragCoord.z`.
 *    The vertex body assigns its clip `zw` to {@link clipDepthVarying}; per fragment
 *    `z / w` is the rasterizer's own screen-linear depth (perspective-correct
 *    interpolation of `z` and of `w` cancel), remapped to window depth by
 *    {@link GlassFragmentDepthNode} according to the renderer's coordinate system —
 *    `z/w` already IS window depth under `WebGPUCoordinateSystem`, `z/w * 0.5 + 0.5`
 *    under WebGL's. Window depth in [0, 1] is the one quantity both conventions share
 *    with the depth texture, so the compare needs no further care.
 * 2. **The depth texture is sampled at `screenUV`**, exactly as three's own
 *    `viewportDepthTexture` does: `screenUV` is top-left-origin on both backends and
 *    `TextureNode` Y-flips a depth texture on the GLSL builder (`_flipYUniform`), so the
 *    same graph is orientation-correct on WGSL and on the WebGL2 fallback. A depth
 *    texture's node type is `float`, so the sample IS the depth (no `.r`).
 *
 * The guard emits its two `Discard`s onto the CALLER's stack — it must be called inside
 * the fragment `Fn` body, first — rather than through a nested `Fn`, the same rule the
 * other shared helpers in `tsl-helpers.ts` follow.
 *
 * @module rendering/materials/_shared/glass-partition-tsl
 */

import * as THREE from 'three';
import {
  Discard,
  float,
  If,
  nodeObject,
  screenUV,
  texture,
  uniform,
  varying,
  vec2,
} from 'three/tsl';
import { Node, type NodeBuilder } from 'three/webgpu';
import { getGlassDepthTexture } from './glass-partition';
import type { TSLNode } from './tsl-helpers';

/** The two partition nodes every TSL data graph carries (keys match the GLSL uniforms). */
export interface GlassPartitionTSLNodes {
  /** Mode as a float uniform: 0 off, 1 keep behind-or-none, 2 keep front. */
  readonly uGlassPartition: TSLNode;
  /** `texture()` node over the ONE shared refracting-glass depth texture. */
  readonly uGlassDepth: TSLNode;
}

/**
 * Fresh live nodes for a material wrapper: the partition uniform the split writes
 * through the `proxyIUniform` bridge, and a texture node over the shared depth texture
 * (bound once; the split's render target is created WITH that texture, so no rebind —
 * and hence no graph rebuild — is ever needed).
 */
export function glassPartitionNodes(): GlassPartitionTSLNodes {
  return { uGlassPartition: uniform(0), uGlassDepth: texture(getGlassDepthTexture()) };
}

/**
 * The `build*TSLNodesFromUniforms` form: values captured from a GLSL-shaped uniform
 * record when present (the parity harness), else the off / shared-texture defaults so
 * a record that never heard of the partition builds the identical graph.
 */
export function glassPartitionNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): GlassPartitionTSLNodes {
  return {
    uGlassPartition: uniform((uniforms.uGlassPartition?.value as number) ?? 0),
    uGlassDepth: texture(
      (uniforms.uGlassDepth?.value as THREE.Texture | null | undefined) ?? getGlassDepthTexture()
    ),
  };
}

/**
 * The clip-space `zw` varying the vertex body fills right before returning its clip
 * position (`vClipZW.assign(clipPos.zw)`). Declared outside the vertex `Fn` like every
 * other varying; the default is a far, valid `(0, 1)`.
 */
export function clipDepthVarying(): TSLNode {
  return varying(vec2(0.0, 1.0));
}

/**
 * Window depth of the current fragment from its interpolated clip `zw`, remapped per
 * coordinate system at graph-build time (the builder knows the renderer; the graph
 * factories do not, and the same factory serves native WebGPU and the WebGL2 fallback).
 */
export class GlassFragmentDepthNode extends Node {
  constructor(private readonly clipZW: TSLNode) {
    super('float');
  }

  /**
   * Build the window-depth expression for this renderer: `z/w` under the WebGPU
   * coordinate system, `z/w * 0.5 + 0.5` under WebGL's.
   */
  override setup(builder: NodeBuilder): TSLNode {
    const ndcZ: TSLNode = this.clipZW.x.div(this.clipZW.y);
    return builder.renderer.coordinateSystem === THREE.WebGPUCoordinateSystem
      ? ndcZ
      : ndcZ.mul(0.5).add(0.5);
  }
}

/**
 * Emit the partition onto the current fragment stack — call FIRST inside the fragment
 * `Fn` body so a classified-away fragment costs nothing further. Mirrors
 * `GLSL_GLASS_PARTITION_GUARD` line for line:
 *
 * ```
 * inFront = glassDepth < 1.0 && fragmentDepth < glassDepth
 * mode 1 discards inFront; mode 2 discards !inFront
 * ```
 *
 * Both discards are gated on the mode, so a graph running outside the split (mode 0)
 * keeps every fragment; the texture sample is inside the same uniform branch.
 */
export function glassPartitionGuardTSL(nodes: GlassPartitionTSLNodes, vClipZW: TSLNode): void {
  If(nodes.uGlassPartition.notEqual(float(0.0)), () => {
    const fragDepth: TSLNode = nodeObject(new GlassFragmentDepthNode(vClipZW));
    const glassDepth: TSLNode = nodes.uGlassDepth.sample(screenUV);
    const inFront: TSLNode = glassDepth.lessThan(float(1.0)).and(fragDepth.lessThan(glassDepth));
    Discard(nodes.uGlassPartition.equal(float(1.0)).and(inFront));
    Discard(nodes.uGlassPartition.equal(float(2.0)).and(inFront.not()));
  });
}
