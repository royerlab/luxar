/**
 * Live texture binding for TSL post-processing graphs.
 *
 * Kept out of `tsl-helpers.ts` for the same reason `glass-partition-tsl.ts`
 * is a separate file: this needs `NodeUpdateType` from `three/webgpu`, and
 * `tsl-helpers.ts` is imported by modules that must stay clear of the WebGPU
 * cone (#1679). The `-tsl` suffix is what puts this file inside that lazy
 * cone as far as the import restriction is concerned — it is not cosmetic.
 *
 * @module rendering/materials/_shared/live-texture-tsl
 */

import type * as THREE from 'three';
import type { texture } from 'three/tsl';
import { NodeUpdateType } from 'three/webgpu';

/**
 * Bind a TSL `texture()` node to a live `IUniform` slot.
 *
 * `texture()` captures the THREE.Texture handed to it at factory-build
 * time, while the host re-points `uniforms.uInput.value` at a different
 * texture between passes (BloomChain walks the mip pyramid; FxaaPass
 * builds with `null` and assigns the LDR target per render) — so the node
 * has to re-resolve the texture on every render or it samples the
 * placeholder forever and the pass outputs an empty image under WebGPU.
 *
 * The swap MUST happen in `updateBefore`, not in `update`
 * (`.onUpdate(…, 'render')`). Every `node.sample(…)` tap is a CLONE of
 * this node that reads the texture through `referenceNode`, and each
 * clone derives its own render-target Y-flip uniform from that texture
 * inside `TextureNode.update()` — three's WebGL-backend emulation of
 * WebGPU's top-down framebuffer convention. Node updates run in graph
 * order with no guarantee that the bound node comes first, so swapping
 * during `update` lets the taps that were already updated derive their
 * flip from the PREVIOUS texture. On the first bloom downsample pass that
 * predecessor is the non-render-target placeholder, so one of the four
 * taps sampled un-flipped while the other three sampled flipped, and the
 * box average collapsed to half the vertical gradient (#2584). The
 * 5-tap upsample tent is worse, because its first-built tap is the CENTRE
 * (`c.mul(0.5)` is the outer `.add`'s left operand, so the setup-stage
 * depth-first walk registers it ahead of the four 0.125 taps): the stale
 * weight is exactly 0.5, and a 0.5/0.5 mix of a gradient with its own
 * mirror collapses to a spatially CONSTANT contribution equal to the
 * centre-row value — verbatim #2584's `levels: 2` decomposition. The
 * renderer runs the whole graph's `updateBefore` ahead of any `update`,
 * so binding there makes every tap see the same texture.
 *
 * The mismatch only bites while the predecessor texture differs in
 * `isRenderTargetTexture`, i.e. on the first render after each material
 * build; from the second frame on every tap already agrees. A RESIZE is
 * not such a build: `BloomChain.setSize`/`setLevels` only dispose and
 * reallocate the mip targets, while the three NodeMaterials and their
 * bound texture nodes are constructed once and survive, so a resize just
 * rebinds one render-target mip over another. The materials are built by
 * `buildBloomChain`, which runs when bloom goes from disabled to enabled
 * (`PostProcessingManager.setBloomEnabled(true)` — including the first
 * enable that allocates the chain) and on WebGL context restore. That is
 * still enough to matter — every session that shows bloom pays it once —
 * and it is why the fix is applied to FXAA too even though its
 * parity-harness entry supplies a real texture at build time and so never
 * exercises the swap.
 *
 * @param node - The TSL texture node to keep bound.
 * @param slot - The host-owned uniform whose `.value` is the live texture.
 * @param fallback - Stable identity used while `slot.value` is null.
 */
export function bindLiveTexture<T extends ReturnType<typeof texture>>(
  node: T,
  slot: THREE.IUniform,
  fallback: THREE.Texture
): T {
  node.updateBeforeType = NodeUpdateType.OBJECT;
  node.updateBefore = (): undefined => {
    node.value = (slot.value as THREE.Texture | null) ?? fallback;
  };
  return node;
}
