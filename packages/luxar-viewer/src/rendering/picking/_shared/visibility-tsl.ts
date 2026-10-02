/**
 * TSL pick visibility weight — twin of `GLSL_PICK_VISIBILITY`
 * (`./visibility-glsl.ts`, which carries the rationale): the per-element
 * alpha factor (optical depth under `volumetric`), the node opacity and the
 * visual discards' `max(gain, 1)`.
 *
 * `uVolumetric` is a runtime uniform (not a graph flag, unlike the visual
 * factories' compile-time blending branch) so a blending-mode change never
 * rebuilds a pick graph; the pick pass is half-resolution and rare, and both
 * arms are a handful of ALU.
 *
 * @module rendering/picking/_shared/visibility-tsl
 */
import type * as THREE from 'three';
import { float, int, mix, min, uniform } from 'three/tsl';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import { ALPHA_CLAMP } from '../../materials/_shared/volumetric';
import {
  PICK_VISIBILITY_DEFAULTS,
  PICK_VISIBILITY_UNIFORM_NAMES,
  type PickVisibilityUniformName,
} from './visibility-uniforms';

/** The four synced uniforms the weight reads. */
export type PickVisibilityTSLNodes = Readonly<Record<PickVisibilityUniformName, TSLNode>>;

/** Fresh leaves at the neutral defaults (a wrapper's own value holders). */
export function createPickVisibilityTSLNodes(): Record<PickVisibilityUniformName, TSLNode> {
  return {
    uIntensity: uniform(PICK_VISIBILITY_DEFAULTS.uIntensity),
    uOpacity: uniform(PICK_VISIBILITY_DEFAULTS.uOpacity),
    uHasElementAlpha: uniform(PICK_VISIBILITY_DEFAULTS.uHasElementAlpha),
    uVolumetric: uniform(PICK_VISIBILITY_DEFAULTS.uVolumetric),
  };
}

/** `proxyIUniform` records over a wrapper's leaves, for its `uniforms` map. */
export function proxyPickVisibilityUniforms(
  nodes: PickVisibilityTSLNodes
): Record<PickVisibilityUniformName, THREE.IUniform> {
  const out = {} as Record<PickVisibilityUniformName, THREE.IUniform>;
  for (const name of PICK_VISIBILITY_UNIFORM_NAMES) out[name] = proxyIUniform(nodes[name]);
  return out;
}

/** Snapshot-adapter leaves from a flat `IUniform` record (neutral when absent). */
export function pickVisibilityTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): PickVisibilityTSLNodes {
  const out = {} as Record<PickVisibilityUniformName, TSLNode>;
  for (const name of PICK_VISIBILITY_UNIFORM_NAMES) {
    out[name] = uniform((uniforms[name]?.value as number) ?? PICK_VISIBILITY_DEFAULTS[name]);
  }
  return out;
}

/** The visual pass's alpha factor for a (sanitized) per-element alpha. */
export function pickAlphaFactorTSL(alpha: TSLNode, nodes: PickVisibilityTSLNodes): TSLNode {
  const opticalDepth: TSLNode = mix(
    float(1.0),
    min(alpha, float(ALPHA_CLAMP)).oneMinus().log().negate(),
    nodes.uHasElementAlpha
  );
  return int(nodes.uVolumetric).equal(int(1)).select(opticalDepth, alpha);
}

/** alpha factor × node opacity × max(gain, 1) — `luxarPickWeight` in GLSL. */
export function pickWeightTSL(alpha: TSLNode, nodes: PickVisibilityTSLNodes): TSLNode {
  return pickAlphaFactorTSL(alpha, nodes)
    .mul(nodes.uOpacity)
    .mul(nodes.uIntensity.max(float(1.0)));
}
