/**
 * TSL twin of `./refraction-apodization.ts` — the refraction-shift edge apodization on
 * WebGPU.
 *
 * Three's TSL refraction (`getIBLVolumeRefraction` in `PhysicalLightingModel`) is
 * module-private, so the shift is shaped through its inputs, exactly as the GLSL twin
 * does: `thickness` and `attenuationDistance` — the property nodes three's
 * `setupVariants` has just assigned — are rescaled by the ray scale `λ`, computed from
 * the same `n`, `v` and position three's `start()` passes (`normalWorld`, the camera
 * direction, `positionWorld`). The math, the knee and the floor are the reference
 * module's; see it for the derivation.
 *
 * The NDC flip WebGPU applies to `y` does not matter here: the room is measured to the
 * nearest border of each axis, which is symmetric in the sign.
 *
 * @module rendering/materials/mesh-physical/refraction-apodization-tsl
 */

import {
  abs,
  attenuationDistance,
  cameraPosition,
  cameraProjectionMatrix,
  cameraViewMatrix,
  dispersion,
  float,
  ior,
  length,
  max,
  min,
  modelWorldMatrix,
  normalize,
  normalWorld,
  positionWorld,
  refract,
  tanh,
  thickness,
  vec3,
  vec4,
} from 'three/tsl';
import type { TSLNode } from '../_shared/tsl-helpers';
import {
  DISPERSION_HALF_SPREAD_PER_UNIT,
  REFRACTION_MIN_RAY_SCALE,
  REFRACTION_SHIFT_CEILING,
  REFRACTION_SHIFT_KNEE,
} from './refraction-apodization';

/** `apodizeShiftAxis`, as a node: the factor for one axis of the shift. */
function apodizeShiftAxisTSL(shift: TSLNode, room: TSLNode): TSLNode {
  const knee: TSLNode = room.mul(REFRACTION_SHIFT_KNEE).toVar();
  const span: TSLNode = room.mul(REFRACTION_SHIFT_CEILING - REFRACTION_SHIFT_KNEE).toVar();
  const rolled: TSLNode = knee
    .add(span.mul(tanh(shift.sub(knee).div(max(span, REFRACTION_MIN_RAY_SCALE)))))
    .div(max(shift, REFRACTION_MIN_RAY_SCALE));
  return shift
    .lessThanEqual(knee)
    .select(float(1.0), span.lessThanEqual(0.0).select(float(0.0), rolled));
}

/**
 * The ray scale `λ` for the current fragment, mirroring `luxarRefractionRayScale`.
 * @param withDispersion - Three's `useDispersion`: trace the most refracted IOR.
 */
export function refractionRayScaleTSL(withDispersion: boolean): TSLNode {
  const v: TSLNode = cameraPosition.sub(positionWorld).normalize().toVar();
  const iorMax: TSLNode = withDispersion
    ? ior.add(ior.sub(1.0).mul(dispersion.mul(DISPERSION_HALF_SPREAD_PER_UNIT))).toVar()
    : ior;
  const model: TSLNode = modelWorldMatrix;
  const modelScale: TSLNode = vec3(
    length(model.element(0).xyz),
    length(model.element(1).xyz),
    length(model.element(2).xyz)
  );
  const ray: TSLNode = normalize(
    refract(v.negate(), normalize(normalWorld), float(1.0).div(iorMax))
  )
    .mul(thickness.mul(modelScale))
    .toVar();
  const c0: TSLNode = cameraProjectionMatrix
    .mul(cameraViewMatrix.mul(vec4(positionWorld, 1.0)))
    .toVar();
  const c1: TSLNode = cameraProjectionMatrix
    .mul(cameraViewMatrix.mul(vec4(positionWorld.add(ray), 1.0)))
    .toVar();
  const w0: TSLNode = max(c0.w, REFRACTION_MIN_RAY_SCALE);
  const w1: TSLNode = max(c1.w, REFRACTION_MIN_RAY_SCALE);
  const b: TSLNode = c0.xy.div(w0).toVar();
  const s: TSLNode = abs(c1.xy.div(w1).sub(b)).toVar();
  const room: TSLNode = max(float(1.0).sub(abs(b)), 0.0).toVar();
  const kappa: TSLNode = min(
    apodizeShiftAxisTSL(s.x, room.x),
    apodizeShiftAxisTSL(s.y, room.y)
  ).toVar();
  const lambda: TSLNode = kappa.mul(w0).div(kappa.mul(w0).add(float(1.0).sub(kappa).mul(w1)));
  const behind: TSLNode = c0.w.lessThanEqual(0.0).or(c1.w.lessThanEqual(0.0));
  return behind.select(float(REFRACTION_MIN_RAY_SCALE), max(lambda, REFRACTION_MIN_RAY_SCALE));
}

/**
 * Rescale the property nodes three's transmission reads. Call at the END of
 * `setupVariants`, after three has assigned `ior`, `thickness`, `attenuationDistance`
 * and (with dispersion) `dispersion`.
 */
export function applyRefractionApodizationTSL(withDispersion: boolean): void {
  const scale: TSLNode = refractionRayScaleTSL(withDispersion).toVar('LuxarRefractionRayScale');
  (thickness as TSLNode).assign(thickness.mul(scale));
  (attenuationDistance as TSLNode).assign(attenuationDistance.mul(scale));
}
