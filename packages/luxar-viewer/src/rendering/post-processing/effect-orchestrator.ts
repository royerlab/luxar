/**
 * Effect-orchestrator helpers for the post-processing pipeline.
 *
 * The hot logic of `rebuildEffectPass()` is the partition: given the
 * canonical, ordered list of active effects, split them into "Pass A"
 * (the primary `EffectPass`) and "Pass B" (a fall-back `EffectPass`
 * for incompatibilities).
 *
 * pmndrs/postprocessing has known incompatibilities between
 * convolution effects (Bloom) and UV-transformation effects
 * (ChromaticLensDistortion) — they cannot share a pass. The
 * partitioning rules here mirror those constraints.
 *
 * Extracted from `post-processing-manager.ts` so the partition is
 * unit-tested without an `EffectComposer` or any live effect
 * instance — we feed in `{ effect, name }` records, the partitioner
 * is generic over the effect type.
 *
 * @module rendering/post-processing/effect-orchestrator
 */

/** Names of effects classified as UV-transform (cannot share with convolution). */
const UV_TRANSFORM_EFFECT_NAMES: ReadonlySet<string> = new Set(['ChromaticLensDistortion']);

/**
 * Names of effects classified as convolution (cannot share with UV-transform).
 *
 * Bloom uses mipmap-blur which samples multiple texels; it is a
 * convolution effect by pmndrs's classification.
 */
const CONVOLUTION_EFFECT_NAMES: ReadonlySet<string> = new Set(['Bloom']);

export function isUVTransformEffectName(name: string): boolean {
  return UV_TRANSFORM_EFFECT_NAMES.has(name);
}

export function isConvolutionEffectName(name: string): boolean {
  return CONVOLUTION_EFFECT_NAMES.has(name);
}

/** One entry in the canonical ordered effect list. */
export interface OrderedEffect<T> {
  effect: T;
  name: string;
}

/**
 * Optional effect references plus the AA-mode flags. Each effect
 * field is added to the ordered list only when defined; `smaaEnabled`
 * / `fxaaEnabled` gate the (mutually exclusive) anti-aliasing entry.
 *
 * Generic over the effect type so this stays decoupled from the
 * pmndrs `Effect` class — tests use string markers.
 */
export interface EffectSlots<T> {
  bloom?: T;
  dof?: T;
  ao?: T;
  chromaticLensDistortion?: T;
  detectorNoise?: T;
  toneMapping?: T;
  vignette?: T;
  smaa?: T;
  fxaa?: T;
  /** Whether SMAA should be added. Wins over FXAA when both are true. */
  smaaEnabled: boolean;
  /** Whether FXAA should be added (only honored when smaaEnabled is false). */
  fxaaEnabled: boolean;
}

/**
 * Build the canonical ordered list of active effects from a slot
 * record. The order is the project-defined visual order (HDR effects
 * before tone mapping → LDR effects after → AA last). Only-defined
 * slots contribute; AA selection is gated on the `*Enabled` flags
 * with SMAA winning over FXAA when both flags are true.
 *
 * Pure: no side effects, no logging, no scene access. The caller
 * passes the result to {@link partitionEffectsIntoPasses} to split
 * into Pass A / Pass B for the EffectComposer.
 */
export function buildOrderedEffects<T>(slots: EffectSlots<T>): OrderedEffect<T>[] {
  const ordered: OrderedEffect<T>[] = [];

  // HDR effects (before tone mapping)
  if (slots.bloom) ordered.push({ effect: slots.bloom, name: 'Bloom' });
  if (slots.dof) ordered.push({ effect: slots.dof, name: 'DOF' });
  if (slots.ao) ordered.push({ effect: slots.ao, name: 'AO' });

  // Chromatic Lens Distortion: wavelength-dependent radial distortion.
  if (slots.chromaticLensDistortion) {
    ordered.push({ effect: slots.chromaticLensDistortion, name: 'ChromaticLensDistortion' });
  }

  // Detector noise: after lens distortion / chromatic aberration, before tone mapping.
  if (slots.detectorNoise) {
    ordered.push({ effect: slots.detectorNoise, name: 'DetectorNoise' });
  }

  // Tone mapping (HDR → LDR conversion).
  if (slots.toneMapping) ordered.push({ effect: slots.toneMapping, name: 'ToneMapping' });

  // LDR effects (after tone mapping). pmndrs v7 requires vignette after tone mapping.
  if (slots.vignette) ordered.push({ effect: slots.vignette, name: 'Vignette' });

  // Anti-aliasing always last; SMAA wins over FXAA when both flags are true.
  if (slots.smaaEnabled && slots.smaa) {
    ordered.push({ effect: slots.smaa, name: 'SMAA' });
  } else if (slots.fxaaEnabled && slots.fxaa) {
    ordered.push({ effect: slots.fxaa, name: 'FXAA' });
  }

  return ordered;
}

/**
 * Result of {@link partitionEffectsIntoPasses}. `passA*` is always the
 * primary pass; `passB*` is non-empty only when an incompatibility
 * forced a split.
 */
export interface EffectPartition<T> {
  passA: T[];
  passB: T[];
  passANames: string[];
  passBNames: string[];
  /** First effect name that triggered the split (if any). */
  splitAt: string | null;
}

/**
 * Partition the ordered effect list into Pass A and Pass B.
 *
 * Algorithm: walk the list adding each effect to Pass A. As soon as
 * adding an effect would violate the (UV-transform ↔ convolution)
 * exclusion rule, switch to Pass B and put that effect (and all
 * remaining effects) there.
 *
 * Pure — operates only on the names; the partitioner is generic over
 * the effect object so it can be unit-tested with plain markers.
 */
export function partitionEffectsIntoPasses<T>(
  ordered: ReadonlyArray<OrderedEffect<T>>
): EffectPartition<T> {
  const passA: T[] = [];
  const passB: T[] = [];
  const passANames: string[] = [];
  const passBNames: string[] = [];
  let usingPassB = false;
  let splitAt: string | null = null;

  for (const { effect, name } of ordered) {
    if (usingPassB) {
      passB.push(effect);
      passBNames.push(name);
      continue;
    }

    const hasUV = passANames.some(isUVTransformEffectName);
    const hasConv = passANames.some(isConvolutionEffectName);
    const incompatible =
      (isUVTransformEffectName(name) && hasConv) || (isConvolutionEffectName(name) && hasUV);

    if (incompatible) {
      usingPassB = true;
      splitAt = name;
      passB.push(effect);
      passBNames.push(name);
    } else {
      passA.push(effect);
      passANames.push(name);
    }
  }

  return { passA, passB, passANames, passBNames, splitAt };
}
