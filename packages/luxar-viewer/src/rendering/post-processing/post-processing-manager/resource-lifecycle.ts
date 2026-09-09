/**
 * Pure helpers for `PostProcessingManager` resource construction +
 * sizing. The orchestrator stays focused on event dispatch and setter
 * routing.
 *
 * Everything here is a factory or a pure function — no mutation of
 * external state. The orchestrator owns the resources and assigns
 * the returned values back to its private fields.
 *
 * @module rendering/post-processing/post-processing-manager/resource-lifecycle
 */

import * as THREE from 'three';
import { BloomChain } from '../bloom/chain';
import { FxaaPass } from '../fxaa/pass';
import { FullscreenPass } from '../fullscreen/pass';
import { computeEffectiveRenderSize, computeRenderTargetAllocation } from '../render-target-sizing';
import { resolveToneMappingDefault } from '../tone-mapping';
import { materialManager, type LuxarMegaShaderMaterial } from '../../material-manager';
import { config } from '../../../config';
import type { Renderer, RendererCapabilities } from '../../renderer-capabilities';
import { DataRefractionSplit } from './refraction-split';
import type { GlassPartition } from '../../materials/_shared/glass-partition';

/** Inputs to the size-derivation pair. */
export interface SizingInputs {
  readonly renderer: Renderer;
  readonly renderSize: { readonly width: number; readonly height: number };
  readonly ssaaEnabled: boolean;
  readonly ssaaMultiplier: number;
  readonly maxPhysicalDimension: number;
}

/** Effective (logical SSAA) render size in CSS pixels. */
export function computeEffectiveSize(s: SizingInputs): { width: number; height: number } {
  return computeEffectiveRenderSize(s.renderSize, s.ssaaEnabled, s.ssaaMultiplier);
}

/**
 * Physical-pixel framebuffer size derived from the effective size, DPR,
 * and framebuffer limit. Render targets and the mega-shader / bloom /
 * FXAA passes are all sized here so they match the renderer's canvas
 * backbuffer AND what materials read from `renderer.getDrawingBufferSize()`.
 * At DPR > 1 a mismatch would silently brighten the scene via over-coverage.
 */
export function getPhysicalSize(s: SizingInputs): { width: number; height: number } {
  return getRenderTargetAllocation(s).physical;
}

/** Matching logical renderer size and physical attachment size. */
export function getRenderTargetAllocation(s: SizingInputs) {
  return computeRenderTargetAllocation(
    s.renderSize,
    s.ssaaEnabled,
    s.ssaaMultiplier,
    s.renderer.getPixelRatio(),
    s.maxPhysicalDimension
  );
}

/** Allocate the HDR target the scene renders into. */
export function createHdrTarget(
  physW: number,
  physH: number,
  msaaSamples: number
): THREE.WebGLRenderTarget {
  const t = new THREE.WebGLRenderTarget(physW, physH, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: true,
    stencilBuffer: false,
    samples: msaaSamples,
  });
  t.texture.name = 'PostProcessing.hdrTarget';
  return t;
}

/** Allocate the LDR intermediate (mega-shader output, HalfFloat for lossless EXR). */
export function createLdrTarget(physW: number, physH: number): THREE.WebGLRenderTarget {
  const t = new THREE.WebGLRenderTarget(physW, physH, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });
  t.texture.name = 'PostProcessing.ldrTarget';
  return t;
}

/** Bundle of mutable resources the orchestrator carries between pipeline calls. */
export interface PipelineResources {
  hdrTarget: THREE.WebGLRenderTarget;
  ldrTarget: THREE.WebGLRenderTarget;
  megaShader: LuxarMegaShaderMaterial;
  megaPass: FullscreenPass;
  bloomChain: BloomChain | null;
  fxaaPass: FxaaPass | null;
  /**
   * The scene-pass split that lets `refract_data` glass refract the data behind it
   * while the data in front stays crisp — on both backends. Nullable only for the
   * disposed state.
   */
  refractionSplit: DataRefractionSplit | null;
}

/** Configuration handed in by the orchestrator for resource construction. */
export interface BuildResourcesConfig {
  readonly physW: number;
  readonly physH: number;
  readonly msaaSamples: number;
  readonly fxaaEnabled: boolean;
  readonly capabilities: RendererCapabilities;
  readonly bloomLevels: number;
  readonly bloomRadius: number;
  readonly bloomThreshold: number;
  readonly bloomIntensity: number;
  /**
   * When false (context-restore path), the caller will restore user
   * toggle state from a snapshot — skip the config-defaults bloom
   * allocation so we don't override a previously-disabled choice.
   */
  readonly allocateBloomFromDefaults: boolean;
  /**
   * Source of the visible refracting glass for the refraction split (the depth-sort
   * coordinator's `collectRefractingGlass`). Injected so this module owns no scene
   * knowledge.
   */
  readonly collectRefractingGlass: (out: THREE.Mesh[]) => THREE.Mesh[];
  /**
   * Source of the visible meshes three's own materials draw (the coordinator's
   * `collectUnpartitionedMeshes`): drawn in the split's pass A only.
   */
  readonly collectUnpartitionedMeshes: (out: THREE.Mesh[]) => THREE.Mesh[];
  /** The per-pass partition-mode broadcast (the coordinator's `applyGlassPartition`). */
  readonly setGlassPartition: (mode: GlassPartition) => void;
}

/**
 * Build the full transient-resource set. Returns the bundle; the
 * orchestrator assigns each field back to its private store.
 */
export function buildTransientResources(c: BuildResourcesConfig): PipelineResources {
  const hdrTarget = createHdrTarget(c.physW, c.physH, c.msaaSamples);
  const ldrTarget = createLdrTarget(c.physW, c.physH);

  // Mega-shader + fullscreen mesh. The materialManager dispatches on
  // caps.apiSurface so the WebGPU path returns the TSL counterpart.
  const megaShader = materialManager.createMegaShaderMaterial({
    exposure: config.renderingControls.defaults.exposure,
    globalOffset: config.renderingControls.defaults.globalOffset,
    globalGamma: config.renderingControls.defaults.globalGamma,
    toneMapping: resolveToneMappingDefault(),
  });
  megaShader.setResolution(c.physW, c.physH);

  // FullscreenPass owns the caps-aware triangle geometry so the
  // orchestrator never has to think about framebuffer-Y orientation.
  const megaPass = new FullscreenPass(megaShader, c.capabilities);

  // Bloom chain (built only when enabled; on by default per config).
  // Skip during context-restore rebuilds — the caller restores the
  // user's bloom-enabled choice from a snapshot.
  let bloomChain: BloomChain | null = null;
  if (c.allocateBloomFromDefaults && config.renderingControls.defaults.bloomEnabled) {
    bloomChain = new BloomChain({
      levels: c.bloomLevels,
      threshold: c.bloomThreshold,
      smoothing: 0.01,
      radius: c.bloomRadius,
      width: c.physW,
      height: c.physH,
      caps: c.capabilities,
    });
    megaShader.toggleBloom(true);
    megaShader.setBloom(c.bloomIntensity, bloomChain.outputTexture);
  }

  // FXAA pass (built only when enabled).
  const fxaaPass = c.fxaaEnabled ? new FxaaPass(c.physW, c.physH, c.capabilities) : null;

  // The refraction split runs on both renderers: the data partition (glass depth
  // pre-pass, data behind, glass, data in front) is backend-agnostic; only the copy
  // + screen quad that three's WebGLRenderer transmission pass needs is WebGL-only,
  // and the split branches on `apiSurface` for that.
  const refractionSplit = new DataRefractionSplit({
    width: c.physW,
    height: c.physH,
    transmissionResolutionScale: config.renderingControls.refraction.transmissionResolutionScale,
    apiSurface: c.capabilities.apiSurface,
    collectRefractingGlass: c.collectRefractingGlass,
    collectUnpartitionedMeshes: c.collectUnpartitionedMeshes,
    setGlassPartition: c.setGlassPartition,
  });

  return { hdrTarget, ldrTarget, megaShader, megaPass, bloomChain, fxaaPass, refractionSplit };
}

/** Allocate a fresh bloom chain and bind it to the mega-shader. */
export function buildBloomChain(
  width: number,
  height: number,
  cfg: {
    levels: number;
    threshold: number;
    radius: number;
    intensity: number;
    caps: RendererCapabilities;
    megaShader: LuxarMegaShaderMaterial;
  }
): BloomChain {
  const chain = new BloomChain({
    levels: cfg.levels,
    threshold: cfg.threshold,
    smoothing: 0.01,
    radius: cfg.radius,
    width,
    height,
    caps: cfg.caps,
  });
  cfg.megaShader.toggleBloom(true);
  cfg.megaShader.setBloom(cfg.intensity, chain.outputTexture);
  return chain;
}

/** Dispose every transient resource in the bundle. */
export function disposeTransientResources(r: PipelineResources): void {
  r.hdrTarget?.dispose();
  r.ldrTarget?.dispose();
  r.bloomChain?.dispose();
  r.fxaaPass?.dispose();
  r.megaShader?.dispose();
  r.megaPass?.dispose();
  r.refractionSplit?.dispose();
}

/** Inputs for the noise re-scaling helper. */
export interface ScaledNoiseInputs {
  readonly megaShader: LuxarMegaShaderMaterial;
  readonly currentDPRScale: number;
  readonly baseNoiseSettings: {
    readonly readoutSigma: number;
    readonly photonGain: number;
    readonly fpnSigma: number;
  };
}

/**
 * Re-apply base sigmas through the DPR scaling.
 *
 * Scaling contract (MED-28): the photon-gain term scales as `DPRScale²`
 * while the readout and FPN sigmas scale **linearly** with `DPRScale`.
 * This asymmetry is intentional and reflects the underlying physics:
 *
 *   - photon-gain models shot noise, whose variance is proportional to
 *     the number of samples integrated per output pixel (≈ DPRScale²).
 *     The gain term enters the shader as a variance, so it scales by
 *     DPRScale².
 *   - readout and FPN sigmas are per-pixel noise terms independent of
 *     integration time / sample count, so they scale linearly with
 *     `DPRScale` to preserve the visual standard deviation as the
 *     effective pixel size changes.
 *
 * Do NOT "fix" this for visual symmetry — squaring or linearising
 * both terms breaks the noise model. When `DPR < 1` the effective
 * pixel is larger, so all three terms shrink (gain quadratically,
 * sigmas linearly). When `DPR > 1` they grow in the same proportions.
 */
export function applyScaledNoiseSettings(s: ScaledNoiseInputs): void {
  if (!s.megaShader.isDetectorNoiseEnabled()) return;
  const k = s.currentDPRScale;
  s.megaShader.setDetectorNoise({
    readoutSigma: s.baseNoiseSettings.readoutSigma * k,
    photonGain: s.baseNoiseSettings.photonGain * k * k,
    fpnSigma: s.baseNoiseSettings.fpnSigma * k,
  });
}
