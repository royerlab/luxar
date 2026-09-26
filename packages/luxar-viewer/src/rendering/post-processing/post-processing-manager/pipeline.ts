/**
 * Pipeline runner for PostProcessingManager.
 *
 * The function is pure over its `PipelineCtx` argument — no `this`
 * reference — so it can be unit-tested with a synthesised mock renderer
 * and mesh stack.
 *
 * @module rendering/post-processing/post-processing-manager/pipeline
 */

import * as THREE from 'three';
import type { Renderer } from '../../renderer-capabilities';
import type { LuxarMegaShaderMaterial } from '../../material-manager';
import type { BloomChain } from '../bloom/chain';
import type { FullscreenPass } from '../fullscreen/pass';
import type { FxaaPass } from '../fxaa/pass';
import type { DataRefractionSplit } from './refraction-split';

/** Read-only references the pipeline runner needs from the orchestrator. */
export interface PipelineCtx {
  readonly renderer: Renderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.Camera;
  readonly hdrTarget: THREE.WebGLRenderTarget;
  readonly ldrTarget: THREE.WebGLRenderTarget;
  readonly megaShader: LuxarMegaShaderMaterial;
  readonly megaPass: FullscreenPass;
  readonly bloomChain: BloomChain | null;
  readonly fxaaPass: FxaaPass | null;
  /**
   * The scene-pass split for `refract_data` glass (spec §3.4 Phase 3), both backends;
   * null only once disposed. See {@link renderSceneToHdr}.
   */
  readonly refractionSplit: DataRefractionSplit | null;
}

/**
 * Stage (0) of the pipeline, shared with the raw HDR capture path: the scene into the
 * HDR target. Binds the target, then either lets the refraction split draw the frame
 * in its passes (glass depth, data behind, glass, data in front) — when some visible
 * glass asks to refract the data — or renders the scene once, exactly as before
 * Phase 3. Leaves the HDR target bound.
 *
 * Each path clears the target itself: the split's first HDR pass renders with
 * `autoClear` on, and so does the plain render in every pipeline configuration. An
 * explicit `clear()` here as well cleared the full HDR target a second time every
 * frame; it is kept only for a caller that has turned `autoClear` off.
 */
export function renderSceneToHdr(ctx: PipelineCtx): void {
  ctx.renderer.setRenderTarget(ctx.hdrTarget);
  if (ctx.refractionSplit?.render(ctx.renderer, ctx.scene, ctx.camera, ctx.hdrTarget)) return;
  if (!ctx.renderer.autoClear) ctx.renderer.clear();
  ctx.renderer.render(ctx.scene, ctx.camera);
}

/**
 * Run the full pipeline: scene → HDR → (bloom) → mega-shader →
 * (FXAA) → finalTarget.
 *
 * - `applyFxaa = true`: mega-shader writes to ldrTarget, FXAA reads
 *   ldrTarget and writes to `finalTarget`.
 * - `applyFxaa = false`: mega-shader writes directly to `finalTarget`.
 * - `finalTarget = null`: write to the canvas backbuffer.
 *
 * The capture paths use `applyFxaa = false` + an explicit target so
 * they can read the post-tone-mapping LDR buffer without an FXAA
 * pass between mega-shader and readback.
 */
export function runPipeline(
  ctx: PipelineCtx,
  opts: { applyFxaa: boolean; finalTarget: THREE.WebGLRenderTarget | null }
): void {
  // Defensive save/restore: most call sites don't care about
  // pre-existing renderer state, but a future caller might invoke
  // runPipeline while another target is bound. Mirrors
  // BloomChain.render's pattern so the pipeline stays composable.
  const prevTarget = ctx.renderer.getRenderTarget();
  const prevAutoClear = ctx.renderer.autoClear;
  try {
    // (0) Scene → HDR target (one pass, or the refraction split's two)
    renderSceneToHdr(ctx);

    // (1) Bloom pyramid
    if (ctx.bloomChain) {
      ctx.bloomChain.render(ctx.renderer, ctx.hdrTarget.texture);
    }

    // (2) Mega-shader
    ctx.megaShader.setHdrSceneTexture(ctx.hdrTarget.texture);

    if (opts.applyFxaa && ctx.fxaaPass) {
      // Mega → ldrTarget → FXAA → finalTarget
      ctx.renderer.setRenderTarget(ctx.ldrTarget);
      ctx.megaPass.render(ctx.renderer);
      ctx.renderer.setRenderTarget(opts.finalTarget);
      ctx.fxaaPass.render(ctx.renderer, ctx.ldrTarget.texture);
    } else {
      // Mega → finalTarget directly (no FXAA)
      ctx.renderer.setRenderTarget(opts.finalTarget);
      ctx.megaPass.render(ctx.renderer);
    }
  } finally {
    // Cast is a TypeScript-only narrowing: `setRenderTarget` on
    // the `Renderer` union is typed against `WebGLRenderTarget`
    // while `getRenderTarget` returns the looser `RenderTarget`.
    // Round-tripping at runtime is safe on either backend.
    ctx.renderer.setRenderTarget(prevTarget as THREE.WebGLRenderTarget | null);
    ctx.renderer.autoClear = prevAutoClear;
  }
}
