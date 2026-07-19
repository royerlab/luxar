/**
 * Expected THREE material state per Luxar blending mode — the numeric
 * twins of `getCompleteBlendingState` (src/rendering/blending-state.ts),
 * shared by blending-modes.spec.ts and lines-blending-modes.spec.ts.
 *
 * THREE enum values (three/src/constants.js, verified against three@r184):
 *   blending:      NormalBlending=1, AdditiveBlending=2, CustomBlending=5
 *   blendEquation: AddEquation=100, MaxEquation=104
 *   blendSrc/Dst:  OneFactor=201, SrcAlphaFactor=204,
 *                  OneMinusSrcAlphaFactor=205
 *
 * `normal.depthWrite` here is the opacity>=0.99 value (fixtures author
 * their normal layers at the default opacity 1.0); callers comparing
 * layers at other opacities must derive it via
 * `opacity >= 0.99` (normalModeDepthWrite) instead, as the points half
 * of blending-modes.spec.ts does.
 */

export interface ExpectedBlendState {
  blending: number;
  blendEquation: number;
  blendSrc: number;
  blendDst: number;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
}

export const EXPECTED_BLEND_STATE: Record<string, ExpectedBlendState> = {
  additive: {
    blending: 2, // AdditiveBlending
    blendEquation: 100, // AddEquation
    blendSrc: 204, // SrcAlphaFactor
    blendDst: 201, // OneFactor
    depthTest: false,
    depthWrite: false,
    transparent: true,
  },
  luminous: {
    blending: 2, // AdditiveBlending (like additive, but depth-tested)
    blendEquation: 100, // AddEquation
    blendSrc: 204, // SrcAlphaFactor
    blendDst: 201, // OneFactor
    depthTest: true,
    depthWrite: false,
    transparent: true,
  },
  max: {
    blending: 5, // CustomBlending
    blendEquation: 104, // MaxEquation
    blendSrc: 201, // OneFactor
    blendDst: 201, // OneFactor
    depthTest: true,
    depthWrite: false,
    transparent: true,
  },
  opaque: {
    blending: 1, // NormalBlending
    blendEquation: 100, // AddEquation
    blendSrc: 204, // SrcAlphaFactor
    blendDst: 205, // OneMinusSrcAlphaFactor
    depthTest: true,
    depthWrite: true,
    transparent: false,
  },
  normal: {
    blending: 1, // NormalBlending
    blendEquation: 100, // AddEquation
    blendSrc: 204, // SrcAlphaFactor
    blendDst: 205, // OneMinusSrcAlphaFactor
    depthTest: true,
    depthWrite: true, // at opacity >= 0.99 (normalModeDepthWrite)
    transparent: true,
  },
  // PHASE-1 POINTS/LINES VALUE: point/line materials intercept
  // 'volumetric' and apply the ADDITIVE state (the exact κ=0 limit —
  // VOLUMETRIC_BLENDING_SPEC.md §5.1) until phases 3–4 implement the
  // emission–absorption math for those geometries. The two consumers of
  // this map iterate POINTS and LINES layers, so this row deliberately
  // mirrors `additive` above. GSplats use
  // EXPECTED_GSPLAT_VOLUMETRIC_STATE below. userData.blendingMode still
  // reads 'volumetric' (the requested mode is preserved).
  volumetric: {
    blending: 2, // AdditiveBlending (phase-1 fallback)
    blendEquation: 100, // AddEquation
    blendSrc: 204, // SrcAlphaFactor
    blendDst: 201, // OneFactor
    depthTest: false,
    depthWrite: false,
    transparent: true,
  },
};

/**
 * The REAL volumetric state — gsplats only in phase 1: premultiplied
 * emission–absorption over the One/OneMinusSrcAlpha framebuffer state
 * (numerically identical to the gsplat `normal` state; the semantics
 * live in the fragment shader). depthWrite is false UNCONDITIONALLY.
 */
export const EXPECTED_GSPLAT_VOLUMETRIC_STATE: ExpectedBlendState = {
  blending: 5, // CustomBlending
  blendEquation: 100, // AddEquation
  blendSrc: 201, // OneFactor (shader premultiplies)
  blendDst: 205, // OneMinusSrcAlphaFactor
  depthTest: true,
  depthWrite: false,
  transparent: true,
};
