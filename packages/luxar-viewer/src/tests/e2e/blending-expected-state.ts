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
};
