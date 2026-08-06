/**
 * Expected THREE material state per Luxar blending mode — the numeric
 * twins of `getCompleteBlendingState` (src/rendering/blending-state.ts),
 * shared by blending-modes.spec.ts and lines-blending-modes.spec.ts.
 *
 * Deliberately duplicated as NUMBERS, not imported: the Playwright spec
 * bundle must not import viewer/THREE modules (an import would make the
 * expectation derive from the very code under test — and drag the THREE
 * dependency into the node-side test runner). Do not "DRY this up".
 *
 * THREE enum values (three/src/constants.js, verified against three@r184):
 *   blending:      NormalBlending=1, AdditiveBlending=2, CustomBlending=5
 *   blendEquation: AddEquation=100, MaxEquation=104
 *   blendSrc/Dst:  OneFactor=201, SrcAlphaFactor=204,
 *                  OneMinusSrcAlphaFactor=205
 *
 * `normal.depthWrite: true` here is the LINE value: lines keep the
 * opacity-gated `normalModeDepthWrite` predicate (true at the fixtures'
 * default opacity 1.0), and lines-blending-modes.spec.ts depends on this
 * numeric. POINTS, by contrast, ALWAYS use `depthWrite: false` in
 * `normal` (a point sprite stamps a flat depth plane across the whole
 * disc, so sorted transparency never depth-writes — #1002);
 * blending-modes.spec.ts overrides the points expectation to false.
 */

/**
 * Shape of one expected THREE material blend state: the numeric
 * blending/blendEquation/blendSrc/blendDst plus the depthTest/depthWrite/
 * transparent flags a blending-mode spec asserts against.
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

/**
 * Expected THREE material state keyed by Luxar blending mode (additive,
 * normal, ...) — the hard-coded numeric twin of `getCompleteBlendingState`,
 * asserted by the blending-mode E2E specs (see the file header for why these
 * values are duplicated as literals rather than imported).
 */
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
  // The REAL volumetric state — all three geometry types (gsplats
  // phase 1, points phase 3, lines phase 4): premultiplied
  // emission–absorption over the One/OneMinusSrcAlpha framebuffer
  // state (numerically identical to the gsplat `normal` state; the
  // semantics live in the fragment shader). depthWrite is false
  // UNCONDITIONALLY.
  volumetric: {
    blending: 5, // CustomBlending
    blendEquation: 100, // AddEquation
    blendSrc: 201, // OneFactor (shader premultiplies)
    blendDst: 205, // OneMinusSrcAlphaFactor
    depthTest: true,
    depthWrite: false,
    transparent: true,
  },
};
