// @vitest-environment jsdom
/**
 * The auto line-primitive policy (#1352 follow-up): the effective-load
 * rule, per-node resolution precedence, policy parsing, and the settings
 * sanitize path. The toggle chain itself (override > default, shader
 * pair selection) is covered by `line-primitive-toggle.test.ts`.
 */
import { afterEach, describe, expect, it } from 'vitest';

import {
  AUTO_QUAD_EFFECTIVE_SEGMENTS,
  DEFAULT_LINE_PRIMITIVE,
  LINE_PRIMITIVE_POLICIES,
  MIN_RENDERED_WIDTH_PX,
  NOMINAL_VIEWPORT_PX,
  effectiveSegmentLoad,
  resolveLinePrimitive,
  resolveLinePrimitiveForNode,
  sceneEffectiveLineLoad,
  setSceneLineLoad,
  setLinePrimitiveOverride,
  setLinePrimitivePolicy,
} from '../../../../types/line-primitive';
import { loadUserSettings, defaultUserSettings } from '../../../../config/user-settings';
import { StorageKeys } from '../../../../utils/storage-keys';

afterEach(() => {
  setSceneLineLoad(0);
  setLinePrimitiveOverride(null);
  setLinePrimitivePolicy('auto');
  localStorage.clear();
});

function lineNode(nSegments: number, attrs: Record<string, unknown> = {}) {
  return {
    path: `/line-${nSegments}`,
    type: 'lines',
    attrs: { n_segments: nSegments, ...attrs },
    hasSpatialIndex: false,
  };
}

function groupNode(kind: 'lod' | 'partition' | undefined, children: ReturnType<typeof lineNode>[]) {
  return {
    path: '/group',
    type: kind ? 'group' : 'scene',
    attrs: kind ? { kind } : {},
    hasSpatialIndex: false,
    children,
  };
}

/** Store a settings object with one advanced field overridden, then load. */
function loadWithStoredPolicy(value: string) {
  const stored = defaultUserSettings() as unknown as { advanced: Record<string, unknown> };
  stored.advanced = { ...stored.advanced, linePrimitivePolicy: value };
  localStorage.setItem(StorageKeys.settings, JSON.stringify(stored));
  return loadUserSettings();
}

describe('LINE_PRIMITIVE_POLICIES', () => {
  it('is the whole policy vocabulary, and is NOT the primitive vocabulary', () => {
    expect(LINE_PRIMITIVE_POLICIES).toEqual(['auto', 'capsule', 'quad']);
    // The two vocabularies are deliberately distinct: the policy says
    // 'quad', the primitive says 'screen-space'. Neither name may leak
    // into the other's validation ('quad' stays unrecognised in
    // parseLinePrimitive — pinned in line-primitive-toggle.test.ts).
    expect(LINE_PRIMITIVE_POLICIES).not.toContain('screen-space');
  });
});

describe('effectiveSegmentLoad', () => {
  it('is the plain segment count when no width/extent information exists', () => {
    expect(effectiveSegmentLoad({ nSegments: 1000 })).toBe(1000);
    expect(effectiveSegmentLoad({ nSegments: 1000, maxWidth: 2 })).toBe(1000);
    expect(effectiveSegmentLoad({ nSegments: 1000, bboxDiagonal: 50 })).toBe(1000);
  });

  it('is zero for missing, non-finite, or non-positive counts', () => {
    expect(effectiveSegmentLoad({})).toBe(0);
    expect(effectiveSegmentLoad({ nSegments: 0 })).toBe(0);
    expect(effectiveSegmentLoad({ nSegments: -5 })).toBe(0);
    expect(effectiveSegmentLoad({ nSegments: Number.NaN })).toBe(0);
  });

  it('scales by the estimated opening-framing pixel width above the render floor', () => {
    // openingPx = maxWidth / diag * NOMINAL_VIEWPORT_PX. Pick values where
    // openingPx = 15 px → factor 15 / MIN_RENDERED_WIDTH_PX = 10.
    const diag = NOMINAL_VIEWPORT_PX; // maxWidth 15 → openingPx exactly 15
    const load = effectiveSegmentLoad({ nSegments: 100_000, maxWidth: 15, bboxDiagonal: diag });
    expect(load).toBeCloseTo(100_000 * (15 / MIN_RENDERED_WIDTH_PX));
  });

  it('floors the width factor at 1 — sub-clamp widths cost like thin lines', () => {
    // openingPx far below the 1.5 px shader clamp: factor must be 1, not <1.
    const load = effectiveSegmentLoad({
      nSegments: 100_000,
      maxWidth: 0.001,
      bboxDiagonal: NOMINAL_VIEWPORT_PX,
    });
    expect(load).toBe(100_000);
  });

  it('ignores degenerate width/extent inputs instead of poisoning the load', () => {
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(effectiveSegmentLoad({ nSegments: 500, maxWidth: bad, bboxDiagonal: 100 })).toBe(500);
      expect(effectiveSegmentLoad({ nSegments: 500, maxWidth: 3, bboxDiagonal: bad })).toBe(500);
    }
  });
});

describe('resolveLinePrimitiveForNode', () => {
  it('auto: capsule below the threshold, quad at and above it', () => {
    expect(resolveLinePrimitiveForNode({ nSegments: AUTO_QUAD_EFFECTIVE_SEGMENTS - 1 })).toBe(
      'capsule'
    );
    expect(resolveLinePrimitiveForNode({ nSegments: AUTO_QUAD_EFFECTIVE_SEGMENTS })).toBe(
      'screen-space'
    );
    expect(resolveLinePrimitiveForNode({ nSegments: 10_000_000 })).toBe('screen-space');
  });

  it('auto: the width factor lowers the effective count threshold', () => {
    // 300k segments alone stay capsule; at ~10× rendered width they cross 2M.
    const load = { nSegments: 300_000, maxWidth: 15, bboxDiagonal: NOMINAL_VIEWPORT_PX };
    expect(resolveLinePrimitiveForNode({ nSegments: 300_000 })).toBe('capsule');
    expect(resolveLinePrimitiveForNode(load)).toBe('screen-space');
  });

  it('auto: no size information means the plain default', () => {
    expect(resolveLinePrimitiveForNode({})).toBe(DEFAULT_LINE_PRIMITIVE);
  });

  it('auto: uses the aggregate concurrent scene load uniformly across sibling nodes', () => {
    const largeScene = groupNode(
      undefined,
      Array.from({ length: 10 }, () => lineNode(1_200_000))
    );
    setSceneLineLoad(sceneEffectiveLineLoad(largeScene));
    expect(resolveLinePrimitiveForNode({ nSegments: 1_200_000 })).toBe('screen-space');
    expect(resolveLinePrimitiveForNode({ nSegments: 100 })).toBe('screen-space');

    const smallScene = groupNode(
      undefined,
      Array.from({ length: 10 }, () => lineNode(100_000))
    );
    setSceneLineLoad(sceneEffectiveLineLoad(smallScene));
    expect(resolveLinePrimitiveForNode({ nSegments: 100_000 })).toBe('capsule');
  });

  it('keeps URL and forced-policy overrides stronger than the aggregate scene load', () => {
    setSceneLineLoad(10_000_000);
    setLinePrimitivePolicy('capsule');
    expect(resolveLinePrimitiveForNode({ nSegments: 100 })).toBe('capsule');
    setLinePrimitiveOverride('screen-space');
    expect(resolveLinePrimitiveForNode({ nSegments: 100 })).toBe('screen-space');
  });

  it('a forced policy replaces the auto rule in both directions', () => {
    setLinePrimitivePolicy('quad');
    expect(resolveLinePrimitiveForNode({ nSegments: 10 })).toBe('screen-space');
    setLinePrimitivePolicy('capsule');
    expect(resolveLinePrimitiveForNode({ nSegments: 10_000_000 })).toBe('capsule');
  });

  it('?linePrimitive= (the session override) beats a forced policy', () => {
    setLinePrimitivePolicy('quad');
    setLinePrimitiveOverride('capsule');
    expect(resolveLinePrimitiveForNode({ nSegments: 10_000_000 })).toBe('capsule');
    expect(resolveLinePrimitive()).toBe('capsule');
  });

  it('a forced policy also drives the session-wide resolution (harness path)', () => {
    setLinePrimitivePolicy('quad');
    expect(resolveLinePrimitive()).toBe('screen-space');
    // An explicit caller value still wins over the forced policy.
    expect(resolveLinePrimitive('capsule')).toBe('capsule');
  });
});

describe('sceneEffectiveLineLoad', () => {
  it('composes partition sums over substitutive LOD maxima', () => {
    const lod = groupNode('lod', [lineNode(400_000), lineNode(1_200_000)]);
    const partition = {
      ...groupNode('partition', [lineNode(900_000)]),
      children: [lineNode(900_000), lod],
    };
    const scene = { ...groupNode(undefined, []), children: [partition] };
    expect(sceneEffectiveLineLoad(scene)).toBe(2_100_000);
  });

  it('takes the maximum substitutive LOD level instead of summing the pyramid', () => {
    const lod = groupNode('lod', [lineNode(400_000), lineNode(1_200_000), lineNode(700_000)]);
    expect(sceneEffectiveLineLoad(lod)).toBe(1_200_000);
  });

  it('uses the same authored width normalization as per-node material resolution', () => {
    const wide = lineNode(300_000, {
      max_width: 15,
      position_bounds: { min: [0, 0], max: [NOMINAL_VIEWPORT_PX, 0] },
    });
    expect(sceneEffectiveLineLoad(wide)).toBeCloseTo(
      effectiveSegmentLoad({
        nSegments: 300_000,
        maxWidth: 15,
        bboxDiagonal: NOMINAL_VIEWPORT_PX,
      })
    );
  });
});

describe('user-settings advanced.linePrimitivePolicy', () => {
  it('defaults to auto and a stored valid value survives the load sanitize', () => {
    expect(defaultUserSettings().advanced.linePrimitivePolicy).toBe('auto');
    expect(loadWithStoredPolicy('quad').advanced.linePrimitivePolicy).toBe('quad');
  });

  it('rejects unknown stored values back to the default', () => {
    expect(loadWithStoredPolicy('volumetric').advanced.linePrimitivePolicy).toBe('auto');
  });
});

describe('extremes and exact crossovers', () => {
  it('one below / at / one above the threshold', () => {
    expect(resolveLinePrimitiveForNode({ nSegments: AUTO_QUAD_EFFECTIVE_SEGMENTS - 1 })).toBe(
      'capsule'
    );
    expect(resolveLinePrimitiveForNode({ nSegments: AUTO_QUAD_EFFECTIVE_SEGMENTS })).toBe(
      'screen-space'
    );
    expect(resolveLinePrimitiveForNode({ nSegments: AUTO_QUAD_EFFECTIVE_SEGMENTS + 1 })).toBe(
      'screen-space'
    );
  });
  it('extreme but finite inputs stay sane', () => {
    expect(effectiveSegmentLoad({ nSegments: 1e12 })).toBe(1e12);
    expect(resolveLinePrimitiveForNode({ nSegments: 1e12 })).toBe('screen-space');
    expect(effectiveSegmentLoad({ nSegments: 1.5 })).toBe(1.5); // fractional count flows, harmless
    expect(effectiveSegmentLoad({ nSegments: Number.MAX_SAFE_INTEGER })).toBe(
      Number.MAX_SAFE_INTEGER
    );
  });
  it('Infinity segments is rejected (guard is isFinite, not just >0)', () => {
    expect(effectiveSegmentLoad({ nSegments: Number.POSITIVE_INFINITY })).toBe(0);
    expect(resolveLinePrimitiveForNode({ nSegments: Number.POSITIVE_INFINITY })).toBe('capsule');
  });
  it('width factor: exact crossover count under a 10x factor', () => {
    // factor 10 => 200k segments is exactly at the effective threshold
    const l = { maxWidth: 15, bboxDiagonal: 1024 };
    expect(resolveLinePrimitiveForNode({ nSegments: 200_000, ...l })).toBe('screen-space');
    expect(resolveLinePrimitiveForNode({ nSegments: 199_999, ...l })).toBe('capsule');
  });
  it('tiny diagonal with huge width does not overflow to a wrong branch', () => {
    expect(
      Number.isFinite(effectiveSegmentLoad({ nSegments: 10, maxWidth: 1e30, bboxDiagonal: 1e-30 }))
    ).toBe(true);
    expect(
      resolveLinePrimitiveForNode({ nSegments: 10, maxWidth: 1e30, bboxDiagonal: 1e-30 })
    ).toBe('screen-space');
  });
});
