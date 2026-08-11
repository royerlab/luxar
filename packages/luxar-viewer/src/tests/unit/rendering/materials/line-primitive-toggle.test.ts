/**
 * The `?linePrimitive=` toggle chain (#1352): resolution precedence,
 * shader-source pair selection in the GLSL material, and the
 * peak-projection define lifecycle. TSL-side selection is covered by the
 * codegen snapshots and the tsl-shader-parity `line-volprim-*` fixtures.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

// Thin dispatch spies for the PICK factories: the real implementations
// still run (they are what the parity fixtures validate); the spies only
// record which factory the wrapper's `_rebuild` selected.
vi.mock('../../../../rendering/picking/line/pick-volumetric.tsl', async (importOriginal) => {
  const mod =
    await importOriginal<typeof import('../../../../rendering/picking/line/pick-volumetric.tsl')>();
  return { ...mod, volumetricLinePickWebGPUFactory: vi.fn(mod.volumetricLinePickWebGPUFactory) };
});
vi.mock('../../../../rendering/picking/line/pick.tsl', async (importOriginal) => {
  const mod = await importOriginal<typeof import('../../../../rendering/picking/line/pick.tsl')>();
  return { ...mod, linePickWebGPUFactory: vi.fn(mod.linePickWebGPUFactory) };
});
vi.mock('../../../../rendering/picking/line/pick-capsule.tsl', async (importOriginal) => {
  const mod =
    await importOriginal<typeof import('../../../../rendering/picking/line/pick-capsule.tsl')>();
  return { ...mod, capsuleLinePickWebGPUFactory: vi.fn(mod.capsuleLinePickWebGPUFactory) };
});

import {
  DEFAULT_LINE_PRIMITIVE,
  LINE_PRIMITIVES,
  parseLinePrimitive,
  resolveLinePrimitive,
  setLinePrimitiveOverride,
} from '../../../../types/line-primitive';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import {
  LINE_FRAGMENT_SHADER,
  LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl';
import {
  VOLUMETRIC_LINE_FRAGMENT_SHADER,
  VOLUMETRIC_LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl-volumetric';
import {
  CAPSULE_LINE_FRAGMENT_SHADER,
  CAPSULE_LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl-capsule';

afterEach(() => {
  setLinePrimitiveOverride(null);
});

describe('types/line-primitive', () => {
  it('parses known primitives case/whitespace-insensitively, null otherwise', () => {
    expect(parseLinePrimitive('volumetric')).toBe('volumetric');
    expect(parseLinePrimitive('capsule')).toBe('capsule');
    expect(parseLinePrimitive(' Capsule ')).toBe('capsule');
    expect(parseLinePrimitive(' Screen-Space ')).toBe('screen-space');
    expect(parseLinePrimitive('quads')).toBeNull();
    expect(parseLinePrimitive('')).toBeNull();
    expect(parseLinePrimitive(null)).toBeNull();
    expect(parseLinePrimitive(undefined)).toBeNull();
  });

  it('resolves explicit > session override > default', () => {
    expect(DEFAULT_LINE_PRIMITIVE).toBe('capsule');
    expect(resolveLinePrimitive()).toBe('capsule');
    setLinePrimitiveOverride('volumetric');
    expect(resolveLinePrimitive()).toBe('volumetric');
    // Explicit (harness) argument bypasses the session override.
    expect(resolveLinePrimitive('screen-space')).toBe('screen-space');
    setLinePrimitiveOverride(null);
    expect(resolveLinePrimitive()).toBe('capsule');
  });

  it('keeps the allowed-values list and the type in sync', () => {
    expect(LINE_PRIMITIVES).toEqual(['screen-space', 'volumetric', 'capsule']);
  });
});

describe('LineMaterial primitive selection (GLSL)', () => {
  it('defaults to the CAPSULE pair (the #1352 flip)', () => {
    const m = new LineMaterial();
    expect(m.vertexShader).toBe(CAPSULE_LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(CAPSULE_LINE_FRAGMENT_SHADER);
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.dispose();
  });

  it('the screen-space pair is still selectable explicitly (pre-deletion)', () => {
    const m = new LineMaterial({ primitive: 'screen-space' });
    expect(m.vertexShader).toBe(LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(LINE_FRAGMENT_SHADER);
    m.dispose();
  });

  it('never stamps LUXAR_PEAK_PROJECTION on the screen-space primitive', () => {
    // Program cache keys of the shipping primitive must not change with
    // the flag off — even under peak-family blending modes.
    const m = new LineMaterial({ blendingMode: 'max' });
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.applyBlendingMode('normal');
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.dispose();
  });

  it('selects the volumetric pair for an explicit primitive', () => {
    const m = new LineMaterial({ primitive: 'volumetric' });
    expect(m.vertexShader).toBe(VOLUMETRIC_LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(VOLUMETRIC_LINE_FRAGMENT_SHADER);
    expect(m.userData.linePrimitive).toBe('volumetric');
    m.dispose();
  });

  it('selects the capsule pair for an explicit primitive', () => {
    const m = new LineMaterial({ primitive: 'capsule' });
    expect(m.vertexShader).toBe(CAPSULE_LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(CAPSULE_LINE_FRAGMENT_SHADER);
    expect(m.userData.linePrimitive).toBe('capsule');
    m.dispose();
  });

  it('capsule never stamps LUXAR_PEAK_PROJECTION (peak-shaped by construction)', () => {
    // The capsule profile is already a peak; there is no sum/peak graph
    // split, so peak-family modes must not perturb its program cache keys.
    const m = new LineMaterial({ primitive: 'capsule', blendingMode: 'max' });
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.applyBlendingMode('normal');
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.applyBlendingMode('volumetric');
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.dispose();
  });

  it('capsule clone() round-trips the primitive through the constructor', () => {
    const m = new LineMaterial({ primitive: 'capsule' });
    const c = m.clone();
    expect(c.vertexShader).toBe(CAPSULE_LINE_VERTEX_SHADER);
    expect(c.userData.linePrimitive).toBe('capsule');
    m.dispose();
    c.dispose();
  });

  it('honours the session override when no explicit primitive is passed', () => {
    setLinePrimitiveOverride('volumetric');
    const m = new LineMaterial();
    expect(m.vertexShader).toBe(VOLUMETRIC_LINE_VERTEX_SHADER);
    m.dispose();
    setLinePrimitiveOverride('capsule');
    const c = new LineMaterial();
    expect(c.vertexShader).toBe(CAPSULE_LINE_VERTEX_SHADER);
    c.dispose();
  });

  it('manages the peak/sum define across blending-mode changes (volumetric only)', () => {
    const m = new LineMaterial({ primitive: 'volumetric', blendingMode: 'additive' });
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.applyBlendingMode('max'); // peak family
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(true);
    m.applyBlendingMode('normal'); // still peak family
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(true);
    m.applyBlendingMode('volumetric'); // sum family
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.dispose();
  });

  it('clone() round-trips the primitive through the constructor', () => {
    const m = new LineMaterial({ primitive: 'volumetric' });
    const c = m.clone();
    expect(c.vertexShader).toBe(VOLUMETRIC_LINE_VERTEX_SHADER);
    expect(c.userData.linePrimitive).toBe('volumetric');
    m.dispose();
    c.dispose();
  });

  it('TSL wrapper mirrors the peak/sum define lifecycle and factory selection', async () => {
    const { LineTSLMaterial } = await import('../../../../rendering/materials/line/material-tsl');
    const m = new LineTSLMaterial({ primitive: 'volumetric', blendingMode: 'max' });
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(true);
    m.applyBlendingMode('additive'); // sum family clears the tracker
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
    m.applyBlendingMode('normal'); // peak family re-stamps it
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(true);
    const c = m.clone();
    expect(c.userData.linePrimitive).toBe('volumetric');
    expect('LUXAR_PEAK_PROJECTION' in (c.defines ?? {})).toBe(true);
    // Screen-space TSL material never stamps it, even under peak modes.
    const q = new LineTSLMaterial({ blendingMode: 'max' });
    expect('LUXAR_PEAK_PROJECTION' in (q.defines ?? {})).toBe(false);
    // Capsule TSL material never stamps it either.
    const cap = new LineTSLMaterial({ primitive: 'capsule', blendingMode: 'max' });
    expect('LUXAR_PEAK_PROJECTION' in (cap.defines ?? {})).toBe(false);
    cap.applyBlendingMode('normal');
    expect('LUXAR_PEAK_PROJECTION' in (cap.defines ?? {})).toBe(false);
    expect(cap.clone().userData.linePrimitive).toBe('capsule');
    m.dispose();
    c.dispose();
    q.dispose();
    cap.dispose();
  });

  it('volumetric PICK shaders never import erf (peak lane only)', async () => {
    // The pick fragment is the peak capsule unconditionally — if an erf
    // symbol shows up, sum-lane machinery leaked into the pick pass.
    const { VOLUMETRIC_LINE_PICK_FRAGMENT_SHADER } =
      await import('../../../../rendering/picking/line/shaders-volumetric');
    expect(VOLUMETRIC_LINE_PICK_FRAGMENT_SHADER).not.toContain('luxarErf');
    expect(VOLUMETRIC_LINE_PICK_FRAGMENT_SHADER).toContain('gl_FragDepth');
  });

  it('volumetric shaders import the SHARED erf implementations', () => {
    // The spike embedded a stale pre-re-solve c0; the production shader
    // must carry the shared module's coefficient (erf.ts, re-solved
    // against the rounded tail) and the A&S form for the mixed lane.
    expect(VOLUMETRIC_LINE_FRAGMENT_SHADER).toContain('1.126454454'); // erfPoly c0
    expect(VOLUMETRIC_LINE_FRAGMENT_SHADER).not.toContain('1.126422828'); // spike c0
    expect(VOLUMETRIC_LINE_FRAGMENT_SHADER).toContain('luxarErfAS'); // A&S mixed lane
  });
});

describe('LinePickingMaterial primitive selection (#1352 PR-3)', () => {
  it('defaults to the CAPSULE pick pair (the #1352 flip)', async () => {
    const { LinePickingMaterial } = await import('../../../../rendering/picking/line/material');
    const { CAPSULE_LINE_PICK_VERTEX_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER } =
      await import('../../../../rendering/picking/line/shaders-capsule');
    const m = new LinePickingMaterial({ nodeId: 7 });
    expect(m.vertexShader).toBe(CAPSULE_LINE_PICK_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(CAPSULE_LINE_PICK_FRAGMENT_SHADER);
    m.dispose();
  });

  it('the screen-space pick pair is still selectable explicitly (pre-deletion)', async () => {
    const { LinePickingMaterial } = await import('../../../../rendering/picking/line/material');
    const { LINE_PICK_VERTEX_SHADER, LINE_PICK_FRAGMENT_SHADER } =
      await import('../../../../rendering/picking/line/shaders');
    const m = new LinePickingMaterial({ nodeId: 7, primitive: 'screen-space' });
    expect(m.vertexShader).toBe(LINE_PICK_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(LINE_PICK_FRAGMENT_SHADER);
    m.dispose();
  });

  it('selects the volumetric pick pair (explicit and via session override)', async () => {
    const { LinePickingMaterial } = await import('../../../../rendering/picking/line/material');
    const { VOLUMETRIC_LINE_PICK_VERTEX_SHADER, VOLUMETRIC_LINE_PICK_FRAGMENT_SHADER } =
      await import('../../../../rendering/picking/line/shaders-volumetric');
    const explicit = new LinePickingMaterial({ nodeId: 7, primitive: 'volumetric' });
    expect(explicit.vertexShader).toBe(VOLUMETRIC_LINE_PICK_VERTEX_SHADER);
    expect(explicit.fragmentShader).toBe(VOLUMETRIC_LINE_PICK_FRAGMENT_SHADER);
    expect(explicit.userData.linePrimitive).toBe('volumetric');
    explicit.dispose();
    // The session override — the production path: create-lines-node passes
    // no primitive, so the pick material must resolve the same override the
    // visual material does or the two rasterize different stencils.
    setLinePrimitiveOverride('volumetric');
    const viaOverride = new LinePickingMaterial({ nodeId: 7 });
    expect(viaOverride.vertexShader).toBe(VOLUMETRIC_LINE_PICK_VERTEX_SHADER);
    viaOverride.dispose();
  });

  it('selects the capsule pick pair (explicit and via session override)', async () => {
    const { LinePickingMaterial } = await import('../../../../rendering/picking/line/material');
    const { CAPSULE_LINE_PICK_VERTEX_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER } =
      await import('../../../../rendering/picking/line/shaders-capsule');
    const explicit = new LinePickingMaterial({ nodeId: 7, primitive: 'capsule' });
    expect(explicit.vertexShader).toBe(CAPSULE_LINE_PICK_VERTEX_SHADER);
    expect(explicit.fragmentShader).toBe(CAPSULE_LINE_PICK_FRAGMENT_SHADER);
    expect(explicit.userData.linePrimitive).toBe('capsule');
    const c = explicit.clone();
    expect(c.vertexShader).toBe(CAPSULE_LINE_PICK_VERTEX_SHADER);
    expect(c.userData.linePrimitive).toBe('capsule');
    explicit.dispose();
    c.dispose();
    setLinePrimitiveOverride('capsule');
    const viaOverride = new LinePickingMaterial({ nodeId: 7 });
    expect(viaOverride.vertexShader).toBe(CAPSULE_LINE_PICK_VERTEX_SHADER);
    viaOverride.dispose();
  });

  it('clone() round-trips the primitive', async () => {
    const { LinePickingMaterial } = await import('../../../../rendering/picking/line/material');
    const { VOLUMETRIC_LINE_PICK_VERTEX_SHADER } =
      await import('../../../../rendering/picking/line/shaders-volumetric');
    const m = new LinePickingMaterial({ nodeId: 7, primitive: 'volumetric' });
    const c = m.clone();
    expect(c.vertexShader).toBe(VOLUMETRIC_LINE_PICK_VERTEX_SHADER);
    expect(c.userData.linePrimitive).toBe('volumetric');
    m.dispose();
    c.dispose();
  });

  it('TSL wrapper dispatches to the volumetric pick factory (and back)', async () => {
    const { LinePickingTSLMaterial } =
      await import('../../../../rendering/picking/line/material-tsl');
    const volFactory = vi.mocked(
      (await import('../../../../rendering/picking/line/pick-volumetric.tsl'))
        .volumetricLinePickWebGPUFactory
    );
    const quadFactory = vi.mocked(
      (await import('../../../../rendering/picking/line/pick.tsl')).linePickWebGPUFactory
    );
    volFactory.mockClear();
    quadFactory.mockClear();

    const vol = new LinePickingTSLMaterial({ nodeId: 7, primitive: 'volumetric' });
    expect(volFactory).toHaveBeenCalledTimes(1);
    expect(quadFactory).not.toHaveBeenCalled();
    // A texture rebind rebuilds through the SAME factory.
    const tex = new (await import('three')).DataTexture(
      new Float32Array(24),
      6,
      1,
      (await import('three')).RGBAFormat,
      (await import('three')).FloatType
    );
    vol.updateLineTexture(tex);
    expect(volFactory).toHaveBeenCalledTimes(2);
    expect(quadFactory).not.toHaveBeenCalled();
    // clone carries the variant through the constructor.
    const c = vol.clone();
    expect(c.userData.linePrimitive).toBe('volumetric');
    expect(quadFactory).not.toHaveBeenCalled();

    volFactory.mockClear();
    const quad = new LinePickingTSLMaterial({ nodeId: 7, primitive: 'screen-space' });
    expect(quadFactory).toHaveBeenCalledTimes(1);
    expect(volFactory).not.toHaveBeenCalled();

    const capFactory = vi.mocked(
      (await import('../../../../rendering/picking/line/pick-capsule.tsl'))
        .capsuleLinePickWebGPUFactory
    );
    capFactory.mockClear();
    quadFactory.mockClear();
    const cap = new LinePickingTSLMaterial({ nodeId: 7, primitive: 'capsule' });
    expect(capFactory).toHaveBeenCalledTimes(1);
    expect(quadFactory).not.toHaveBeenCalled();
    expect(volFactory).not.toHaveBeenCalled();
    expect(cap.clone().userData.linePrimitive).toBe('capsule');

    vol.dispose();
    c.dispose();
    quad.dispose();
    cap.dispose();
    tex.dispose();
  });
});
