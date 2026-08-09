/**
 * The `?linePrimitive=` toggle chain (#1352): resolution precedence,
 * shader-source pair selection in the GLSL material, and the
 * peak-projection define lifecycle. TSL-side selection is covered by the
 * codegen snapshots and the tsl-shader-parity `line-volprim-*` fixtures.
 */
import { afterEach, describe, expect, it } from 'vitest';

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

afterEach(() => {
  setLinePrimitiveOverride(null);
});

describe('types/line-primitive', () => {
  it('parses known primitives case/whitespace-insensitively, null otherwise', () => {
    expect(parseLinePrimitive('volumetric')).toBe('volumetric');
    expect(parseLinePrimitive(' Screen-Space ')).toBe('screen-space');
    expect(parseLinePrimitive('quads')).toBeNull();
    expect(parseLinePrimitive('')).toBeNull();
    expect(parseLinePrimitive(null)).toBeNull();
    expect(parseLinePrimitive(undefined)).toBeNull();
  });

  it('resolves explicit > session override > default', () => {
    expect(DEFAULT_LINE_PRIMITIVE).toBe('screen-space');
    expect(resolveLinePrimitive()).toBe('screen-space');
    setLinePrimitiveOverride('volumetric');
    expect(resolveLinePrimitive()).toBe('volumetric');
    // Explicit (harness) argument bypasses the session override.
    expect(resolveLinePrimitive('screen-space')).toBe('screen-space');
    setLinePrimitiveOverride(null);
    expect(resolveLinePrimitive()).toBe('screen-space');
  });

  it('keeps the allowed-values list and the type in sync', () => {
    expect(LINE_PRIMITIVES).toEqual(['screen-space', 'volumetric']);
  });
});

describe('LineMaterial primitive selection (GLSL)', () => {
  it('defaults to the screen-space pair, byte-identical to the pre-toggle build', () => {
    const m = new LineMaterial();
    expect(m.vertexShader).toBe(LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(LINE_FRAGMENT_SHADER);
    expect('LUXAR_PEAK_PROJECTION' in (m.defines ?? {})).toBe(false);
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

  it('honours the session override when no explicit primitive is passed', () => {
    setLinePrimitiveOverride('volumetric');
    const m = new LineMaterial();
    expect(m.vertexShader).toBe(VOLUMETRIC_LINE_VERTEX_SHADER);
    m.dispose();
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
    m.dispose();
    c.dispose();
    q.dispose();
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
