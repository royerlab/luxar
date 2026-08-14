/**
 * The `?linePrimitive=` toggle chain (#1352): resolution precedence,
 * shader-source pair selection in the GLSL material, and factory
 * dispatch in the TSL wrappers. TSL-side selection is covered by the
 * codegen snapshots and the tsl-shader-parity `line-capsule-*` fixtures.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

// Thin dispatch spies for the PICK factories: the real implementations
// still run (they are what the parity fixtures validate); the spies only
// record which factory the wrapper's `_rebuild` selected.
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
  CAPSULE_LINE_FRAGMENT_SHADER,
  CAPSULE_LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl-capsule';

afterEach(() => {
  setLinePrimitiveOverride(null);
});

describe('types/line-primitive', () => {
  it('parses known primitives case/whitespace-insensitively, null otherwise', () => {
    expect(parseLinePrimitive('capsule')).toBe('capsule');
    expect(parseLinePrimitive(' Capsule ')).toBe('capsule');
    expect(parseLinePrimitive(' Screen-Space ')).toBe('screen-space');
    // The deleted volumetric primitive must be UNRECOGNISED, not silently
    // mapped to something else (#1352 deletion).
    expect(parseLinePrimitive('volumetric')).toBeNull();
    expect(parseLinePrimitive('quads')).toBeNull();
    expect(parseLinePrimitive('')).toBeNull();
    expect(parseLinePrimitive(null)).toBeNull();
    expect(parseLinePrimitive(undefined)).toBeNull();
  });

  it('resolves explicit > session override > default', () => {
    expect(DEFAULT_LINE_PRIMITIVE).toBe('capsule');
    expect(resolveLinePrimitive()).toBe('capsule');
    setLinePrimitiveOverride('screen-space');
    expect(resolveLinePrimitive()).toBe('screen-space');
    // Explicit (harness) argument bypasses the session override.
    expect(resolveLinePrimitive('capsule')).toBe('capsule');
    setLinePrimitiveOverride(null);
    expect(resolveLinePrimitive()).toBe('capsule');
  });

  it('keeps the allowed-values list and the type in sync', () => {
    expect(LINE_PRIMITIVES).toEqual(['screen-space', 'capsule']);
  });
});

describe('LineMaterial primitive selection (GLSL)', () => {
  it('defaults to the CAPSULE pair (the #1352 flip)', () => {
    const m = new LineMaterial();
    expect(m.vertexShader).toBe(CAPSULE_LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(CAPSULE_LINE_FRAGMENT_SHADER);
    m.dispose();
  });

  it('the screen-space pair is still selectable explicitly', () => {
    const m = new LineMaterial({ primitive: 'screen-space' });
    expect(m.vertexShader).toBe(LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(LINE_FRAGMENT_SHADER);
    m.dispose();
  });

  it('selects the capsule pair for an explicit primitive', () => {
    const m = new LineMaterial({ primitive: 'capsule' });
    expect(m.vertexShader).toBe(CAPSULE_LINE_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(CAPSULE_LINE_FRAGMENT_SHADER);
    expect(m.userData.linePrimitive).toBe('capsule');
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
    setLinePrimitiveOverride('screen-space');
    const m = new LineMaterial();
    expect(m.vertexShader).toBe(LINE_VERTEX_SHADER);
    m.dispose();
    setLinePrimitiveOverride('capsule');
    const c = new LineMaterial();
    expect(c.vertexShader).toBe(CAPSULE_LINE_VERTEX_SHADER);
    c.dispose();
  });

  it('TSL wrapper round-trips the primitive through clone()', async () => {
    const { LineTSLMaterial } = await import('../../../../rendering/materials/line/material-tsl');
    const cap = new LineTSLMaterial({ primitive: 'capsule', blendingMode: 'max' });
    expect(cap.clone().userData.linePrimitive).toBe('capsule');
    const quad = new LineTSLMaterial({ primitive: 'screen-space' });
    expect(quad.clone().userData.linePrimitive).toBe('screen-space');
    cap.dispose();
    quad.dispose();
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

  it('the screen-space pick pair is still selectable explicitly', async () => {
    const { LinePickingMaterial } = await import('../../../../rendering/picking/line/material');
    const { LINE_PICK_VERTEX_SHADER, LINE_PICK_FRAGMENT_SHADER } =
      await import('../../../../rendering/picking/line/shaders');
    const m = new LinePickingMaterial({ nodeId: 7, primitive: 'screen-space' });
    expect(m.vertexShader).toBe(LINE_PICK_VERTEX_SHADER);
    expect(m.fragmentShader).toBe(LINE_PICK_FRAGMENT_SHADER);
    m.dispose();
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
    const { LINE_PICK_VERTEX_SHADER } = await import('../../../../rendering/picking/line/shaders');
    const m = new LinePickingMaterial({ nodeId: 7, primitive: 'screen-space' });
    const c = m.clone();
    expect(c.vertexShader).toBe(LINE_PICK_VERTEX_SHADER);
    expect(c.userData.linePrimitive).toBe('screen-space');
    m.dispose();
    c.dispose();
  });

  it('TSL wrapper dispatches on the primitive (and rebuilds through the same factory)', async () => {
    const { LinePickingTSLMaterial } =
      await import('../../../../rendering/picking/line/material-tsl');
    const quadFactory = vi.mocked(
      (await import('../../../../rendering/picking/line/pick.tsl')).linePickWebGPUFactory
    );
    const capFactory = vi.mocked(
      (await import('../../../../rendering/picking/line/pick-capsule.tsl'))
        .capsuleLinePickWebGPUFactory
    );
    capFactory.mockClear();
    quadFactory.mockClear();

    const cap = new LinePickingTSLMaterial({ nodeId: 7, primitive: 'capsule' });
    expect(capFactory).toHaveBeenCalledTimes(1);
    expect(quadFactory).not.toHaveBeenCalled();
    // A texture rebind rebuilds through the SAME factory.
    const tex = new (await import('three')).DataTexture(
      new Float32Array(24),
      6,
      1,
      (await import('three')).RGBAFormat,
      (await import('three')).FloatType
    );
    cap.updateLineTexture(tex);
    expect(capFactory).toHaveBeenCalledTimes(2);
    expect(quadFactory).not.toHaveBeenCalled();
    // clone carries the variant through the constructor.
    const c = cap.clone();
    expect(c.userData.linePrimitive).toBe('capsule');
    expect(quadFactory).not.toHaveBeenCalled();

    capFactory.mockClear();
    const quad = new LinePickingTSLMaterial({ nodeId: 7, primitive: 'screen-space' });
    expect(quadFactory).toHaveBeenCalledTimes(1);
    expect(capFactory).not.toHaveBeenCalled();

    cap.dispose();
    c.dispose();
    quad.dispose();
    tex.dispose();
  });
});
