/**
 * Mesh material define/state lifecycle, on BOTH backends.
 *
 * The mesh emission is a THREE-way choice (cutout / premultiplied / alpha-weighted)
 * where the siblings have two, so both mode defines have to be maintained — and the
 * interesting failures are all "a define survived a mode it does not belong to". An
 * `opaque → additive` switch that stranded `LUXAR_MESH_ALPHA_CUTOUT` would keep
 * discarding fragments in a mode with no cutout, which reads as "translucent mesh
 * randomly loses parts of itself".
 *
 * Every case runs against the GLSL wrapper and the TSL wrapper from one table: the
 * two classes are a drop-in pair for `MaterialManager`, so a divergence in their
 * define bookkeeping is a backend-dependent bug by construction.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { MeshMaterial } from '../../../../../rendering/materials/mesh/material-glsl';
import { MeshTSLMaterial } from '../../../../../rendering/materials/mesh/material-tsl';
import type { MeshMaterialConfig } from '../../../../../rendering/materials/mesh/material-glsl';
import { MESH_DEFAULTS } from '../../../../../rendering/materials/mesh/appearance';
import { BLENDING_MODES, type BlendingMode } from '../../../../../types/blending';
import { isLuxarMaterial, type LuxarMaterial } from '../../../../../ui/layers/luxar-material';

type AnyMeshMaterial = MeshMaterial | MeshTSLMaterial;

const BACKENDS: Array<[string, (c?: MeshMaterialConfig) => AnyMeshMaterial]> = [
  ['glsl', (c) => new MeshMaterial(c)],
  ['tsl', (c) => new MeshTSLMaterial(c)],
];

const CUTOUT = 'LUXAR_MESH_ALPHA_CUTOUT';
const MAX_RGB = 'LUXAR_MAX_RGB_CONTRIBUTION';
const FLAT = 'LUXAR_MESH_FLAT_NORMAL';

const has = (m: AnyMeshMaterial, flag: string): boolean => !!m.defines && flag in m.defines;

describe.each(BACKENDS)('MeshMaterial (%s) — construction defaults', (_label, make) => {
  it("defaults to 'opaque', not the siblings' 'additive'", () => {
    // The one deliberate default asymmetry across the four geometry types: `opaque`
    // is the only mode unconditionally correct without per-triangle depth sorting,
    // and it is what a surface should look like (spec §6.3).
    const m = make();
    expect(m.userData.blendingMode).toBe('opaque');
    expect(m.depthWrite).toBe(true);
    expect(m.depthTest).toBe(true);
    expect(m.transparent).toBe(false);
    expect(has(m, CUTOUT)).toBe(true);
    expect(has(m, MAX_RGB)).toBe(false);
  });

  it('seeds the shade + cutout uniforms from MESH_DEFAULTS', () => {
    const m = make();
    expect(m.uniforms.uAmbient.value).toBe(MESH_DEFAULTS.ambient);
    expect(m.uniforms.uShadeExponent.value).toBe(MESH_DEFAULTS.shadeExponent);
    expect(m.uniforms.uAlphaCutoff.value).toBe(MESH_DEFAULTS.alphaCutoff);
  });

  it('starts on FrontSide, leaving `side` to the epoch', () => {
    // `side` is per-`displayDims`-epoch state (`applyMeshSide`), not config: an
    // undecidable winding frame forces DoubleSide regardless of `double_sided`. Both
    // backends must agree on the pre-commit value or the first frame differs.
    expect(make().side).toBe(THREE.FrontSide);
  });

  it('carries no absorption uniform — volumetric is not a mesh mode', () => {
    // Asserted rather than assumed: the sibling configs all have one, so its absence
    // is the kind of thing a well-meaning symmetry sweep would "fix".
    expect(make().uniforms.uAbsorption).toBeUndefined();
  });
});

describe.each(BACKENDS)('MeshMaterial (%s) — mode transitions', (_label, make) => {
  it('opaque → additive CLEARS the cutout define', () => {
    const m = make();
    expect(has(m, CUTOUT)).toBe(true);
    m.applyBlendingMode('additive');
    expect(has(m, CUTOUT)).toBe(false);
    expect(has(m, MAX_RGB)).toBe(false);
    expect(m.depthWrite).toBe(false);
    expect(m.transparent).toBe(true);
  });

  it('additive → max sets the premultiply define and the Max blend state', () => {
    const m = make({ blendingMode: 'additive' });
    m.applyBlendingMode('max');
    expect(has(m, MAX_RGB)).toBe(true);
    expect(has(m, CUTOUT)).toBe(false);
    expect(m.blending).toBe(THREE.CustomBlending);
    expect(m.blendEquation).toBe(THREE.MaxEquation);
    expect(m.blendSrc).toBe(THREE.OneFactor);
    expect(m.blendDst).toBe(THREE.OneFactor);
  });

  it('max → opaque clears the premultiply define and sets the cutout', () => {
    const m = make({ blendingMode: 'max' });
    expect(has(m, MAX_RGB)).toBe(true);
    m.applyBlendingMode('opaque');
    expect(has(m, MAX_RGB)).toBe(false);
    expect(has(m, CUTOUT)).toBe(true);
    expect(m.depthWrite).toBe(true);
  });

  it('never leaves BOTH emission defines set, for any mode in the union', () => {
    // The invariant behind all three cases above, stated once over the whole mode
    // set: the fragment has exactly one emission, so two defines means one branch is
    // dead and which one wins depends on preprocessor order.
    const m = make();
    for (const mode of BLENDING_MODES) {
      m.applyBlendingMode(mode);
      expect(has(m, CUTOUT) && has(m, MAX_RGB), `both defines set in ${mode}`).toBe(false);
    }
  });

  it('degrades volumetric to opaque at construction AND at runtime', () => {
    // It can arrive either way: composed from an ancestor group at load, or picked in
    // the layers panel (whose dropdown is shared across geometry types).
    const built = make({ blendingMode: 'volumetric' });
    expect(built.userData.blendingMode).toBe('opaque');
    expect(has(built, CUTOUT)).toBe(true);

    const switched = make({ blendingMode: 'additive' });
    switched.applyBlendingMode('volumetric');
    expect(switched.userData.blendingMode).toBe('opaque');
    expect(has(switched, CUTOUT)).toBe(true);
    expect(switched.depthWrite).toBe(true);
  });
});

describe.each(BACKENDS)('MeshMaterial (%s) — the shading variant', (_label, make) => {
  it('is a define, set from config', () => {
    expect(has(make(), FLAT)).toBe(false);
    expect(has(make({ flatNormal: true }), FLAT)).toBe(true);
  });

  it('toggles both ways through updateFlatNormal', () => {
    const m = make();
    m.updateFlatNormal(true);
    expect(has(m, FLAT)).toBe(true);
    m.updateFlatNormal(false);
    expect(has(m, FLAT)).toBe(false);
  });

  it('is a NO-OP when already in the wanted state', () => {
    // Load-bearing, not tidiness: the commit calls this on EVERY epoch, and a flip
    // recompiles the program (GLSL) or rebuilds the graph (TSL). An unguarded write
    // would pay that on every slice move of every mesh.
    //
    // Observed through `material.version`, not `needsUpdate`: the latter is a
    // WRITE-ONLY setter in three (it has no getter and just bumps `version`), so
    // reading it back yields `undefined` for both a recompile and a no-op.
    const m = make();
    const before = m.version;
    m.updateFlatNormal(false); // already false
    expect(m.version, 'a redundant call must not bump the program version').toBe(before);
    m.updateFlatNormal(true); // a real flip
    expect(m.version).toBeGreaterThan(before);
  });

  it('survives clone()', () => {
    // The layers panel clones on first interaction. A clone that lost the flag would
    // switch a flat-shaded node to smooth and read an unbound `normal` attribute —
    // (0,0,0) — shading the whole surface flat at uAmbient.
    const m = make({ flatNormal: true, blendingMode: 'max' });
    const c = m.clone();
    expect(has(c, FLAT)).toBe(true);
    expect(c.userData.blendingMode).toBe('max');
    expect(has(c, MAX_RGB)).toBe(true);
  });
});

describe.each(BACKENDS)('MeshMaterial (%s) — clone fidelity', (_label, make) => {
  it('round-trips the shade + cutout uniforms and the live `side`', () => {
    const m = make({ ambient: 0.75, shadeExponent: 3, alphaCutoff: 0.125, gamma: 2.2 });
    // `side` is epoch state the commit owns, so it must be copied rather than
    // re-derived: a clone taken while an undecidable frame forced DoubleSide has to
    // keep drawing both faces until the next commit re-applies it.
    m.side = THREE.DoubleSide;
    const c = m.clone();
    expect(c.uniforms.uAmbient.value).toBe(0.75);
    expect(c.uniforms.uShadeExponent.value).toBe(3);
    expect(c.uniforms.uAlphaCutoff.value).toBe(0.125);
    expect(c.userData.gamma).toBe(2.2);
    expect(c.side).toBe(THREE.DoubleSide);
  });
});

describe.each(BACKENDS)('MeshMaterial (%s) — the GOG / gamma fast paths', (_label, make) => {
  it('sets LUXAR_GAMMA_ONE at gamma 1 and clears it away from 1', () => {
    const m = make({ gamma: 1.0 });
    expect(has(m, 'LUXAR_GAMMA_ONE')).toBe(true);
    m.updateGamma(2.2);
    expect(has(m, 'LUXAR_GAMMA_ONE')).toBe(false);
    m.updateGamma(1.0);
    expect(has(m, 'LUXAR_GAMMA_ONE')).toBe(true);
  });

  it('sets LUXAR_NO_GOG only while intensity == 1 && offset == 0', () => {
    const m = make();
    expect(has(m, 'LUXAR_NO_GOG')).toBe(true);
    m.updateIntensity(2);
    expect(has(m, 'LUXAR_NO_GOG')).toBe(false);
    m.updateIntensity(1);
    expect(has(m, 'LUXAR_NO_GOG')).toBe(true);
    m.updateOffset(0.1);
    expect(has(m, 'LUXAR_NO_GOG')).toBe(false);
  });
});

describe('the two backends agree on every mode', () => {
  it.each(BLENDING_MODES)('%s produces identical state and defines', (mode: BlendingMode) => {
    const g = new MeshMaterial({ blendingMode: mode });
    const t = new MeshTSLMaterial({ blendingMode: mode });
    expect(t.userData.blendingMode).toBe(g.userData.blendingMode);
    expect(t.depthTest).toBe(g.depthTest);
    expect(t.depthWrite).toBe(g.depthWrite);
    expect(t.transparent).toBe(g.transparent);
    expect(t.blending).toBe(g.blending);
    expect(Object.keys(t.defines ?? {}).sort()).toEqual(Object.keys(g.defines ?? {}).sort());
  });
});

describe('the layers-panel contract', () => {
  it('is satisfied by both backends, at compile time AND by the runtime guard', () => {
    // Compile-time: these annotations are the assertion. `LuxarMaterial` used to
    // extend `CameraAwareMaterial`, which made a mesh material — a perfectly valid
    // leaf material with the full layer-control surface — unrepresentable in the
    // panel, and would have pushed whoever wires the mesh row toward adding an empty
    // `updateCameraParams` (a lie, and a per-frame cost per node). The panel never
    // called that method and its own `isLuxarMaterial` guard never checked for it, so
    // the requirement came off the interface instead.
    const glsl: LuxarMaterial = new MeshMaterial();
    const tsl: LuxarMaterial = new MeshTSLMaterial();

    // Runtime: the guard that actually produces the type at the call site.
    expect(isLuxarMaterial(glsl)).toBe(true);
    expect(isLuxarMaterial(tsl)).toBe(true);

    // And the whole surface the panel drives is really there.
    for (const m of [glsl, tsl]) {
      for (const method of [
        'updateIntensity',
        'updateOffset',
        'updateGamma',
        'updateOpacity',
        'updateColormapTexture',
        'updateScalarRange',
        'applyBlendingMode',
      ] as const) {
        expect(typeof (m as unknown as Record<string, unknown>)[method], method).toBe('function');
      }
    }
  });
});

describe('the two backends agree across the FULL flag cross-product', () => {
  // The codegen snapshots pin five hand-picked variants. This sweeps every
  // combination of the four independent config flags against every blending mode,
  // because the divergences that matter are in combinations nobody thought to
  // snapshot (colormap + max, flat + colormap, gammaOne + colormap — where the
  // colormap path owns gamma pre-LUT and the fragment must NOT apply it again).
  const flag = [false, true];
  const cases: MeshMaterialConfig[] = [];
  for (const blendingMode of BLENDING_MODES) {
    for (const colormap of flag) {
      for (const flatNormal of flag) {
        for (const gammaOne of flag) {
          for (const noGOG of flag) {
            cases.push({
              blendingMode,
              flatNormal,
              gamma: gammaOne ? 1.0 : 2.2,
              intensity: noGOG ? 1.0 : 3.0,
              offset: noGOG ? 0.0 : 0.1,
              colormapTexture: colormap ? new THREE.DataTexture() : undefined,
            });
          }
        }
      }
    }
  }

  it('produces identical defines, blend state and shade uniforms for every configuration', () => {
    expect(cases).toHaveLength(BLENDING_MODES.length * 16);
    const divergences: string[] = [];
    for (const config of cases) {
      const g = new MeshMaterial(config);
      const t = new MeshTSLMaterial(config);
      const label =
        `${config.blendingMode} colormap=${!!config.colormapTexture} ` +
        `flat=${config.flatNormal} gamma=${config.gamma} intensity=${config.intensity}`;
      const gd = Object.keys(g.defines ?? {}).sort();
      const td = Object.keys(t.defines ?? {}).sort();
      if (JSON.stringify(gd) !== JSON.stringify(td)) {
        divergences.push(`${label}: defines glsl=${JSON.stringify(gd)} tsl=${JSON.stringify(td)}`);
      }
      for (const [prop, gv, tv] of [
        ['depthTest', g.depthTest, t.depthTest],
        ['depthWrite', g.depthWrite, t.depthWrite],
        ['transparent', g.transparent, t.transparent],
        ['blending', g.blending, t.blending],
        ['side', g.side, t.side],
        ['blendingMode', g.userData.blendingMode, t.userData.blendingMode],
      ] as const) {
        if (gv !== tv) divergences.push(`${label}: ${prop} glsl=${String(gv)} tsl=${String(tv)}`);
      }
      // The shade uniforms must seed identically too — including through the exponent
      // clamp, the one place a value is transformed on the way in.
      for (const u of ['uAmbient', 'uShadeExponent', 'uAlphaCutoff', 'uInvGamma'] as const) {
        if (g.uniforms[u].value !== t.uniforms[u].value) {
          divergences.push(
            `${label}: ${u} glsl=${String(g.uniforms[u].value)} tsl=${String(t.uniforms[u].value)}`
          );
        }
      }
    }
    expect(divergences, `backend divergences:\n${divergences.join('\n')}`).toEqual([]);
  });
});
