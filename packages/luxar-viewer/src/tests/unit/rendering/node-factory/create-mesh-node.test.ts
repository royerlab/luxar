/**
 * Mesh node construction: the shading-variant decision, the attribute set the
 * placeholder is born with, and the viewer-side blending-mode defaults.
 *
 * The three things worth pinning here are all cases where the WRONG answer renders
 * something plausible rather than failing: a mesh shaded from an unbound `normal`
 * attribute is uniformly dark, a `volumetric` mesh silently has no fragment branch,
 * and an attribute added to a live geometry breaks only on the WebGPU backend.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  createEmptyMeshNode,
  createMeshMaterial,
  resolveFlatNormal,
  applyMeshShading,
  applyMeshSide,
} from '../../../../rendering/node-factory/create-mesh-node';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { MESH_DEFAULTS } from '../../../../rendering/materials/mesh/appearance';
import type { MeshDataLoader, MeshMetadata } from '../../../../types/mesh';
import { log } from '../../../../utils/log';

const ATTRS: MeshMetadata = {
  type: 'mesh',
  n_vertices: 3,
  n_faces: 1,
  ndim: 3,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  shading: 'flat',
  double_sided: true,
  ordering: 'none',
};

const loader = {} as MeshDataLoader;

const hasFlat = (m: THREE.Material): boolean =>
  !!(m as MeshMaterial).defines && 'LUXAR_MESH_FLAT_NORMAL' in (m as MeshMaterial).defines;

describe('resolveFlatNormal — the §3.4 rule in one place', () => {
  it.each([
    // shading,   has_normals, usable, → flat?
    ['flat', true, true, true, 'an explicit flat overrides valid stored normals'],
    ['smooth', true, true, false, 'the only case that reads the stored normals'],
    ['smooth', false, false, true, 'no normals to read'],
    ['smooth', true, false, true, 'normals authored for other axes'],
    ['flat', false, false, true, 'flat and normal-less agree'],
  ] as const)(
    'shading=%s has_normals=%s usable=%s → flat=%s (%s)',
    (shading, has_normals, usable, expected, _why) => {
      expect(resolveFlatNormal({ ...ATTRS, shading, has_normals } as MeshMetadata, usable)).toBe(
        expected
      );
    }
  );
});

describe('createEmptyMeshNode — the attribute set is complete from birth', () => {
  it('binds normal + aScalar STUBS when the metadata declares them', () => {
    // Not cosmetic: the attribute set is baked into the WebGPU vertex layout at first
    // draw and never rebuilt, so an attribute that appears on the first commit
    // instead renders the node black on that backend. The stubs are 1-vertex, so the
    // first commit only replaces contents.
    const node = createEmptyMeshNode(
      '/surface',
      { ...ATTRS, has_normals: true, has_scalars: true, colormap: 'viridis' },
      loader,
      null
    );
    expect(node.geometry.getAttribute('normal')).toBeDefined();
    expect(node.geometry.getAttribute('aScalar')).toBeDefined();
    expect(node.geometry.getAttribute('normal').count).toBe(1);
    // The scalar-presence stamp `supportsScalarColormap('mesh', …)` fails closed on.
    expect(node.geometry.userData.hasScalars).toBe(true);
  });

  it('binds NEITHER when the metadata declares neither', () => {
    // The other half of the same invariant — an unconditional bind would upload
    // V*12 + V*4 bytes of zeros for every mesh that has no normals or scalars, and
    // would make the smooth variant read (0,0,0) as though it were data.
    const node = createEmptyMeshNode('/surface', ATTRS, loader, null);
    expect(node.geometry.getAttribute('normal')).toBeUndefined();
    expect(node.geometry.getAttribute('aScalar')).toBeUndefined();
    expect(node.geometry.userData.hasScalars).toBeUndefined();
  });

  it('always binds `color`, so a bare add_mesh is white and not GL-default black', () => {
    const node = createEmptyMeshNode('/surface', ATTRS, loader, null);
    const color = node.geometry.getAttribute('color');
    expect(color).toBeDefined();
    expect(Array.from(color.array as Float32Array)).toEqual([1, 1, 1]);
  });

  it('marks the material node-owned so the layers panel mutates it in place', () => {
    expect(createEmptyMeshNode('/surface', ATTRS, loader, null).userData._layerMaterialCloned).toBe(
      true
    );
  });

  it("starts on the AUTHORED side, which the first commit's epoch may override", () => {
    expect(
      (
        createEmptyMeshNode('/s', { ...ATTRS, double_sided: true }, loader, null)
          .material as THREE.Material
      ).side
    ).toBe(THREE.DoubleSide);
    expect(
      (
        createEmptyMeshNode('/s', { ...ATTRS, double_sided: false }, loader, null)
          .material as THREE.Material
      ).side
    ).toBe(THREE.FrontSide);
  });

  it('opens on the SMOOTH variant for a normal-bearing smooth mesh', () => {
    // The view-independent half of the rule is all that is knowable pre-fetch, and
    // guessing smooth is right for the common case (normals authored for the axes the
    // scene opens on). `applyMeshShading` corrects it on the first commit otherwise.
    const smooth = createEmptyMeshNode(
      '/s',
      { ...ATTRS, shading: 'smooth', has_normals: true, normal_dims: [0, 1, 2] },
      loader,
      null
    );
    expect(hasFlat(smooth.material as THREE.Material)).toBe(false);
    const flat = createEmptyMeshNode('/s', ATTRS, loader, null);
    expect(hasFlat(flat.material as THREE.Material)).toBe(true);
  });
});

describe('applyMeshShading — the per-epoch correction', () => {
  const smoothAttrs: MeshMetadata = {
    ...ATTRS,
    shading: 'smooth',
    has_normals: true,
    normal_dims: [0, 1, 2],
  };

  it('flips a smooth node to flat when the epoch frame stops matching, and back', () => {
    const node = createEmptyMeshNode('/s', smoothAttrs, loader, null);
    expect(hasFlat(node.material as THREE.Material)).toBe(false);

    applyMeshShading(node, smoothAttrs, false); // displayDims moved off the frame
    expect(hasFlat(node.material as THREE.Material)).toBe(true);

    applyMeshShading(node, smoothAttrs, true); // and back
    expect(hasFlat(node.material as THREE.Material)).toBe(false);
  });

  it('never flips an explicitly FLAT node, whatever the projection reports', () => {
    // `shading` is view-independent, so a flat node is statically the flat variant.
    const node = createEmptyMeshNode('/s', ATTRS, loader, null);
    applyMeshShading(node, ATTRS, true);
    expect(hasFlat(node.material as THREE.Material)).toBe(true);
  });

  it('costs no program recompile when the variant is unchanged', () => {
    // It runs on EVERY commit, including every slice move.
    const node = createEmptyMeshNode('/s', smoothAttrs, loader, null);
    const material = node.material as THREE.Material;
    const before = material.version;
    applyMeshShading(node, smoothAttrs, true);
    applyMeshShading(node, smoothAttrs, true);
    expect(material.version).toBe(before);
  });
});

describe('applyMeshSide', () => {
  it('only touches the material when the side actually changes', () => {
    const node = createEmptyMeshNode('/s', { ...ATTRS, double_sided: false }, loader, null);
    const material = node.material as THREE.Material;
    const before = material.version;
    applyMeshSide(node, 'front');
    expect(material.version).toBe(before);
    applyMeshSide(node, 'double');
    expect(material.side).toBe(THREE.DoubleSide);
    expect(material.version).toBeGreaterThan(before);
  });
});

describe('createMeshMaterial — the viewer-side mode defaults', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("defaults to 'opaque' when the attrs name no mode", () => {
    expect(createMeshMaterial(ATTRS, false).userData.blendingMode).toBe('opaque');
  });

  it('honours an INHERITED mode rather than overriding it', () => {
    // The composed attrs are what arrive here, so an ancestor group's `additive`
    // must survive — this is why the default lives in the viewer and is never
    // stamped into the node by the writer.
    expect(
      createMeshMaterial({ ...ATTRS, blending_mode: 'additive' } as MeshMetadata, false).userData
        .blendingMode
    ).toBe('additive');
  });

  it('warns ONCE, naming the node, when volumetric is inherited — then falls back', () => {
    const material = createMeshMaterial(
      { ...ATTRS, blending_mode: 'volumetric' } as MeshMetadata,
      false,
      undefined,
      '/organ/surface'
    );
    expect(material.userData.blendingMode).toBe('opaque');
    expect(warn).toHaveBeenCalledTimes(1);
    const message = String(warn.mock.calls[0][1]);
    expect(message).toContain('/organ/surface');
    expect(message).toContain('volumetric');
    // A warning, not a throw: the mode can come from an ancestor the mesh knows
    // nothing about, so refusing would let an unrelated group setting break it.
  });

  it('suppresses a colormap when the geometry carries no scalars, and says so', () => {
    const geometry = new THREE.BufferGeometry();
    const material = createMeshMaterial(
      { ...ATTRS, has_scalars: true, colormap: 'viridis' } as MeshMetadata,
      false,
      geometry,
      '/surface'
    );
    expect(material.defines?.USE_COLORMAP).toBeUndefined();
    expect(String(warn.mock.calls[0][1])).toContain('Colormap suppressed');
  });

  it('re-expresses an authored gain as the scalar WINDOW, not a post-LUT gain (#936)', () => {
    // The double-apply rule the three siblings already encode: once the colormap
    // takes over, `intensity`/`offset` mean the LUT window, so the post-LUT GOG must
    // be reset to identity or the gain lands twice.
    const geometry = new THREE.BufferGeometry();
    geometry.userData.hasScalars = true;
    const material = createMeshMaterial(
      {
        ...ATTRS,
        has_scalars: true,
        colormap: 'viridis',
        intensity: 4,
        offset: 0,
        scalar_data_range: [0, 1],
      } as MeshMetadata,
      false,
      geometry,
      '/surface'
    );
    expect(material.defines?.USE_COLORMAP).toBe('');
    expect(material.uniforms.uIntensity.value).toBe(1);
    expect(material.uniforms.uOffset.value).toBe(0);
    // The gain moved into the window instead of vanishing.
    expect(material.uniforms.uScalarScale.value).not.toBe(1);
  });
});

describe('createMeshMaterial — authored shade knobs (§6.2)', () => {
  it('reads ambient / shade_exponent / alpha_cutoff from the composed attrs', () => {
    // These ride in through `add_mesh(**attrs)` (the writer never stamps them), so
    // they were ALREADY reachable in the composed attrs and were being silently
    // dropped. An authored value that does nothing is worse than one that is refused.
    const m = createMeshMaterial(
      { ...ATTRS, ambient: 0.75, shade_exponent: 4, alpha_cutoff: 0.125 } as MeshMetadata,
      false
    );
    expect(m.uniforms.uAmbient.value).toBe(0.75);
    expect(m.uniforms.uShadeExponent.value).toBe(4);
    expect(m.uniforms.uAlphaCutoff.value).toBe(0.125);
  });

  it('clamps them, because they are FRACTIONS rather than gains', () => {
    // `ambient` is the shade floor and `alpha_cutoff` is compared against a coverage
    // already in [0, 1], so out-of-range values are meaningless, not merely odd:
    // ambient = 1e9 multiplies the surface to white and cutoff = 1e9 discards every
    // fragment, i.e. the mesh vanishes with no diagnostic. Both are author-reachable.
    const hot = createMeshMaterial(
      { ...ATTRS, ambient: 1e9, alpha_cutoff: 1e9, shade_exponent: 0 } as MeshMetadata,
      false
    );
    expect(hot.uniforms.uAmbient.value).toBe(1);
    expect(hot.uniforms.uAlphaCutoff.value).toBe(1);
    // exponent 0 would make `pow(0, 0)` — undefined GLSL — at any face-away fragment.
    expect(hot.uniforms.uShadeExponent.value).toBeGreaterThan(0);

    const cold = createMeshMaterial(
      { ...ATTRS, ambient: -5, alpha_cutoff: -5 } as MeshMetadata,
      false
    );
    expect(cold.uniforms.uAmbient.value).toBe(0);
    expect(cold.uniforms.uAlphaCutoff.value).toBe(0);
  });

  it('routes NaN/Inf to the documented default rather than a range boundary', () => {
    // Matching the sibling shaders' sanitizer policy: corruption resolves loudly to
    // the default, not to 0 or 1 — which would look like a deliberate setting.
    const m = createMeshMaterial(
      { ...ATTRS, ambient: NaN, alpha_cutoff: Infinity } as MeshMetadata,
      false
    );
    expect(m.uniforms.uAmbient.value).toBe(MESH_DEFAULTS.ambient);
    expect(m.uniforms.uAlphaCutoff.value).toBe(MESH_DEFAULTS.alphaCutoff);
  });
});
