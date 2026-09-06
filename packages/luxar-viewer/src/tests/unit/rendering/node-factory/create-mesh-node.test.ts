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
  resolveMeshShading,
  applyMeshShading,
  applyMeshSide,
  applyMeshTexture,
  applyMeshVertexAlpha,
} from '../../../../rendering/node-factory/create-mesh-node';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { PhysicalMeshMaterial } from '../../../../rendering/materials/mesh-physical/material-glsl';
import { isPhysicalMeshMaterial } from '../../../../rendering/materials/mesh-physical/config';
import { MeshPickingMaterial } from '../../../../rendering/picking/mesh/material';
import { materialManager } from '../../../../rendering/material-manager';
import { MESH_DEFAULTS } from '../../../../rendering/materials/mesh/appearance';
import type { MeshDataLoader, MeshMetadata } from '../../../../types/mesh';
import { applyEffectiveAttrs } from '../../../../data/scene-loader/view-state/effective-attrs';
import type { SceneNode } from '../../../../data/data-loader-types';
import { log } from '../../../../utils/log';
import type { PickingSystem } from '../../../../rendering/picking/picking-system';

const ATTRS: MeshMetadata = {
  type: 'mesh',
  n_vertices: 3,
  n_faces: 1,
  ndim: 3,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  has_uvs: false,
  has_texture: false,
  shading: 'flat',
  double_sided: true,
  ordering: 'none',
};

const loader = {} as MeshDataLoader;

const hasFlat = (m: THREE.Material): boolean =>
  !!(m as MeshMaterial).defines && 'LUXAR_MESH_FLAT_NORMAL' in (m as MeshMaterial).defines;

describe('resolveMeshShading — the §3.4 rule in one place', () => {
  it.each([
    // shading,   has_normals, usable, → resolved
    ['flat', true, true, 'flat', 'an explicit flat overrides valid stored normals'],
    ['smooth', true, true, 'smooth', 'the only case that reads the stored normals'],
    ['smooth', false, false, 'flat', 'no normals to read'],
    ['smooth', true, false, 'flat', 'normals authored for other axes'],
    ['flat', false, false, 'flat', 'flat and normal-less agree'],
    // The unlit arm short-circuits AHEAD of every stored-normal question, and both
    // of these pin that ordering rather than just the value. A `none` node with
    // perfectly valid stored normals must still resolve to `none` — falling through
    // would compute a normal the shader then discards — and a `none` node with
    // UNUSABLE normals must not be rescued into `flat`, which is what a rule that
    // tested usability first would do.
    ['none', true, true, 'none', 'unlit wins over valid stored normals'],
    ['none', true, false, 'none', 'unlit is not view-dependent, so a frame mismatch is irrelevant'],
    ['none', false, false, 'none', 'unlit needs no normals in the first place'],
  ] as const)(
    'shading=%s has_normals=%s usable=%s → %s (%s)',
    (shading, has_normals, usable, expected, _why) => {
      expect(resolveMeshShading({ ...ATTRS, shading, has_normals } as MeshMetadata, usable)).toBe(
        expected
      );
    }
  );
});

describe('createEmptyMeshNode — the attribute set is complete from birth', () => {
  it('stamps an ancestor-composed layer order', () => {
    const node = createEmptyMeshNode(
      '/surface',
      { ...ATTRS, layer_order: 7 } as unknown as MeshMetadata,
      loader,
      null
    );
    expect(node.userData.layerOrder).toBe(7);
  });

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

describe('applyMeshTexture', () => {
  it('disposes both blank shader placeholders when the real image arrives', () => {
    const attrs = {
      ...ATTRS,
      has_uvs: true,
      has_texture: true,
      texture_width: 1,
      texture_height: 1,
      texture_channels: 3,
      texture_color_space: 'srgb',
    } as MeshMetadata;
    const pickingSystem = {
      allocatePickId: () => 1,
      registerNode: (main: THREE.Object3D, pick: THREE.Object3D) => {
        main.userData.pickNode = pick;
      },
    } as unknown as PickingSystem;
    const node = createEmptyMeshNode('/surface', attrs, loader, pickingSystem);
    const placeholders = (node.userData as { meshTexturePlaceholders: THREE.Texture[] })
      .meshTexturePlaceholders;
    const disposals = placeholders.map((placeholder) => vi.spyOn(placeholder, 'dispose'));

    applyMeshTexture(node, attrs, {
      kind: 'raw',
      pixels: new Uint8Array([255, 0, 0]),
      width: 1,
      height: 1,
      channels: 3,
    });

    expect(disposals).toHaveLength(2);
    expect(disposals.every((dispose) => dispose.mock.calls.length === 1)).toBe(true);
    expect(node.userData.meshTexturePlaceholders).toBeUndefined();
  });

  it('disposes the uploaded GPU texture with the shared geometry', () => {
    const attrs = {
      ...ATTRS,
      has_uvs: true,
      has_texture: true,
      texture_width: 1,
      texture_height: 1,
      texture_channels: 3,
      texture_color_space: 'srgb',
    } as MeshMetadata;
    const node = createEmptyMeshNode('/surface', attrs, loader, null);
    applyMeshTexture(node, attrs, {
      kind: 'raw',
      pixels: new Uint8Array([255, 0, 0]),
      width: 1,
      height: 1,
      channels: 3,
    });
    const texture = (node.userData as { meshTexture: THREE.Texture }).meshTexture;
    const dispose = vi.spyOn(texture, 'dispose');
    node.geometry.dispose();
    expect(dispose).toHaveBeenCalledTimes(1);
  });
});

describe('createMeshMaterial — the viewer-side mode defaults', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("defaults to 'opaque' when the attrs name no mode", () => {
    expect(createMeshMaterial(ATTRS, 'smooth').userData.blendingMode).toBe('opaque');
  });

  it('honours an INHERITED mode rather than overriding it', () => {
    // The composed attrs are what arrive here, so an ancestor group's `additive`
    // must survive — this is why the default lives in the viewer and is never
    // stamped into the node by the writer.
    expect(
      createMeshMaterial({ ...ATTRS, blending_mode: 'additive' } as MeshMetadata, 'smooth').userData
        .blendingMode
    ).toBe('additive');
  });

  it('warns ONCE, naming the node, when volumetric is inherited — then falls back', () => {
    const material = createMeshMaterial(
      { ...ATTRS, blending_mode: 'volumetric' } as MeshMetadata,
      'smooth',
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
      'smooth',
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
      'smooth',
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

describe('createMeshMaterial — the §6.3 opaque default survives the composed load path (#1272)', () => {
  // The raw-attrs tests above hand ATTRS (no blending_mode) straight to the
  // factory, so its `?? 'opaque'` fallback fires. The REAL load path first
  // runs the node through `applyEffectiveAttrs`, which used to stamp the
  // composed default 'additive' onto an unset chain — killing the fallback and
  // rendering every default mesh additive. Compose an actual scene graph to
  // pin the whole path, not just the leaf factory.
  it("keeps 'opaque' when no ancestor sets a mode", () => {
    const meshLeaf: SceneNode = {
      path: 'surface',
      type: 'mesh',
      attrs: { ...ATTRS, opacity: 1.0 },
      hasSpatialIndex: false,
    };
    const root: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [meshLeaf],
    };
    const composed = applyEffectiveAttrs(root, meshLeaf) as unknown as MeshMetadata;
    expect(createMeshMaterial(composed, 'smooth').userData.blendingMode).toBe('opaque');
  });

  it('honours an inherited mode set by an ancestor group', () => {
    const meshLeaf: SceneNode = {
      path: 'grp/surface',
      type: 'mesh',
      attrs: { ...ATTRS, opacity: 1.0 },
      hasSpatialIndex: false,
    };
    const group: SceneNode = {
      path: 'grp',
      type: 'group',
      attrs: { blending_mode: 'additive' },
      hasSpatialIndex: false,
      children: [meshLeaf],
    };
    const root: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [group],
    };
    const composed = applyEffectiveAttrs(root, meshLeaf) as unknown as MeshMetadata;
    expect(createMeshMaterial(composed, 'smooth').userData.blendingMode).toBe('additive');
  });
});

describe('createMeshMaterial — authored shade knobs (§6.2)', () => {
  it('reads all mesh appearance attrs from the composed attrs', () => {
    // These ride in through `add_mesh(**attrs)` (the writer never stamps them).
    // Authoring support and the material reads landed together, so accepted values
    // must reach the uniforms rather than becoming dead metadata.
    const m = createMeshMaterial(
      {
        ...ATTRS,
        ambient: 0.75,
        shade_exponent: 4,
        specular: 0.3,
        shininess: 48,
        alpha_cutoff: 0.125,
      } as MeshMetadata,
      'smooth'
    );
    expect(m.uniforms.uAmbient.value).toBe(0.75);
    expect(m.uniforms.uShadeExponent.value).toBe(4);
    expect(m.uniforms.uSpecular.value).toBe(0.3);
    expect(m.uniforms.uShininess.value).toBe(48);
    expect(m.uniforms.uAlphaCutoff.value).toBe(0.125);
  });

  it('clamps them, because they are FRACTIONS rather than gains', () => {
    // `ambient` is the shade floor and `alpha_cutoff` is compared against a coverage
    // already in [0, 1], so out-of-range values are meaningless, not merely odd:
    // ambient = 1e9 multiplies the surface to white and cutoff = 1e9 discards every
    // fragment, i.e. the mesh vanishes with no diagnostic. Both are author-reachable.
    const hot = createMeshMaterial(
      {
        ...ATTRS,
        ambient: 1e9,
        specular: 1e9,
        alpha_cutoff: 1e9,
        shade_exponent: 0,
        shininess: 0,
      } as MeshMetadata,
      'smooth'
    );
    expect(hot.uniforms.uAmbient.value).toBe(1);
    expect(hot.uniforms.uSpecular.value).toBe(1);
    expect(hot.uniforms.uAlphaCutoff.value).toBe(1);
    // exponent 0 would make `pow(0, 0)` — undefined GLSL — at any face-away fragment.
    expect(hot.uniforms.uShadeExponent.value).toBeGreaterThan(0);
    expect(hot.uniforms.uShininess.value).toBeGreaterThan(0);

    const cold = createMeshMaterial(
      { ...ATTRS, ambient: -5, alpha_cutoff: -5 } as MeshMetadata,
      'smooth'
    );
    expect(cold.uniforms.uAmbient.value).toBe(0);
    expect(cold.uniforms.uAlphaCutoff.value).toBe(0);
  });

  it('routes NaN/Inf to the documented default rather than a range boundary', () => {
    // Matching the sibling shaders' sanitizer policy: corruption resolves loudly to
    // the default, not to 0 or 1 — which would look like a deliberate setting.
    const m = createMeshMaterial(
      { ...ATTRS, ambient: NaN, alpha_cutoff: Infinity } as MeshMetadata,
      'smooth'
    );
    expect(m.uniforms.uAmbient.value).toBe(MESH_DEFAULTS.ambient);
    expect(m.uniforms.uAlphaCutoff.value).toBe(MESH_DEFAULTS.alphaCutoff);
  });
});

describe("createEmptyMeshNode — the material='physical' family (MESH_PHYSICAL_MATERIALS_SPEC §3.2)", () => {
  const PHYSICAL: MeshMetadata = {
    ...ATTRS,
    material: 'physical',
    roughness: 0.4,
    metalness: 1.0,
    clearcoat: 1.0,
    clearcoat_roughness: 0.05,
    sheen: 0.5,
    sheen_color: '#ff4d6d',
  };

  it('builds three’s physical material, not the house shader, and maps the knobs', () => {
    const node = createEmptyMeshNode('/shell', PHYSICAL, loader, null);
    const m = node.material as PhysicalMeshMaterial;
    expect(m).toBeInstanceOf(PhysicalMeshMaterial);
    expect(m).not.toBeInstanceOf(MeshMaterial);
    expect(isPhysicalMeshMaterial(m)).toBe(true);
    expect(m.roughness).toBe(0.4);
    expect(m.metalness).toBe(1.0);
    expect(m.clearcoat).toBe(1.0);
    expect(m.clearcoatRoughness).toBe(0.05);
    expect(m.sheen).toBe(0.5);
    expect(m.vertexColors).toBe(true);
    // Everything shared with the house family still happens.
    expect(node.userData.nodeType).toBe('mesh');
    expect(node.userData._layerMaterialCloned).toBe(true);
    expect(node.geometry.getAttribute('color')).toBeDefined();
  });

  it('honours the authored side and maps flat/smooth onto flatShading', () => {
    const flat = createEmptyMeshNode('/s', { ...PHYSICAL, double_sided: false }, loader, null);
    expect((flat.material as THREE.Material).side).toBe(THREE.FrontSide);
    expect((flat.material as PhysicalMeshMaterial).flatShading).toBe(true);
    const smooth = createEmptyMeshNode(
      '/s',
      { ...PHYSICAL, shading: 'smooth', has_normals: true, normal_dims: [0, 1, 2] },
      loader,
      null
    );
    expect((smooth.material as PhysicalMeshMaterial).flatShading).toBe(false);
    expect((smooth.material as THREE.Material).side).toBe(THREE.DoubleSide);
  });

  it('does not throw on the texture-placeholder bookkeeping (no `uniforms` on this family)', () => {
    expect(() => createEmptyMeshNode('/shell', PHYSICAL, loader, null)).not.toThrow();
    expect(
      (createEmptyMeshNode('/shell', PHYSICAL, loader, null).userData as Record<string, unknown>)
        .meshTexturePlaceholders
    ).toBeUndefined();
  });

  it('picks through the HOUSE mesh pick material, seeded from the data-driven compositing', () => {
    const registerNode = vi.fn();
    const pickingSystem = {
      allocatePickId: vi.fn(() => 42),
      registerNode,
    } as unknown as PickingSystem;
    const opaque = createEmptyMeshNode('/shell', PHYSICAL, loader, pickingSystem);
    expect(opaque.userData.pickId).toBe(42);
    const [, opaquePickNode] = registerNode.mock.calls[0] as [THREE.Mesh, THREE.Mesh, number];
    const opaquePick = opaquePickNode.material as MeshPickingMaterial;
    expect(opaquePick).toBeInstanceOf(MeshPickingMaterial);
    // `opaque` pick mode = the alpha cutout is applied in the pick pass (§6.5).
    expect(opaquePick.uniforms.uAlphaCutout.value).toBe(1);

    const translucent = createEmptyMeshNode(
      '/shell',
      { ...PHYSICAL, opacity: 0.3 },
      loader,
      pickingSystem
    );
    expect((translucent.material as THREE.Material).transparent).toBe(true);
    const [, translucentPickNode] = registerNode.mock.calls[1] as [THREE.Mesh, THREE.Mesh, number];
    const translucentPick = translucentPickNode.material as MeshPickingMaterial;
    // Translucent: every fragment is pickable, matching what is drawn.
    expect(translucentPick.uniforms.uAlphaCutout.value).toBe(0);
  });

  it('applyMeshShading / applyMeshTexture are no-ops on a physical mesh', () => {
    const node = createEmptyMeshNode('/shell', PHYSICAL, loader, null);
    const before = (node.material as PhysicalMeshMaterial).flatShading;
    expect(() => applyMeshShading(node, PHYSICAL, true)).not.toThrow();
    expect((node.material as PhysicalMeshMaterial).flatShading).toBe(before);
  });

  it('learns vertex alpha at commit time through applyMeshVertexAlpha', () => {
    const node = createEmptyMeshNode('/shell', PHYSICAL, loader, null);
    expect((node.material as THREE.Material).transparent).toBe(false);
    applyMeshVertexAlpha(node, 4);
    expect((node.material as THREE.Material).transparent).toBe(true);
    expect((node.material as THREE.Material).depthWrite).toBe(false);
    // A house mesh is untouched by the same call.
    const house = createEmptyMeshNode('/house', ATTRS, loader, null);
    const houseTransparent = (house.material as THREE.Material).transparent;
    applyMeshVertexAlpha(house, 4);
    expect((house.material as THREE.Material).transparent).toBe(houseTransparent);
  });

  it('ignores an INHERITED blending_mode with a one-time notice naming the node', () => {
    const warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const node = createEmptyMeshNode(
      '/shell',
      { ...PHYSICAL, blending_mode: 'additive' },
      loader,
      null
    );
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][1])).toContain('/shell');
    expect(String(warn.mock.calls[0][1])).toContain("blending_mode='additive'");
    // The material composites by its own rule: still opaque, and stamped so.
    expect((node.material as THREE.Material).transparent).toBe(false);
    expect((node.material as THREE.Material).userData.blendingMode).toBe('opaque');
    warn.mockRestore();
  });

  it('house meshes never create a physical material (so the environment is never built for them)', () => {
    const listener = vi.fn();
    const off = materialManager.onPhysicalMaterialCreated(listener);
    createEmptyMeshNode('/house', ATTRS, loader, null);
    createEmptyMeshNode('/house2', { ...ATTRS, material: 'luxar' }, loader, null);
    expect(listener).not.toHaveBeenCalled();
    createEmptyMeshNode('/shell', PHYSICAL, loader, null);
    expect(listener).toHaveBeenCalledTimes(1);
    off();
  });
});
