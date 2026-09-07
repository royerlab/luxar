/**
 * Construct the `THREE.Mesh` for a Mesh node, with its shaded material.
 *
 * Three things here are mesh-specific rather than copies of the sibling factories,
 * and each is named in `docs/specs/MESH_NODE_SPEC.md`:
 *
 * 1. **The default blending mode is `opaque`, not `additive`** (§6.3) — and it is
 *    applied HERE, viewer-side, never stamped by the writer. A stamped
 *    `blending_mode` would override an ancestor's under the nearest-setter-wins
 *    rule, silently breaking `group(blending_mode="additive")` for its mesh
 *    children.
 * 2. **`volumetric` degrades to `opaque` with a one-time warning** (§6.3), again
 *    viewer-side and for the same reason: the mode can arrive by INHERITANCE from an
 *    ancestor the mesh knows nothing about at write time, so refusing the load would
 *    make an unrelated group setting break a valid mesh.
 * 3. **The shading variant is decided here, once, and handed to both backends**
 *    (§6.2). It is a compile-time shader variant because a declared-but-unbound
 *    `normal` attribute reads `(0, 0, 0, 1)` rather than "absent" — there is no
 *    runtime value meaning "no normals". Its `shading` half is view-independent; its
 *    `normal_dims == displayDims` half is not, so {@link applyMeshShading} re-applies
 *    the decision per epoch from the projection's `storedNormalsUsable`.
 *
 * @module rendering/node-factory/create-mesh-node
 */

import * as THREE from 'three';
import { createMeshGeometry } from '../mesh-geometry';
import { applyTransform } from './transforms';
import {
  materialManager,
  type BlendingMode,
  type LuxarMeshMaterial,
  type LuxarMeshPickingMaterial,
  type LuxarPhysicalMeshMaterial,
} from '../material-manager';
import {
  isPhysicalMeshMaterial,
  physicalSetVertexAlpha,
  type PhysicalMeshHost,
} from '../materials/mesh-physical/config';
import { isMeshPickAwareMaterial } from '../picking/mesh/pick-mode';
import type { PickingSystem } from '../picking/picking-system';
import { getColormapTexture } from '../colormap-textures';
import { supportsScalarColormap } from '../material-colormap-helpers';
import { resolveColormapWindow } from '../display-range';
import { scheduleBlendModeProgramWarmupForObject } from '../webgl-blend-warmup';
import { log, Modules } from '../../utils/log';
import type { MeshSide } from '../../data/mesh/projection';
import type { MeshShadingMode } from '../materials/mesh/appearance';
import { createMeshTexture } from '../mesh-texture';
import type { MeshTextureData } from '../../types/mesh';
import type { MeshDataLoader, MeshMetadata, MeshUserData } from '../../types/mesh';

/**
 * The blending mode a mesh node asks for, defaulting to `opaque` (§6.3).
 *
 * The `?? 'opaque'` is the whole mesh-specific asymmetry — the three siblings all
 * default to `additive` — so it lives in ONE place, read by both the visual material
 * and the pick material. Two copies would be two chances for the pick pass to apply
 * a cutout the screen does not, or vice versa.
 */
export function resolveRequestedMeshMode(attrs: MeshMetadata): BlendingMode {
  return (attrs.blending_mode as BlendingMode | undefined) ?? 'opaque';
}

/** Map the projection's epoch decision onto a three.js side constant. */
function threeSide(side: MeshSide): THREE.Side {
  return side === 'double' ? THREE.DoubleSide : THREE.FrontSide;
}

/**
 * Apply the epoch's material side to a mesh object.
 *
 * Separate from construction because `side` is a property of the current
 * `displayDims` epoch, not of the node: an odd-parity reflection stays
 * single-sided (the index post-pass restored winding) while an undecidable frame
 * forces double-sided regardless of the authored `double_sided`. So the commit path
 * re-applies it on every rebuild.
 *
 * Guarded on change: assigning `material.side` unconditionally would set
 * `needsUpdate` semantics in motion on every slice move, and on the WebGPU backend a
 * side change forces a pipeline rebuild.
 */
export function applyMeshSide(object: THREE.Mesh, side: MeshSide): void {
  const material = object.material as THREE.Material;
  const wanted = threeSide(side);
  if (material.side !== wanted) {
    material.side = wanted;
    material.needsUpdate = true;
  }
  // The pick pass must cull the same faces (§6.5). Not merely for tidiness: with
  // the visual material on FrontSide and the pick material left DoubleSide, the
  // interior faces of a sliced closed isosurface rasterize into the pick buffer at
  // true surface depth — pickable where nothing is drawn, AND occluding picks of
  // the nodes actually visible through the opening.
  //
  // Also pushed per pick render by `PickingSystem.renderPickBuffer`, which is what
  // covers a runtime `material.side` write that does not come through here. Doing
  // it in both places is deliberate: this one keeps the FIRST pick after a commit
  // correct even if the render sync were ever narrowed.
  const pickMaterial = (object.userData.pickNode as THREE.Mesh | undefined)?.material;
  if (pickMaterial && !Array.isArray(pickMaterial) && isMeshPickAwareMaterial(pickMaterial)) {
    pickMaterial.setPickSide(wanted);
  }
}

/**
 * Re-apply the epoch's shading variant (spec §6.2 / §3.4).
 *
 * The stored-normal half of the rule is view-dependent — `normal_dims` must equal
 * the *active* `displayDims`, not merely be a permutation of it — so a `displayDims`
 * change can flip a `shading="smooth"` node between the two variants. That is the
 * same event that re-extracts the display-space positions, so the commit calls this
 * right next to {@link applyMeshSide}.
 *
 * The `shading` half never changes, which is why an explicitly flat node is pinned
 * to the derivative variant here regardless of what the projection reports.
 *
 * Guarded on change inside the material (both wrappers no-op when the define is
 * already in the wanted state), because flipping the variant recompiles the program
 * on the GLSL backend and rebuilds the graph on the TSL one — a per-slice-move cost
 * that must not be paid when nothing changed.
 */
export function applyMeshShading(
  object: THREE.Mesh,
  attrs: MeshMetadata,
  storedNormalsUsable: boolean
): void {
  const material = object.material as LuxarMeshMaterial | LuxarPhysicalMeshMaterial;
  if (typeof material.updateShading !== 'function') return;
  material.updateShading(resolveMeshShading(attrs, storedNormalsUsable));
  scheduleBlendModeProgramWarmupForObject(object);
}

/**
 * Upload the decoded texture into the material, once the data has arrived.
 *
 * Separate from node creation because a texture is DATA: `createEmptyMeshNode` runs
 * before any fetch, so the material is built with a blank placeholder and the real
 * image is installed here, on the commit that carries it.
 *
 * What is decided at CREATION and never here is the shader VARIANT — the define and
 * the `uv` attribute both key off `attrs.has_texture`, a per-node constant. That
 * split is the point: the variant is fixed from birth (WebGPU bakes the attribute
 * set into the pipeline at first draw), and only the texture's IDENTITY changes,
 * which on GLSL is a plain uniform write with no recompile.
 *
 * Idempotent, so the steady state after the first commit costs nothing: the setter
 * compares the incoming texture against the bound one and only rebuilds the TSL
 * graph when they differ. That comparison is load-bearing on the TSL backend, where
 * `texture()` captures its Texture at construction — a slice move that re-installed
 * the same image would otherwise rebuild the graph on every frame of a scrub.
 */
export function applyMeshTexture(
  object: THREE.Mesh,
  attrs: MeshMetadata,
  data: MeshTextureData
): void {
  const material = object.material as LuxarMeshMaterial;
  if (typeof material.updateBaseColorTexture !== 'function') return;

  // Uploaded ONCE per node and cached on the object, keyed by the payload's
  // identity. The loader caches the decoded mesh for its whole life and hands back
  // the same `data.texture` on every `updateView`, so identity is a sound key — and
  // re-uploading a 2048x1024 basemap on every slice move is the regression this
  // avoids. Cached on the node rather than in a module map so it is collected with
  // the node and cannot outlive a dataset switch.
  const cache = object.userData as {
    meshTextureSource?: MeshTextureData;
    meshTexture?: THREE.Texture;
    meshTexturePlaceholders?: THREE.Texture[];
  };
  cache.meshTexturePlaceholders?.forEach((placeholder) => placeholder.dispose());
  delete cache.meshTexturePlaceholders;
  let texture = cache.meshTexture ?? null;
  if (cache.meshTextureSource !== data || !texture) {
    cache.meshTexture?.dispose();
    texture = createMeshTexture(data, attrs, materialManager.getTextureCapabilities(), object.name);
    object.geometry.addEventListener('dispose', () => texture?.dispose());
    cache.meshTextureSource = data;
    cache.meshTexture = texture;
  }

  const luminance = attrs.texture_channels === 1;
  material.updateBaseColorTexture(texture, luminance);

  // The pick pass samples the same texture, for its alpha: texture alpha multiplies
  // coverage, so an RGBA basemap's holes are real holes on screen and must not stay
  // pickable or depth-occluding.
  //
  // Through the METHOD, not the uniform, and the difference is backend-visible: on
  // GLSL the uniform IS the binding, but the TSL twin's `texture()` node captured
  // its Texture at construction, so a uniform write there would leave the pick pass
  // sampling the blank placeholder — alpha 0 everywhere, making the whole mesh
  // unpickable. Both wrappers expose the same method name so there is no branch.
  const pickMaterial = (object.userData.pickNode as THREE.Mesh | undefined)?.material;
  if (pickMaterial && !Array.isArray(pickMaterial)) {
    const pick = pickMaterial as { updateBaseColorTexture?: (t: THREE.Texture | null) => void };
    pick.updateBaseColorTexture?.(texture);
  }
}

/**
 * How this node obtains its normals, as ONE value.
 *
 * The full §3.4 rule in one place: stored normals are used **iff**
 * `shading === 'smooth'` AND the node actually has them AND they were authored for
 * the axes currently displayed. Every other lit case — explicit `'flat'`, absent
 * normals, or a frame mismatch — takes the derivative fallback, which is correct for
 * the projected geometry, just faceted rather than smoothed.
 *
 * `'none'` short-circuits ahead of all of that, and the ORDER matters: an unlit mesh
 * needs no normal, so falling through to the derivative fallback would compute one
 * and discard it. It is also the only arm that is NOT view-dependent — the other two
 * can flip on a `displayDims` change, this one is purely authored — which is why it
 * is answered before anything reads `storedNormalsUsable`.
 *
 * Returns an enum rather than the earlier `flatNormal` boolean: a second boolean for
 * the unlit arm would admit a meaningless `flatNormal && noShading` state and double
 * the shader variant count for something with one behaviour.
 *
 * `storedNormalsUsable` already folds in "has normals at all" (it is `false` without
 * `normal_dims`), so `has_normals` is checked here only to keep the rule readable
 * against a hand-built attrs object in a test.
 */
export function resolveMeshShading(
  attrs: MeshMetadata,
  storedNormalsUsable: boolean
): MeshShadingMode {
  if (attrs.shading === 'none') return 'none';
  if (attrs.shading === 'flat') return 'flat';
  if (!attrs.has_normals) return 'flat';
  return storedNormalsUsable ? 'smooth' : 'flat';
}

/**
 * Build the mesh material, applying the colormap when the node carries scalars.
 *
 * Mirrors `createPointsMaterial` — including the #936 double-apply rule, where an
 * authored gain that turns out to be a scalar WINDOW is re-expressed as the LUT
 * range and cleared from the post-LUT GOG.
 *
 * `attrs` is the COMPOSED effective attrs (`load-mesh-node.ts` passes
 * `ctx.applyEffectiveAttrs(node)`). `leafAttrs` is the node's RAW uncomposed attrs,
 * needed only to tell a leaf-authored scalar window from an inherited ancestor gain;
 * it defaults to `attrs`, which is exactly right when no ancestor authored a gain.
 */
export function createMeshMaterial(
  attrs: MeshMetadata,
  shading: MeshShadingMode,
  geometry?: THREE.BufferGeometry,
  path?: string,
  leafAttrs?: Partial<MeshMetadata>
): LuxarMeshMaterial {
  const composedIntensity = attrs.intensity ?? 1.0;
  const composedOffset = attrs.offset ?? 0.0;
  const requestedMode = resolveRequestedMeshMode(attrs);

  if (requestedMode === 'volumetric') {
    // Once per node, naming it — §6.3. A warning rather than a failure because the
    // mode may be inherited; the material maps it to `opaque` on its own, so this is
    // purely the user-facing half.
    log.warning(
      Modules.SCENE_LOADER,
      `[${path ?? '<mesh>'}] blending_mode='volumetric' has no meaning for a mesh: ` +
        'volumetric blending integrates emission and absorption along the view ray ' +
        'through a participating medium, and a triangle is a zero-thickness surface ' +
        "with no path length to absorb over. Falling back to 'opaque'."
    );
  }

  const material = materialManager.getMeshMaterial({
    opacity: attrs.opacity ?? 1.0,
    gamma: attrs.gamma ?? 1.0,
    // The shade knobs ride in through `add_mesh(**attrs)` — the writer never stamps
    // them, so they are present only when authored. This authoring path and the
    // material reads landed together; the material clamps fractions and exponents
    // to their defined domains.
    ambient: attrs.ambient,
    shadeExponent: attrs.shade_exponent,
    specular: attrs.specular,
    shininess: attrs.shininess,
    alphaCutoff: attrs.alpha_cutoff,
    // The authored gain starts life as the post-LUT colour GOG (the direct-colour
    // meaning). If the colormap takes over below it is RESET to identity there and
    // re-expressed as the scalar window instead — applying it as both
    // double-applies (#936).
    intensity: composedIntensity,
    offset: composedOffset,
    blendingMode: requestedMode,
    shading,
    // A BLANK placeholder, not the real image — which has not been fetched yet.
    // Seeding it here rather than at the first commit is what fixes the shader
    // VARIANT from birth: `has_texture` is a per-node constant, so the define never
    // flips and (on GLSL) the arriving image is a uniform write with no recompile.
    // Nothing draws in the meantime — the placeholder geometry has zero faces.
    ...(attrs.has_texture
      ? {
          baseColorTexture: new THREE.Texture(),
          baseColorTextureLuminance: attrs.texture_channels === 1,
        }
      : {}),
  });

  const colormapName = attrs.colormap;
  if (colormapName && attrs.has_scalars) {
    // Fail closed: `USE_COLORMAP` reads the `aScalar` attribute, whose presence
    // rides the `userData.hasScalars` stamp the geometry builder writes. When no
    // geometry is supplied (tests constructing the material directly), trust the
    // caller — same rule as `createPointsMaterial`.
    const guardOK = !geometry || supportsScalarColormap('mesh', geometry);
    if (!guardOK) {
      log.warning(
        Modules.SCENE_LOADER,
        `[${path ?? '<mesh>'}] Scalar colormap requested but no scalar data is bound on ` +
          'the mesh geometry. Colormap suppressed; rendering with vertex colors.'
      );
    } else {
      const lutBytes = (attrs as { customLutBytes?: Uint8Array }).customLutBytes;
      const colormapTex = getColormapTexture(colormapName, lutBytes);
      if (colormapTex) {
        material.updateColormapTexture(colormapTex);
        const leaf = leafAttrs ?? attrs;
        const scalarRange = resolveColormapWindow(
          attrs.scalar_data_range ?? [0, 1],
          { intensity: leaf.intensity ?? 1.0, offset: leaf.offset ?? 0.0 },
          { intensity: composedIntensity, offset: composedOffset }
        );
        material.updateScalarRange(scalarRange[0], scalarRange[1]);
        // The window now drives the LUT lookup; clear the post-LUT gain the material
        // was built with so it does not double-apply (#936).
        material.updateIntensity(1);
        material.updateOffset(0);
      }
    }
  }

  return material;
}

/**
 * Build the PHYSICAL material for a `material="physical"` mesh
 * (`MESH_PHYSICAL_MATERIALS_SPEC.md` §3.2).
 *
 * The third material family, and deliberately not a branch inside
 * {@link createMeshMaterial}: nothing that function does — the volumetric warning, the
 * shade knobs, the colormap window, the texture placeholder — applies here. The
 * authoring side refuses every one of those knobs under this material, so what is
 * mapped is exactly what can arrive: the numeric knobs of `PHYSICAL_MESH_KNOBS` (the
 * six surface fractions and the glass family), the two `#rrggbb` colours,
 * the compositing attrs `opacity` / `intensity` / `offset` / `gamma` (each with the
 * meaning `materials/mesh-physical/config.ts` documents), `alpha_cutoff` as
 * `alphaTest`, and the flat/smooth half of `shading` as `flatShading`.
 *
 * An INHERITED `blending_mode` can still reach a physical mesh from an ancestor group
 * that the mesh knows nothing about at write time. It is ignored — a physical mesh
 * composites by its own data-driven rule — with a one-time notice naming the node, the
 * same policy the house shader applies to an inherited `volumetric`.
 */
export function createPhysicalMeshMaterial(
  attrs: MeshMetadata,
  shading: MeshShadingMode,
  path?: string
): LuxarPhysicalMeshMaterial {
  if (attrs.blending_mode !== undefined) {
    log.warning(
      Modules.SCENE_LOADER,
      `[${path ?? '<mesh>'}] blending_mode='${attrs.blending_mode}' reached a ` +
        "material='physical' mesh (inherited from an ancestor group) and is ignored: a " +
        'physical mesh composites by its own rule — opaque unless its opacity or vertex ' +
        'alpha is below 1 — and has no additive/luminous/max emission to select.'
    );
  }
  return materialManager.getMeshPhysicalMaterial({
    opacity: attrs.opacity ?? 1.0,
    intensity: attrs.intensity ?? 1.0,
    offset: attrs.offset ?? 0.0,
    gamma: attrs.gamma ?? 1.0,
    roughness: attrs.roughness,
    metalness: attrs.metalness,
    clearcoat: attrs.clearcoat,
    clearcoatRoughness: attrs.clearcoat_roughness,
    iridescence: attrs.iridescence,
    sheen: attrs.sheen,
    sheenColor: attrs.sheen_color,
    transmission: attrs.transmission,
    ior: attrs.ior,
    thickness: attrs.thickness,
    attenuationDistance: attrs.attenuation_distance,
    attenuationColor: attrs.attenuation_color,
    dispersion: attrs.dispersion,
    refractData: attrs.refract_data === true,
    alphaCutoff: attrs.alpha_cutoff,
    // `'none'` is refused at authoring; `'flat'` is either authored or the fallback
    // for a mesh with no stored normals — in both cases three derives per-triangle
    // normals, which is exactly what the house shader's flat variant does.
    flatShading: shading !== 'smooth',
    // Unknown until the first commit decodes the colour buffer — see
    // {@link applyMeshVertexAlpha}. Starting opaque is the conservative guess: a
    // translucent placeholder would flicker for one frame on an RGB mesh.
    vertexAlpha: false,
  });
}

/**
 * Tell a physical mesh whether its committed colours carry alpha.
 *
 * `MeshMetadata` records `has_colors` but not the component count, and the node is
 * created before any fetch, so translucency-by-vertex-alpha can only be decided on
 * the commit that carries the buffer. A no-op for a house material (which reads
 * per-vertex alpha in its shader regardless) and when nothing changed.
 */
export function applyMeshVertexAlpha(
  object: THREE.Mesh,
  colorComponents: number | undefined
): void {
  const material = object.material;
  if (Array.isArray(material) || !isPhysicalMeshMaterial(material)) return;
  physicalSetVertexAlpha(material as unknown as PhysicalMeshHost, colorComponents === 4);
}

/**
 * Create an empty mesh node — a `THREE.Mesh` with a one-vertex degenerate geometry,
 * attached before any data is fetched.
 *
 * Same contract as the sibling `createEmptyXNode` factories: the commit path finds
 * this object by name and populates it, so an initial-load failure leaves a
 * recoverable scene state rather than a hole, and a later retry has somewhere to
 * write. The geometry is replaced in place on first commit.
 */
export function createEmptyMeshNode(
  path: string,
  attrs: MeshMetadata,
  loader: MeshDataLoader,
  pickingSystem: PickingSystem | null,
  leafAttrs?: Partial<MeshMetadata>
): THREE.Mesh {
  const geometry = createMeshGeometry({
    // One vertex and no indices: a valid, drawable-but-empty geometry.
    //
    // The choice is conservative rather than forced, and it is worth saying so
    // precisely because an earlier version of this comment claimed a zero-vertex buffer
    // makes `computeBoundingSphere` produce NaN bounds. MEASURED against three r184,
    // that is false in every direction: a 0-count `position` gives `radius = 0`, and an
    // ABSENT `position` gives three's `radius = -1` "no geometry" sentinel. Neither is
    // NaN. What one vertex does buy is that every attribute — including the
    // always-bound default `color` — has `count >= 1`, so the geometry is
    // non-degenerate for anything that divides by or iterates over the count, and its
    // bounding sphere is a real sphere rather than the -1 sentinel.
    position: new Float32Array(3),
    // The placeholder's buffer is brand new, so it is trivially "changed".
    positionChanged: true,
    indices: new Uint32Array(0),
    colors: null,
    // One-vertex STUBS for the optional attributes the node declares, so the
    // attribute SET is complete from birth and the first commit only replaces
    // contents. Adding an attribute to a live geometry instead would grow the
    // vertex layout the WebGPU backend caches at first draw. Keyed off the
    // metadata, which is what the real arrays' presence will agree with.
    normals: attrs.has_normals ? new Float32Array(3) : null,
    scalars: attrs.has_scalars ? new Float32Array(1) : null,
    uvs: attrs.has_uvs ? new Float32Array(2) : null,
    vertexCount: 1,
    faceCount: 0,
  });

  // The stored-normal half of the shading rule needs the active `displayDims`, which
  // no one knows yet — the first projection decides it. Start from the
  // view-INDEPENDENT half alone, which is the right guess for the overwhelmingly
  // common case (`normal_dims` authored for the axes the scene opens on), and let
  // the first commit's `applyMeshShading` correct it if not.
  const shading = resolveMeshShading(attrs, attrs.has_normals);
  // The material FAMILY branches here and only here (spec §3.2): everything below —
  // geometry, picking with the house mesh-pick material, side, transform,
  // `layer_order`, userData — is shared by both families.
  const physical = attrs.material === 'physical';
  const material: LuxarMeshMaterial | LuxarPhysicalMeshMaterial = physical
    ? createPhysicalMeshMaterial(attrs, shading, path)
    : createMeshMaterial(attrs, shading, geometry, path, leafAttrs);
  material.side = attrs.double_sided ? THREE.DoubleSide : THREE.FrontSide;

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = path;
  // Safe for the same reason the other three are: every commit path refreshes the
  // bounds whenever `position` is replaced (see `updateMeshGeometry`), so the
  // sphere the frustum test reads never goes stale. Unlike the emissive types
  // there is no footprint expansion to apply — a triangle's extent IS its
  // vertices, so the vertex-spanning box is already the rendered footprint.
  mesh.frustumCulled = true;

  const userData: MeshUserData = {
    nodeType: 'mesh',
    loader,
    attrs,
    path,
    visibleTriangleCount: 0,
    visibleVertexCount: 0,
    // Per-node material from creation: the layers panel and the LOD cross-fade
    // honor this marker and mutate the material directly instead of
    // clone-on-first-use (mirrors createPointsNode / createLinesNode).
    _layerMaterialCloned: true,
  };
  mesh.userData = userData;
  mesh.userData.layerOrder = (attrs as unknown as Record<string, unknown>).layer_order;

  const texturePlaceholders: THREE.Texture[] = [];
  // Optional-chained because the physical family has no `uniforms` — it is three's
  // own material — and cannot carry a texture placeholder (textures are refused at
  // authoring under `material="physical"`).
  const visualPlaceholder = (material as { uniforms?: Record<string, { value?: unknown }> })
    .uniforms?.uBaseColorTex?.value;
  if (visualPlaceholder instanceof THREE.Texture) texturePlaceholders.push(visualPlaceholder);

  if (attrs.transform) applyTransform(mesh, attrs.transform);

  if (pickingSystem) {
    const pickId = pickingSystem.allocatePickId();
    mesh.userData.pickId = pickId;
    // The pick material reads the same coverage the visual one does, so it starts
    // from the same node opacity and cutoff. Both are re-pushed by the layers panel
    // through `syncMeshPickAppearance`; seeding them here keeps the FIRST pick
    // (which can precede any panel interaction) consistent with the screen.
    const pickPlaceholder = attrs.has_texture ? new THREE.Texture() : null;
    const pickMaterial = materialManager.createMeshPickingMaterial({
      nodeId: pickId,
      opacity: attrs.opacity ?? 1.0,
      alphaCutoff: attrs.alpha_cutoff,
      // Same placeholder-at-creation rule as the visual material above: this
      // decides whether the pick program declares a sampler, and `has_texture` is a
      // per-node constant. `applyMeshTexture` installs the real image at commit.
      baseColorTexture: pickPlaceholder,
    });
    if (pickPlaceholder) texturePlaceholders.push(pickPlaceholder);
    materialManager.register(pickMaterial);
    // Share the same indexed BufferGeometry — only the material differs. The
    // picking system re-syncs `geometry` from the main node every pick render, so a
    // slice move's index rewrite (and the drawRange that rides with it) reaches the
    // pick pass without any per-commit plumbing.
    const pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);
    pickNode.matrixWorld.copy(mesh.matrixWorld);
    pickingSystem.registerNode(mesh, pickNode, pickId);
    // registerNode stamps `mesh.userData.pickNode`, which is how applyMeshSide and
    // the appearance sync below find their way back here.
    pickMaterial.setPickSide(material.side);
    // A physical mesh has no Luxar blending mode; its pick mode follows the same
    // data-driven rule its compositing does (`derivePhysicalCompositing`): the cutout
    // when opaque, every fragment when translucent. The per-render resync in
    // `PickingSystem.renderPickBuffer` reads the same live `transparent` property.
    pickMaterial.setPickMode(
      physical ? (material.transparent ? 'normal' : 'opaque') : resolveRequestedMeshMode(attrs)
    );
  }

  if (texturePlaceholders.length > 0) {
    (
      mesh.userData as MeshUserData & { meshTexturePlaceholders: THREE.Texture[] }
    ).meshTexturePlaceholders = texturePlaceholders;
    for (const placeholder of texturePlaceholders) {
      geometry.addEventListener('dispose', () => placeholder.dispose());
    }
  }

  return mesh;
}

/**
 * Push the coverage-affecting appearance values onto a node's pick material.
 *
 * `opacity` and `alpha_cutoff` are the two visual values the pick pass reads (they
 * are the whole coverage term and the cutout threshold — §6.5), and the layers panel
 * can change either at runtime. Without this the pick pass would keep the
 * load-time values: dragging opacity below the cutoff would dissolve the mesh on
 * screen while leaving every triangle pickable.
 *
 * A no-op for a node with no pick material (picking disabled, or a test-built node).
 *
 * Returns whether a real mesh pick material was updated, so callers can invalidate
 * the cached pick buffer for meshes only — the other geometry types (whose pick
 * coverage this never touches) get `false` and skip the needless offscreen re-render.
 */
export function syncMeshPickAppearance(
  object: THREE.Mesh,
  values: { opacity?: number; alphaCutoff?: number }
): boolean {
  const pickMaterial = (object.userData.pickNode as THREE.Mesh | undefined)?.material;
  if (!pickMaterial || Array.isArray(pickMaterial)) return false;
  if (!isMeshPickAwareMaterial(pickMaterial)) return false;
  const pick = pickMaterial as LuxarMeshPickingMaterial;
  if (values.opacity !== undefined) pick.updateOpacityUniform(values.opacity);
  if (values.alphaCutoff !== undefined) pick.updateAlphaCutoff(values.alphaCutoff);
  return true;
}
