/**
 * Per-draw projection-variant selection for materials whose graph is
 * specialized on the camera's projection KIND (orthographic vs perspective).
 *
 * The kind is taken from the projection matrix of the camera being drawn with
 * (P[3][3] — the same test as GLSL `luxarIsOrthoProjection()` and TSL
 * `isOrthoProjectionTSL()`), in the mesh's `onBeforeRender`, which three's
 * WebGPU renderer runs for every draw BEFORE it looks up the draw's render
 * object. Nothing is pushed from the CPU, so a draw through a different
 * projection than the main camera's (the scene environment capture's cube
 * faces) selects its own variant, and the next main-view draw selects the
 * other one back.
 *
 * Only the TSL screen-space line quad (visual + pick) is specialized this way:
 * it is the one graph whose single runtime-ortho form measured slower (+4.6% of
 * the GPU pass on the 10M-segment ortho quad under WebGPU). Every other
 * material reads the ortho test at runtime and has no `selectProjectionVariant`,
 * so the hook is a no-op for it.
 *
 * @module rendering/materials/_shared/projection-variant
 */
import type * as THREE from 'three';

/** A screen-space quad graph's compile-time projection kind. */
export type LineProjectionVariant = 'ortho' | 'perspective';

/** A material that can re-point itself at the graph of a projection kind. */
export interface ProjectionVariantMaterial {
  /** Select the variant for `camera`'s projection kind (no-op when unchanged). */
  selectProjectionVariant(camera: THREE.Camera): void;
}

/** True for an orthographic projection matrix (P[3][3] == 1; perspective has 0). */
export function isOrthographicProjection(camera: THREE.Camera): boolean {
  return camera.projectionMatrix.elements[15] > 0.5;
}

function hasProjectionVariant(material: unknown): material is ProjectionVariantMaterial {
  return (
    typeof material === 'object' &&
    material !== null &&
    typeof (material as Partial<ProjectionVariantMaterial>).selectProjectionVariant === 'function'
  );
}

/**
 * Make `mesh` select the drawn material's projection variant before every
 * draw. Acts on the material three is ABOUT to draw (the hook's argument), not
 * on `mesh.material` at install time, so a later material swap (layers-panel
 * clone, primitive toggle) needs no re-install.
 */
export function installProjectionVariantHook(mesh: THREE.Mesh): void {
  mesh.onBeforeRender = (_renderer, _scene, camera, _geometry, material) => {
    if (hasProjectionVariant(material)) material.selectProjectionVariant(camera);
  };
}
