/**
 * Pick materials with a switchable depth convention: the point, line and
 * gsplat pick wrappers (GLSL and TSL). The picking system's per-render mode
 * sync detects the capability via {@link isSurfacePickAwareMaterial},
 * mirroring the `CameraAwareMaterial` guard idiom. (Mesh implements the
 * richer `MeshPickAwareMaterial` instead — `../mesh/pick-mode.ts`.)
 *
 * @module rendering/picking/_shared/surface-pick
 */

export interface SurfacePickAwareMaterial {
  /**
   * Select the pick depth convention: `true` = real projected depth
   * (front-most wins; the depth-ordered `opaque` / `normal` surface modes),
   * `false` = brightness-as-depth (brightest wins; commutative modes).
   */
  setSurfacePickDepth(on: boolean): void;
}

/** Type guard for {@link SurfacePickAwareMaterial}. */
export function isSurfacePickAwareMaterial(
  material: unknown
): material is SurfacePickAwareMaterial {
  return (
    typeof material === 'object' &&
    material !== null &&
    'setSurfacePickDepth' in material &&
    typeof (material as Record<string, unknown>).setSurfacePickDepth === 'function'
  );
}
