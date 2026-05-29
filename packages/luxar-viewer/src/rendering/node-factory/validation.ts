/**
 * Validation helpers used by NodeFactory at load time.
 *
 * All functions are pure over their inputs, aside from logging
 * side-effects.
 *
 * @module rendering/node-factory/validation
 */

import { log, Modules } from '../../utils/log';
import type { LoadedPointsData } from '../../data/data-loader-types';

/**
 * Validate a LoadedPointsData payload. Logs warnings; throws only on
 * malformed positions.
 *
 * @param isPlaceholder - When true, the payload is the empty placeholder
 *   built before the first fetch (see `NodeFactory.createEmptyPointsNode`).
 *   An empty count is expected in that case, so the verbose diagnostic
 *   and the "empty dataset" warning are suppressed — they would otherwise
 *   fire once per points node on every scene load and look like errors.
 *   A genuinely-empty *committed* dataset (default `false`) still warns.
 */
export function validateLoadedPointsData(
  data: LoadedPointsData,
  isPlaceholder = false
): void {
  const pointCount = data.positions.length / 3;

  // Expected pre-fetch placeholder — stay silent.
  if (isPlaceholder && pointCount === 0) {
    return;
  }

  log.info(Modules.SCENE_LOADER, 'Points Data Validation:', {
    pointCount,
    positionsLength: data.positions.length,
    positionsType: data.positions.constructor.name,
    hasColors: !!data.colors,
    colorsType: data.colors?.constructor.name,
    colorsLength: data.colors?.length,
    hasRadii: !!data.radii,
    radiiType: data.radii?.constructor.name,
    radiiLength: data.radii?.length,
    hasSharpness: !!data.sharpness,
    sharpnessType: data.sharpness?.constructor.name,
    sharpnessLength: data.sharpness?.length,
  });

  if (pointCount === 0) {
    log.warning(Modules.SCENE_LOADER, 'Empty dataset detected - no points to render');
    return;
  }

  if (data.positions.length % 3 !== 0) {
    const error = `Malformed positions array: length ${data.positions.length} is not divisible by 3`;
    log.error(Modules.SCENE_LOADER, error);
    throw new Error(error);
  }

  if (data.colors && data.colors.length !== data.positions.length) {
    log.warning(
      Modules.SCENE_LOADER,
      `Colors length mismatch: expected ${data.positions.length}, got ${data.colors.length}`,
      { expected: data.positions.length, actual: data.colors.length }
    );
  }

  if (data.radii && data.radii.length !== pointCount) {
    log.warning(
      Modules.SCENE_LOADER,
      `Radii length mismatch: expected ${pointCount}, got ${data.radii.length}`,
      { expected: pointCount, actual: data.radii.length }
    );
  }

  if (data.sharpness && data.sharpness.length !== pointCount) {
    log.warning(
      Modules.SCENE_LOADER,
      `Sharpness length mismatch: expected ${pointCount}, got ${data.sharpness.length}`,
      { expected: pointCount, actual: data.sharpness.length }
    );
  }

  log.success(Modules.SCENE_LOADER, `Points data validated: ${pointCount} points`);
}

/**
 * Validate color mode consistency. Ensures color array type matches
 * expected encoding (HDR Float32Array vs SDR normalized integer).
 */
export function validateColorMode(
  colors: Uint8Array | Uint16Array | Float32Array,
  nodeMetadata: Record<string, unknown> | null | undefined
): void {
  const isHDR = colors instanceof Float32Array;
  const isSDR = colors instanceof Uint8Array || colors instanceof Uint16Array;

  if (isSDR && nodeMetadata?.color_mode === 'hdr') {
    log.warning(
      Modules.SCENE_LOADER,
      `Node metadata indicates HDR colors but array is ${colors.constructor.name}. ` +
        'HDR colors should use Float32Array. This may indicate incorrect encoding.'
    );
  }

  if (isHDR) {
    const hasHDRValues = Array.from(colors).some((v) => v > 1.0);
    if (!hasHDRValues && nodeMetadata?.color_mode === 'hdr') {
      log.info(
        Modules.SCENE_LOADER,
        'HDR color mode specified but all values in [0, 1] range. Consider using SDR mode for better compression.'
      );
    }
  }

  const colorMode = isHDR ? 'HDR (float32)' : 'SDR (normalized integer)';
  log.info(Modules.SCENE_LOADER, `Colors: ${colors.constructor.name} - ${colorMode}`);
}

/**
 * Validate transform matrix format. Throws when the matrix appears to be
 * stored row-major (NumPy) rather than column-major (THREE.js / OpenGL),
 * which is almost always a producer bug — column-major translation lives
 * at indices [12,13,14], row-major at [3,7,11].
 *
 * Edge case (both translation bands non-zero): a producer bug that
 * encodes a row-major matrix WITH a non-zero col-major slot (e.g.
 * row-major + shear, or row-major + scale touching index 12) used to
 * pass silently as "probably column-major". The current contract is to
 * throw a descriptive "ambiguous" error so the producer fixes the
 * encoding rather than letting the viewer place geometry in the wrong
 * location. Genuine column-major matrices have a `[0, 0, 0, 1]` last
 * row, so we use indices [3, 7, 11] as the disambiguator.
 */
export function validateTransformFormat(transform: readonly number[]): void {
  const colMajorTranslation = [transform[12], transform[13], transform[14]];
  const rowMajorTranslation = [transform[3], transform[7], transform[11]];

  const colMajorNonZero = colMajorTranslation.some((v) => Math.abs(v) > 0.001);
  const rowMajorNonZero = rowMajorTranslation.some((v) => Math.abs(v) > 0.001);

  if (rowMajorNonZero && !colMajorNonZero) {
    throw new Error(
      'Transform matrix appears to be stored in row-major (NumPy) format ' +
        'instead of column-major (THREE.js). Translation detected at ' +
        'indices [3,7,11] instead of [12,13,14]. Python should transpose ' +
        'before storing: matrix.T.ravel().tolist()'
    );
  }

  if (rowMajorNonZero && colMajorNonZero) {
    throw new Error(
      'Transform matrix is ambiguous — likely producer bug. Both the ' +
        'column-major translation slots [12,13,14] AND the row-major ' +
        'slots [3,7,11] contain non-zero values, so the major cannot be ' +
        'inferred. A correct column-major matrix has its last row ' +
        '[3,7,11,15] equal to [0,0,0,1]. Fix the producer to write a ' +
        'proper column-major matrix (Python: matrix.T.ravel().tolist()).'
    );
  }
}
