/** Per-point hidden-dimension membership when effective radii are unavailable.
 * Keep the absolute 0.5 discrete gate aligned with effective-radius-calculator.ts
 * and partition-slice-gate.ts. */
export function points_slice_membership(
  positions: Float32Array,
  displayDims: Uint32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  discreteDims: Uint8Array,
  ndim: number,
  numPoints: number,
  output: Uint8Array
): number {
  const isDisplayDim = new Uint8Array(ndim);
  for (const d of displayDims) isDisplayDim[d] = 1;
  let visible = 0;
  for (let i = 0; i < numPoints; i++) {
    let matches = true;
    for (let d = 0; d < ndim; d++) {
      if (isDisplayDim[d] || tolerance[d] >= 1e9) continue;
      const delta = Math.abs(Math.fround(positions[i * ndim + d] - slicePosition[d]));
      if (!(delta <= (discreteDims[d] ? 0.5 : tolerance[d]))) {
        matches = false;
        break;
      }
    }
    output[i] = matches ? 1 : 0;
    if (matches) visible++;
  }
  return visible;
}
