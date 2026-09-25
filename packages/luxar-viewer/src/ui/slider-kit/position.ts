/** Project a value onto a slider's `[0, 1]` track fraction. */
export function valueToFraction(value: number, min: number, max: number): number {
  const range = max - min;
  return range === 0 ? 0.5 : (value - min) / range;
}

/** Map a `[0, 1]` track fraction back into value space. */
export function fractionToValue(fraction: number, min: number, max: number): number {
  const range = max - min;
  return range === 0 ? min : min + fraction * range;
}

/** Convert a track fraction to the thumb's left pixel offset. */
export function fractionToThumbLeft(
  fraction: number,
  containerWidth: number,
  thumbWidth: number
): number {
  return fraction * (containerWidth - thumbWidth);
}
