/** Return the DOM step that can represent the extra-fine tier. */
export function fineTrackStep(baseStep: number): string {
  return Number.isFinite(baseStep) && baseStep > 0 ? String(baseStep / 100) : 'any';
}

/** Snap a value to a min-anchored step grid with stable midpoint rounding. */
export function snapToGrid(value: number, min: number, step: number): number {
  if (!Number.isFinite(step) || step <= 0) return value;
  const quotient = Number(((value - min) / step).toPrecision(12));
  return min + Math.round(quotient) * step;
}
