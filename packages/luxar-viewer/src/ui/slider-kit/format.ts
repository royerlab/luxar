/** Return the decimal precision represented by a fixed or exponent-form step. */
export function decimalsForStep(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0;
  const [coefficient, exponentText] = String(step).toLowerCase().split('e');
  const coefficientDecimals = coefficient.split('.')[1]?.length ?? 0;
  const exponent = exponentText === undefined ? 0 : Number(exponentText);
  return Math.max(0, coefficientDecimals - exponent);
}

/** Format a number with the precision implied by its step. */
export function formatNumber(value: number, step?: number): string {
  return step === undefined ? String(value) : value.toFixed(decimalsForStep(step));
}

/** Format a slider readout, widening precision when it sits off the drag grid. */
export function formatSliderValue(
  value: number,
  baseStep: number,
  min: number,
  minimumDecimals = 2
): string {
  const quotient = baseStep > 0 ? (value - min) / baseStep : 0;
  const onBaseGrid = Math.abs(quotient - Math.round(quotient)) < 1e-9;
  const step = onBaseGrid ? baseStep : baseStep / 100;
  return value.toFixed(Math.max(minimumDecimals, Math.min(12, decimalsForStep(step))));
}
