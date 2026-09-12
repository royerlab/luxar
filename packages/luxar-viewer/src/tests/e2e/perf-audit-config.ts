export interface LodBiasArm {
  value: number | null;
  scenarioSuffix: string;
  query: string;
}

export function parseLodBiasArms(raw: string | undefined): LodBiasArm[] {
  const entries = (raw ?? '')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (entries.length === 0) return [{ value: null, scenarioSuffix: '', query: '' }];

  const values = [...new Set(entries.map(Number))];
  for (const value of values) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(
        `LUXAR_PERF_AUDIT_LOD_BIASES must contain positive finite numbers; received '${raw}'`
      );
    }
  }
  return values.map((value) => ({
    value,
    scenarioSuffix: `-lod-bias-${value}`,
    query: `&lod-bias=${value}`,
  }));
}
