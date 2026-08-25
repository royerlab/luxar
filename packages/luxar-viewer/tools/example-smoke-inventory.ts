import { readdirSync } from 'node:fs';

export type ExampleSmokeExclusions = Readonly<Record<string, string>>;

export function discoverExampleDatasets(
  examplesRoot: string,
  exclusions: ExampleSmokeExclusions
): string[] {
  const datasets = readdirSync(examplesRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.endsWith('.luxar.zarr'))
    .map((entry) => entry.name)
    .sort();
  const generated = new Set(datasets);
  const excluded = new Set(Object.keys(exclusions));

  for (const [dataset, reason] of Object.entries(exclusions)) {
    if (!reason.trim()) {
      throw new Error(`Example smoke exclusion ${dataset} must include a reason`);
    }
    if (!generated.has(dataset)) {
      throw new Error(`Example smoke exclusion ${dataset} is not present in ${examplesRoot}`);
    }
  }

  return datasets.filter((dataset) => !excluded.has(dataset));
}
