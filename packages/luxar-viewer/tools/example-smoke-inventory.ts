import { existsSync, readdirSync } from 'node:fs';

export type ExampleSmokeExclusions = Readonly<Record<string, string>>;

export function discoverExampleDatasets(
  examplesRoot: string,
  exclusions: ExampleSmokeExclusions
): string[] {
  if (!existsSync(examplesRoot)) {
    throw new Error(
      `Examples directory not found: ${examplesRoot}. Run "make run-examples" from the repository root.`
    );
  }

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

export function validateExampleDatasetReferences(
  datasets: readonly string[],
  exclusions: ExampleSmokeExclusions,
  references: readonly string[],
  description: string
): void {
  const generated = new Set([...datasets, ...Object.keys(exclusions)]);

  for (const dataset of references) {
    if (!generated.has(dataset)) {
      throw new Error(`${description} ${dataset} is not present in the generated example corpus`);
    }
  }
}
