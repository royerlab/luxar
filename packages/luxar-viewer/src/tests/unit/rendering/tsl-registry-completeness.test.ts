/**
 * Registry-completeness gate for the TSL <-> GLSL parity harness.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * Every production `ShaderSource` carries two implementations of the same
 * shader — a GLSL pair for WebGL and a TSL factory for WebGPU — and the ONLY
 * thing that checks they agree is `tsl-shader-parity.spec.ts`, which drives
 * the harness registry. A source with no registry entry therefore has its two
 * halves compared by nothing at all, silently.
 *
 * That had already happened: `BLOOM_DOWNSAMPLE_SOURCE` and
 * `BLOOM_UPSAMPLE_SOURCE` were production shaders with no entry (audit A4-08).
 * Adding the two entries fixes today; this test fixes tomorrow, which is the
 * more valuable half. Adding a shader without registering it now fails here.
 *
 * WHY STATIC ANALYSIS RATHER THAN IMPORTING THE REGISTRY
 * ------------------------------------------------------
 * Importing `SHADER_REGISTRY` would pull in three.js WebGPU and the whole TSL
 * factory graph, which needs a browser. This test has to run in the cheap
 * node-environment unit suite to be a useful gate, so it reads both sides as
 * text. The cost is that it verifies a source is REFERENCED by the harness,
 * not that its entry is meaningful — the parity spec itself covers that.
 */

import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const VIEWER_SRC = join(HERE, '..', '..', '..');
const RENDERING_DIR = join(VIEWER_SRC, 'rendering');
const HARNESS_DIR = join(VIEWER_SRC, 'tests', 'e2e', 'harnesses', 'tsl-harness');

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...walk(full));
    } else if (entry.endsWith('.ts')) {
      out.push(full);
    }
  }
  return out;
}

/** `export const NAME: ShaderSource = {` across the production rendering tree. */
function findProductionShaderSources(): Map<string, string> {
  const found = new Map<string, string>();
  const pattern = /export\s+const\s+([A-Z0-9_]+)\s*:\s*ShaderSource\s*=/g;
  for (const file of walk(RENDERING_DIR)) {
    const text = readFileSync(file, 'utf8');
    for (const match of text.matchAll(pattern)) {
      found.set(match[1], file.slice(VIEWER_SRC.length + 1));
    }
  }
  return found;
}

/** Identifiers the harness modules mention anywhere. */
function harnessText(): string {
  return walk(HARNESS_DIR)
    .map((file) => readFileSync(file, 'utf8'))
    .join('\n');
}

describe('TSL parity registry completeness', () => {
  it('finds the production shader sources at all', () => {
    // Guards every assertion below: if the export spelling ever changes, this
    // test would otherwise pass by scanning an empty set.
    const sources = findProductionShaderSources();
    expect(sources.size).toBeGreaterThanOrEqual(12);
    expect([...sources.keys()]).toContain('MEGA_SOURCE');
  });

  it('registers every production ShaderSource in the parity harness', () => {
    const sources = findProductionShaderSources();
    const harness = harnessText();

    const unregistered = [...sources.entries()]
      .filter(([name]) => !harness.includes(name))
      .map(([name, file]) => `${name} (${file})`)
      .sort();

    expect(
      unregistered,
      [
        'These ShaderSource values have no entry in the TSL parity harness, so',
        'their WebGL and WebGPU implementations are compared by nothing.',
        'Add an entry in src/tests/e2e/harnesses/tsl-harness/ — see the',
        "'bloom-downsample' entry in post-processing.ts for the minimal shape.",
      ].join(' ')
    ).toEqual([]);
  });

  it('does not reference shader sources that no longer exist', () => {
    // The other direction: a renamed or deleted source leaves a dangling
    // harness import, which fails the parity run in a much less obvious way.
    const sources = findProductionShaderSources();
    const harness = harnessText();

    // The harness defines shaders of its own — `ERF_SOURCE` exists only to
    // exercise the shared erf math against its TSL twin and has no production
    // counterpart by design. Those are known, not dangling.
    const harnessLocal = new Set(
      [...harness.matchAll(/const\s+([A-Z0-9_]+)\s*:\s*ShaderSource\s*=/g)].map((m) => m[1])
    );
    expect(harnessLocal).toContain('ERF_SOURCE');

    const referenced = [...harness.matchAll(/\b([A-Z0-9_]*_SOURCE)\b/g)].map((m) => m[1]);
    const dangling = [...new Set(referenced)]
      .filter((name) => !sources.has(name) && !harnessLocal.has(name))
      .sort();
    expect(dangling).toEqual([]);
  });
});
