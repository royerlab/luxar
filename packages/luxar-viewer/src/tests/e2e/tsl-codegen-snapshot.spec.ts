/**
 * TSL → generated-shader snapshot harness.
 *
 * The visual TSL source can look reasonable while the generated GLSL
 * / WGSL is bloated (extra `select(...).toVar()` branches, duplicate
 * brightness computations across `colorNode` and `depthNode`,
 * unnecessary varyings, etc.). This spec captures the actual shader
 * code emitted by `WebGPURenderer({ forceWebGL: true })` for each
 * pinned shader variant (points, lines, gsplats + their pick and
 * fast-path builds) and pins it to a checked-in snapshot file under
 * `src/tests/__codegen__/`.
 *
 * Two purposes:
 *   1. **Regression detection.** Every commit that touches the TSL
 *      graph re-runs this spec. If the generated code diverges from
 *      the snapshot (e.g. a supposedly equivalent change adds an `if`
 *      branch or duplicates a `pow()`), the diff is the alarm.
 *   2. **Optimization evidence.** Commits that intentionally change
 *      generated code update the snapshot and the diff goes into the
 *      commit body — concrete proof that, e.g., `forceSinglePass` or
 *      a `select` → `If` conversion landed.
 *
 * Counters (`pow`, `tan`, `sqrt`/`length`, `texture`/`textureSample`,
 * varying count, branch count) are printed for each shader so the
 * delta is easy to scan at PR-review time.
 *
 * Set `LUXAR_UPDATE_SNAPSHOTS=1` to overwrite the snapshot files
 * instead of asserting equality. Inspect the diff before re-running.
 *
 * @module tests/e2e/tsl-codegen-snapshot.spec
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { test, expect, type Page } from '@playwright/test';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SNAPSHOT_DIR = path.resolve(__dirname, '../__codegen__');
const HARNESS_URL = '/tsl-harness.html';

interface TSLResult {
  pixels: number[];
  vertexShader: string;
  fragmentShader: string;
}

async function bootHarness(page: Page): Promise<void> {
  await page.goto(HARNESS_URL);
  await page.waitForFunction(() => Boolean(window.__tslHarness));
  await page.evaluate(() => window.__tslHarness!.ready);
}

async function runTSL(page: Page, shaderName: string): Promise<TSLResult> {
  return page.evaluate(async (name) => {
    const result = await window.__tslHarness!.renderTSL(name);
    return {
      pixels: Array.from(result.pixels),
      vertexShader: result.vertexShader,
      fragmentShader: result.fragmentShader,
    };
  }, shaderName);
}

/**
 * Cheap counts of expensive / control-flow tokens in a generated
 * shader. Returned as a sorted Markdown line for printing.
 */
function counters(label: string, src: string): string {
  const count = (re: RegExp) => (src.match(re) ?? []).length;
  // We deliberately use word-boundary regexes that match common
  // generated names; the false-positive rate is acceptable for
  // before-vs-after diff comparison.
  const pow = count(/\bpow\s*\(/g);
  const tan = count(/\btan\s*\(/g);
  const sqrtOrLen = count(/\b(sqrt|length)\s*\(/g);
  const tex = count(/\b(texture|textureSample|texture2D)\s*\(/g);
  // Generated GLSL declares varyings as "in float nodeVaryingN;" or
  // "flat  in float nodeVaryingN;". Generated WGSL uses
  // `@interpolate(...)` attributes. Match both styles.
  const varyings =
    count(/^\s*(?:flat\s+)?in\s+(?:highp\s+|mediump\s+|lowp\s+)?\w+\s+\w+\s*;/gm) +
    count(/@interpolate/g);
  const branches = count(/\bif\s*\(/g) + count(/\?\s*[^:]+:\s/g);
  return `${label}: pow=${pow} tan=${tan} sqrt/length=${sqrtOrLen} tex=${tex} varyings≈${varyings} branches=${branches}`;
}

function snapshotPath(shader: string, kind: 'vertex' | 'fragment'): string {
  return path.join(SNAPSHOT_DIR, `${shader}.${kind}.glsl.txt`);
}

/** Generated Three.js shaders carry indentation on otherwise blank lines. */
function normalizeSnapshot(source: string): string {
  return source.replace(/[ \t]+$/gm, '');
}

function assertSnapshot(shader: string, kind: 'vertex' | 'fragment', actual: string): void {
  const file = snapshotPath(shader, kind);
  const normalizedActual = normalizeSnapshot(actual);
  const isUpdate = process.env.LUXAR_UPDATE_SNAPSHOTS === '1';
  if (!fs.existsSync(file) || isUpdate) {
    fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
    fs.writeFileSync(file, normalizedActual);

    console.log(
      `  ${isUpdate ? 'updated' : 'created'} snapshot: ${path.relative(__dirname, file)}`
    );
    return;
  }
  const expected = normalizeSnapshot(fs.readFileSync(file, 'utf8'));
  expect(
    normalizedActual,
    `Generated ${kind} shader for "${shader}" differs from snapshot at ${path.relative(
      __dirname,
      file
    )}.\nReview the diff carefully — if intended, re-run with LUXAR_UPDATE_SNAPSHOTS=1 to overwrite.`
  ).toBe(expected);
}

// Shaders to snapshot. Includes all geometry kinds and their picking
// variants so attribute-packing changes (Float16 colours, etc.) have a
// regression gate across Points, Lines, and GSplats.
const SHADERS = [
  // Shared-math erf polynomial (materials/_shared/erf.ts): the snapshot
  // pins the LITERALS the TSL code generator emits for the coefficient
  // values — the textual half of the value-level parity contract (the
  // pixel half is the parity spec's 'erf' test).
  'erf',
  'line',
  'line-pick',
  'line-gamma-one',
  'line-no-gog',
  // Max-mode premultiplied RGB-contribution fragment (lines; GLSL twin:
  // LUXAR_MAX_RGB_CONTRIBUTION) — distinct generated code vs `line`.
  'line-max',
  // Emission–absorption output branch (lines, volumetric; GLSL twin:
  // LUXAR_VOLUMETRIC): τ = κ·alpha, S(τ) screening, the
  // w(a) per-endpoint-alpha map, and the color-discard bypass are
  // distinct generated code no other line variant pins.
  'line-volumetric',
  // COMBINED colormap + volumetric (lines): the LUT value path and the
  // w(a) alpha map both read the single unconditional texel5 fetch —
  // distinct generated code neither single-flag variant pins.
  'line-volumetric-colormap',
  // Lines are the only geometry whose ortho/perspective split is a
  // BUILD-time TSL option (points/gsplats branch on the uIsOrtho
  // uniform at runtime), so the perspective line shaders are distinct
  // generated code that the four ortho variants above never pin. The
  // `-behind` harness variants build with `isOrtho: false` — reuse
  // them to snapshot the perspective visual + pick branches.
  'line-behind',
  'line-pick-behind',
  'point',
  'point-pick',
  'point-gamma-one',
  // No-GOG fast path (points; GLSL twin: LUXAR_NO_GOG) — the GOG
  // mul/add/clamp chain drops from the generated code.
  'point-no-gog',
  // Max-mode premultiplied RGB-contribution fragment (points; GLSL twin:
  // LUXAR_MAX_RGB_CONTRIBUTION) — distinct generated code vs `point`.
  'point-max',
  // Emission–absorption output branch (points, volumetric phase 3; GLSL
  // twin: LUXAR_VOLUMETRIC): τ = κ·alpha, S(τ) screening, the
  // w(a) per-point-alpha map, and the color-discard bypass are distinct
  // generated code no other point variant pins.
  'point-volumetric',
  // COMBINED colormap + volumetric (points): LUT value path + w(a)
  // alpha map off the single texel2 fetch — distinct generated code
  // neither single-flag variant pins.
  'point-volumetric-colormap',
  'gsplat',
  'gsplat-pick',
  'gsplat-gamma-one',
  // No-GOG fast path (gsplats; GLSL twin: LUXAR_NO_GOG) — the GOG
  // mul/add/clamp chain drops from the generated code (the gain-aware
  // visibility discard keeps reading uIntensity).
  'gsplat-no-gog',
  // Blending-mode-specialized gsplat builds: `normal` emits the
  // premultiplied coverage-alpha fragment branch (peak projection,
  // GLSL twin: LUXAR_NORMAL_PREMULT), and the colormap variant emits
  // the LUT-lookup path (USE_COLORMAP). Both are distinct generated
  // code that the base `gsplat` snapshot never pins.
  'gsplat-normal-premult',
  // Pins the opaque→peak projection mapping (same graph family as max).
  'gsplat-opaque',
  // Emission–absorption output branch (sum projection, GLSL twin:
  // LUXAR_VOLUMETRIC): τ/α/S(τ) math + the color-discard bypass are
  // distinct generated code no other variant pins.
  'gsplat-volumetric',
  'gsplat-colormap',
  // Mesh — the six variants of MESH_NODE_SPEC.md §6.4.
  // `mesh` is the `opaque` DEFAULT (unlike the siblings, whose default is the
  // alpha-weighted `additive`), so it pins the hard alpha-cutout emission;
  // `mesh-additive` pins the alpha-weighted one the translucent modes share and
  // `mesh-max` the premultiplied one. `mesh-flat-normal` is the derivative-shaded
  // build, whose generated code must contain NEITHER the `normal` attribute nor its
  // varying, and `mesh-colormap` the LUT path.
  //
  // `mesh-pick` is ONE variant for every blending mode, which is itself the thing
  // being pinned: both mode-dependent behaviours (the `opaque` cutout and the
  // depth convention) are runtime uniforms rather than defines, so a layers-panel
  // mode switch is a uniform write. `mesh-pick-commutative` exercises the other arm
  // at RENDER time in the parity spec but generates byte-identical code, so it earns
  // no snapshot of its own — if it ever needs one, a build flag has crept back in.
  'mesh',
  'mesh-additive',
  'mesh-max',
  'mesh-flat-normal',
  'mesh-colormap',
  'mesh-pick',
] as const;

test.describe('TSL → generated-shader snapshots', () => {
  test('all geometry + pick variants: generated GLSL matches checked-in snapshots', async ({
    page,
  }) => {
    await bootHarness(page);

    for (const shader of SHADERS) {
      const result = await runTSL(page, shader);

      // Sanity: we MUST have recovered a non-empty shader string. If
      // empty, the NodeManager patch in tsl-harness.ts failed.
      expect(
        result.vertexShader.length,
        `Empty vertex shader for "${shader}" — NodeManager capture patch may have broken.`
      ).toBeGreaterThan(0);
      expect(
        result.fragmentShader.length,
        `Empty fragment shader for "${shader}" — NodeManager capture patch may have broken.`
      ).toBeGreaterThan(0);

      console.log(`\n${shader}:`);

      console.log(`  ${counters('vertex  ', result.vertexShader)}`);

      console.log(`  ${counters('fragment', result.fragmentShader)}`);

      assertSnapshot(shader, 'vertex', result.vertexShader);
      assertSnapshot(shader, 'fragment', result.fragmentShader);
    }
  });

  test('mesh-flat-normal strips the ENTIRE stored-normal path from both stages', async ({
    page,
  }) => {
    // The point of making the normal source a compile-time variant rather than a
    // runtime branch: the flat build must not merely skip the stored normal, it must
    // not CONTAIN it — no `normal` attribute, no varying carrying it, no epsilon
    // guard, no `gl_FrontFacing` flip. A runtime branch would leave all four in the
    // generated code (and keep the attribute in the vertex layout, which on WebGPU is
    // baked into the pipeline).
    //
    // The snapshots already pin this byte-for-byte, but only implicitly — a reviewer
    // reading a 111-line diff cannot see which absences are load-bearing. These
    // assertions name them.
    await bootHarness(page);
    const flat = await runTSL(page, 'mesh-flat-normal');
    const smooth = await runTSL(page, 'mesh');
    const both = (r: TSLResult) => `${r.vertexShader}\n${r.fragmentShader}`;

    for (const token of ['gl_FrontFacing', '1e-12']) {
      expect(both(smooth), `smooth build must contain ${token}`).toContain(token);
      expect(both(flat), `flat build must NOT contain ${token}`).not.toContain(token);
    }
    // The attribute itself: declared in the smooth vertex stage, absent in the flat one.
    expect(smooth.vertexShader).toMatch(/\bin\s+vec3\s+normal\s*;/);
    expect(flat.vertexShader).not.toMatch(/\bin\s+vec3\s+normal\s*;/);
  });

  test('mesh-pick keeps both mode behaviours as runtime uniforms, and binds no shading inputs', async ({
    page,
  }) => {
    // Three §6.5 properties the snapshot pins byte-for-byte but only implicitly.
    //
    // (1) ONE variant per mode. Both mode-dependent behaviours must appear as
    //     uniforms the generated code READS, not as absences a define produced —
    //     otherwise a mode switch would recompile the pick program mid-hover.
    // (2) The pick stage binds NEITHER `normal` NOR `aScalar`. They are shading
    //     inputs with no bearing on which vertex was clicked, and on WebGPU an
    //     attribute referenced by the graph is baked into the pipeline layout.
    // (3) The element id comes from the vertex-index BUILT-IN, never from
    //     `aSortedIndex` — mesh has no depth sort, so that attribute does not exist
    //     on the geometry and a reference to it would read garbage.
    await bootHarness(page);
    const pick = await runTSL(page, 'mesh-pick');
    const both = `${pick.vertexShader}\n${pick.fragmentShader}`;

    // (1) Stated as an EQUALITY rather than a token grep, because the generated code
    // names uniforms `nodeUniformN` — the JS-side names are gone, so grepping for
    // `uAlphaCutout` would prove nothing. Two registry entries that differ ONLY in
    // those two uniform values must generate the byte-identical shader. Turn either
    // selector into a build flag and this diverges immediately.
    const commutative = await runTSL(page, 'mesh-pick-commutative');
    expect(commutative.vertexShader).toBe(pick.vertexShader);
    expect(commutative.fragmentShader).toBe(pick.fragmentShader);
    // (2) No shading attributes.
    expect(pick.vertexShader).not.toMatch(/\bin\s+vec3\s+normal\s*;/);
    expect(pick.vertexShader).not.toMatch(/\baScalar\b/);
    // (3) The 16-bit split, off the built-in rather than an ordering attribute.
    expect(both).not.toMatch(/\baSortedIndex\b/);
    expect(both).toContain('65536');
  });
});
