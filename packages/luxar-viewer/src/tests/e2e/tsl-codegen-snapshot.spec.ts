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
 *   1. **Regression detection.** If the generated code diverges from
 *      the snapshot (e.g. a supposedly equivalent change adds an `if`
 *      branch or duplicates a `pow()`), the diff is the alarm. Note
 *      WHERE that alarm rings: this spec belongs to the FULL E2E suite
 *      (`pnpm test:e2e` / `make test-e2e`), not to CI's PR checks — the
 *      `e2e-tests` job in `.github/workflows/ci.yml` is disabled
 *      (`if: false`), and even re-enabled it runs only the five
 *      interaction specs of `pnpm test:e2e:smoke`, which do not include
 *      this file. So a TSL change is gated by whoever runs the full
 *      suite before merging, not by the commit itself.
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

/** One line of a generated `main()` body, with its brace depth inside that body. */
interface FlowLine {
  text: string;
  /** 0 = directly inside `main()`; >= 1 = inside an `if` / `else` block. */
  depth: number;
  /** 1-indexed line number in the generated source, for failure messages. */
  lineNumber: number;
}

/**
 * Splits a generated shader into the lines of its `main()` body, each tagged
 * with the brace depth it sits at. Brace counting is per line, which is exact
 * for Three's generated code (it puts every block brace on its own line).
 */
function mainFlowLines(source: string): FlowLine[] {
  const lines = source.split('\n');
  const mainIndex = lines.findIndex((line) => /^\s*void\s+main\s*\(\s*\)\s*\{/.test(line));
  if (mainIndex < 0) return [];

  const flow: FlowLine[] = [];
  let depth = 0;
  for (let i = mainIndex + 1; i < lines.length; i++) {
    const text = lines[i];
    flow.push({ text, depth, lineNumber: i + 1 });
    depth += (text.match(/\{/g) ?? []).length - (text.match(/\}/g) ?? []).length;
    if (depth < 0) break; // the closing brace of main()
  }
  return flow;
}

/**
 * Asserts that each named variable of a generated fragment shader is ASSIGNED in
 * the unconditional top-level flow of `main()`, before any line that reads it.
 *
 * The defect class this catches: `colorNode` and `depthNode` are INDEPENDENT
 * NodeMaterial entry points, and a value shared between them is assigned wherever
 * three first BUILDS it. Three re-hoists such an assignment into another
 * conditional block when the later reader is itself inside a block, but NOT when
 * that reader sits at the top level of a flow. So when a BRANCHING `depthNode`
 * becomes the first build site — the emission order of the two entry points is not
 * part of three's API, and it flipped from colour-first to depth-first between r184
 * and r185 — the shared chain is assigned inside an `if`/`else` arm while every
 * top-level reader gets an unassigned variable, i.e. 0. In these pick shaders that
 * means the discard conditions fire for every fragment and the pick buffer comes
 * back empty.
 *
 * Keyed on the explicit `.toVar('name')` names the pick factories declare, so it is
 * independent of the generated `nodeVarN` numbering.
 *
 * Two assertions, deliberately kept separate because they have different reach:
 *   - EVERY assignment sits at brace depth 0 — the dead declaration initialiser
 *     included.
 *   - the first READ comes after the first REAL assignment, where "real" excludes
 *     that dead initialiser. The distinction is what makes this half able to fail
 *     at all: a var declared `float(0.0).toVar(name)` makes three emit a
 *     `NAME = 0.0;` line at the var's first build site, which matches the
 *     assignment regex, so comparing the first read against the first assignment
 *     of ANY kind is an identity no defect can break.
 *
 * A declaration with no initializer is NOT counted as a reference: generated GLSL
 * hoists `float NAME;` above `main()` (so it never reaches this helper), but WGSL
 * declares `var NAME : f32;` inside the entry function, and counting that as the first
 * read would fail the helper on correct code. Assignment lines are not counted as
 * reads either — a write is not a read, and the dead initialiser is nothing but one.
 */
function assertAssignedInUnconditionalFlow(
  shader: string,
  fragmentShader: string,
  varNames: readonly string[]
): void {
  const flow = mainFlowLines(fragmentShader);
  expect(
    flow.length,
    `no main() found in the generated "${shader}" fragment shader`
  ).toBeGreaterThan(0);

  for (const name of varNames) {
    // An optional leading type token covers a `float name = …` declaration form;
    // the trailing `[^=]` keeps a comparison (`name == x`) from reading as one. A
    // COMPOUND assignment (`name += …`, `-=`, `*=`, `/=`) counts as an assignment too,
    // so a future one inside a branch fails the depth-0 assertion below instead of
    // slipping past it and resurfacing as a confusing "read before assignment".
    const assignment = new RegExp(`^\\s*(?:const\\s+)?(?:\\w+\\s+)?${name}\\s*(?:[-+*/])?=[^=]`);
    const reference = new RegExp(`\\b${name}\\b`);
    // `float name;` (GLSL) / `var name : f32;` (WGSL) — a declaration, not a read.
    const declarationOnly = new RegExp(
      `^\\s*(?:var\\s+)?(?:\\w+\\s+)?${name}\\s*(?::\\s*\\w+\\s*)?;\\s*$`
    );
    // The DEAD INITIALISER: because these vars are declared `float(0.0).toVar(name)`
    // / `bool(false).toVar(name)`, three emits `NAME = 0.0;` / `NAME = false;` at the
    // var's first build site. It writes the declaration DEFAULT, not the value the
    // prologue computes, so a reader that follows it is still reading nothing. (A real
    // assignment whose RHS is literally `0.0` would be misclassified here; none of the
    // pinned shared values has one — every one assigns an expression.)
    const deadInitialiser = new RegExp(
      `^\\s*(?:var\\s+)?(?:\\w+\\s+)?${name}\\s*(?::\\s*\\w+\\s*)?=\\s*(?:0\\.0|0|false)\\s*;\\s*$`
    );

    const assignments = flow.filter((line) => assignment.test(line.text));
    const realAssignments = assignments.filter((line) => !deadInitialiser.test(line.text));
    expect(
      realAssignments.length,
      `${shader}: "${name}" is never assigned inside main() — the shared fragment value is gone, ` +
        'was renamed, or is left holding its declaration default.'
    ).toBeGreaterThan(0);

    for (const line of assignments) {
      expect(
        line.depth,
        `${shader}: "${name}" is assigned at brace depth ${line.depth} (line ${line.lineNumber}: ` +
          `"${line.text.trim()}"). A value shared between colorNode and depthNode must be ` +
          'assigned in unconditional top-level flow — see the fragment prologue in the factory. ' +
          'This rule is deliberately stronger than strictly necessary: assigning at top level ' +
          'and then REFINING inside a branch is legitimate and already ships elsewhere ' +
          '(`profile` in picking/line/pick-capsule.tsl.ts), but for a value shared across two ' +
          'entry points the guard demands the stronger form, because which flow gets built ' +
          'first is not under our control.'
      ).toBe(0);
    }

    const firstRead = flow.find(
      (line) =>
        reference.test(line.text) && !declarationOnly.test(line.text) && !assignment.test(line.text)
    );
    expect(
      firstRead,
      `${shader}: "${name}" is assigned but never read inside main() — a reader was removed or renamed.`
    ).toBeTruthy();
    expect(
      firstRead?.lineNumber ?? 0,
      `${shader}: "${name}" is READ at line ${firstRead?.lineNumber} (` +
        `"${firstRead?.text.trim()}") before the fragment prologue assigns it at line ` +
        `${realAssignments[0].lineNumber} — so that read sees the declaration default ` +
        '(0 / false), not the computed value. Either the prologue call was dropped from this ' +
        'entry point, or a reader was moved above it.'
    ).toBeGreaterThan(realAssignments[0].lineNumber);
  }
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
  'line',
  'line-pick',
  'line-gamma-one',
  'line-no-gog',
  // Max-mode premultiplied RGB-contribution fragment (lines; GLSL twin:
  // LUXAR_MAX_RGB_CONTRIBUTION) — distinct generated code vs `line`.
  'line-max',
  // Opaque contribution-cutout branch (lines; GLSL twin:
  // LUXAR_OPAQUE_RGB_CONTRIBUTION) — distinct generated code no other
  // line variant pins.
  'line-opaque',
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
  // Opaque contribution-cutout branch (points; GLSL twin:
  // LUXAR_OPAQUE_RGB_CONTRIBUTION) — distinct generated code no other
  // point variant pins.
  'point-opaque',
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
  //
  // The five perspective near-fade entries — `mesh-near-fade`,
  // `mesh-additive-near-fade`, `mesh-pick-near-fade` and the two
  // `*-near-fade-reference` un-faded twins — are absent for the same reason: they
  // differ from `mesh` / `mesh-additive` / `mesh-pick` only in `uIsOrtho` and
  // `uNearCull`, both runtime uniforms, so they generate byte-identical code. (The
  // near fade is therefore pinned textually by the `mesh`, `mesh-additive` and
  // `mesh-pick` snapshots below, and by pixels in the parity spec's perspective
  // entries — which is where the two FOLDS, RGB ramp vs coverage multiply, are told
  // apart. A snapshot shows the line; only a rendered frame shows it landing.)
  'mesh',
  'mesh-additive',
  'mesh-max',
  'mesh-flat-normal',
  'mesh-colormap',
  'mesh-texture',
  'mesh-none-shading',
  'mesh-pick',
  'mesh-pick-texture',
  // Shared math. This pins the LITERALS the TSL code generator emits —
  // the textual half of the module's value-level parity contract (the
  // pixel half is the parity spec's same-named test):
  //   'erf'               materials/_shared/erf.ts, the polynomial coefficients
  // IT MUST STAY AHEAD OF ANY FUTURE GEOMETRY ENTRY (it adds no
  // `render`-group member): the shared `render` std140 uniform group
  // accumulates members in first-encounter order across the whole run,
  // so an entry built BEFORE the geometry shaders reorders
  // `cameraViewMatrix` / `cameraProjectionMatrix` in every subsequent
  // snapshot (49 files of spurious churn when 'erf' briefly led this list).
  'erf',
  // Capsule line primitive (#1352, ?linePrimitive=capsule) — one entry per
  // distinct GRAPH: ortho additive (sideon; joint/fold/taper/fat share its
  // code), perspective (near-clip + fade branches), the max and volumetric
  // mode tails, the colormap fragment, and the pick twin.
  'line-capsule-sideon',
  'line-capsule-endon-persp',
  'line-capsule-max',
  'line-capsule-volumetric',
  'line-capsule-colormap',
  'line-capsule-pick-sideon',
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

  test('mesh-none-shading strips the whole lighting path, derivatives included', async ({
    page,
  }) => {
    // Stronger than `mesh-flat-normal`'s claim, and the reason `shading` became a
    // 3-valued enum rather than gaining a second boolean: the unlit build must not
    // contain the lighting at all. Not the stored normal (which `flat` also drops),
    // and not the DERIVATIVE normal either — `flat` keeps that one, so this is the
    // only variant with no normal of any kind.
    //
    // The distinction matters because the obvious implementation is `shade = 1.0`,
    // which computes both derivatives, normalizes, evaluates a pow() for the wrap
    // term and another for the specular, then multiplies by one. It renders
    // identically and costs all of that per fragment.
    await bootHarness(page);
    const unlit = await runTSL(page, 'mesh-none-shading');
    const flat = await runTSL(page, 'mesh-flat-normal');
    const smooth = await runTSL(page, 'mesh');
    const both = (r: TSLResult) => `${r.vertexShader}\n${r.fragmentShader}`;

    // The derivative pair: present in BOTH lit builds, absent here. Checked against
    // `flat` as well as `smooth` so this cannot pass by accidentally re-testing what
    // the flat-normal test above already covers.
    for (const token of ['dFdx', 'dFdy']) {
      expect(both(smooth), `smooth build must contain ${token}`).toContain(token);
      expect(both(flat), `flat build must contain ${token}`).toContain(token);
      expect(both(unlit), `unlit build must NOT contain ${token}`).not.toContain(token);
    }
    // No normal attribute in the vertex layout, which on WebGPU is baked into the
    // pipeline at first draw.
    expect(unlit.vertexShader).not.toMatch(/\bin\s+vec3\s+normal\s*;/);
    // And the stored-normal machinery, as in the flat test.
    for (const token of ['gl_FrontFacing', '1e-12']) {
      expect(both(unlit), `unlit build must NOT contain ${token}`).not.toContain(token);
    }
  });

  test('mesh-texture samples per FRAGMENT, and only the textured builds bind uv', async ({
    page,
  }) => {
    // The structural claim the whole feature rests on. A colormap LUT is sampled in
    // the VERTEX stage (one scalar per vertex, so interpolating the resulting colour
    // approximates interpolating the scalar); an image has structure BETWEEN
    // vertices, so it must be sampled in the FRAGMENT stage or the mesh resolves
    // exactly one texel per vertex — reproducing the point-cloud limitation this
    // replaces.
    //
    // Asserted as a comparison against `mesh-colormap` rather than in isolation,
    // because "the fragment stage has a sampler" is only meaningful next to a build
    // where the sampler is in the other stage.
    await bootHarness(page);
    const textured = await runTSL(page, 'mesh-texture');
    const colormap = await runTSL(page, 'mesh-colormap');
    const plain = await runTSL(page, 'mesh');

    // Every visual fragment stage carries exactly ONE sampler of its own since the
    // refraction split's glass-depth partition (glass-partition-tsl.ts): the plain
    // build is that baseline, and the colour-source claims are made relative to it.
    const samples = (src: string): number => src.split('texture(').length - 1;
    const baseline = samples(plain.fragmentShader);
    expect(baseline).toBe(1);
    // Colormap: the LUT sampler is in the VERTEX stage, none added to the fragment.
    expect(colormap.vertexShader).toContain('texture(');
    expect(samples(colormap.fragmentShader)).toBe(baseline);
    // Texture: the other way round — one more sampler in the fragment.
    expect(samples(textured.fragmentShader)).toBe(baseline + 1);
    // `uv` enters the vertex layout ONLY for the textured build — an attribute that
    // appears later is silently broken on WebGPU.
    expect(textured.vertexShader).toMatch(/\bin\s+vec2\s+uv\s*;/);
    expect(plain.vertexShader).not.toMatch(/\bin\s+vec2\s+uv\s*;/);
    expect(colormap.vertexShader).not.toMatch(/\bin\s+vec2\s+uv\s*;/);
  });

  test('mesh-pick-texture samples the texture too, so cutout holes are unpickable', async ({
    page,
  }) => {
    // Texture alpha multiplies coverage in the visual shader, so an RGBA basemap's
    // transparent regions are real holes on screen. A pick pass without the sampler
    // would leave them pickable AND depth-occluding — a divergence no amount of
    // `syncMeshPickAppearance` can fix, because it is not an appearance uniform.
    await bootHarness(page);
    const pickTextured = await runTSL(page, 'mesh-pick-texture');
    const pickPlain = await runTSL(page, 'mesh-pick');

    expect(pickTextured.fragmentShader).toContain('texture(');
    expect(pickPlain.fragmentShader).not.toContain('texture(');
    expect(pickTextured.vertexShader).toMatch(/\bin\s+vec2\s+uv\s*;/);
    expect(pickPlain.vertexShader).not.toMatch(/\bin\s+vec2\s+uv\s*;/);
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

  test('pick shaders assign every named shared fragment value in unconditional flow', async ({
    page,
  }) => {
    // The gsplat and mesh factories have a BRANCHING `depthNode` (the `uSurfaceDepth`
    // convention selector), and mesh has a second branch one level deeper (the
    // `uAlphaCutout` arm of the brightness select). Three lowers each to a real
    // `if`/`else`, so a shared fragment value that is first BUILT there is assigned
    // inside an arm — and three only re-hoists such an assignment for readers that are
    // themselves inside a block. Every top-level reader then sees 0: the pick pass
    // discards every fragment and the buffer comes back empty (which is exactly what
    // r185 did before the shared values were moved into an unconditional prologue).
    //
    // The line and capsule-line pick factories are here for the opposite reason: their
    // `depthNode` is branchless TODAY, which is what lets them keep a bare shared
    // `.toVar()` with no prologue, and each one says so at the site. This holds them
    // to it — add a branch there and the shared assignment moves into an arm.
    //
    // The snapshots pin this only implicitly — the difference is one indentation level
    // inside a 150-line file, which is the last thing a reviewer notices. This names it.
    //
    // HONESTY NOTE — the two halves have different reach, and moving the runtime pin from
    // r184 to r185 FLIPPED which one is live rather than making both so.
    //
    // At the pinned r185 the DEPTH flow is built first, and its first statement is the
    // `fragmentPrologue()` call, so every shared value is assigned at brace depth 0 at the
    // top of `main()`. The BRACE-DEPTH half is the live one here: revert the prologue and
    // the chain lands inside the `uSurfaceDepth` `else` arm, which is exactly the
    // zero-pixel pick pass of issue #1683. The READ-ORDER half cannot fail from anything a
    // reader does in `colorNode`, because the prologue already ran before that flow is
    // emitted at all; it still fires for a reader hoisted above the prologue call inside
    // the depth flow itself.
    //
    // Under r184 it was the mirror image — colour-first emission put even the pre-fix
    // free-standing `.toVar()`s at top level, so the brace-depth half was the inert one and
    // the read-order half caught a missing or late prologue call in `colorNode`.
    //
    // A consequence worth stating rather than discovering: at r185 `colorNode`'s own
    // `fragmentPrologue()` call is verified by NOTHING — remove it and the generated GLSL
    // is byte-identical, so neither this guard, the snapshots, nor the pixel-parity spec
    // move. It is kept deliberately, as the insurance that makes both entry points
    // self-sufficient whichever one a future three builds first. Do not delete it as dead.
    await bootHarness(page);

    const shared = [
      {
        shader: 'gsplat-pick',
        vars: ['gsplatPickMahalSq', 'gsplatPickIntensity', 'gsplatPickBrightness'],
      },
      {
        shader: 'mesh-pick',
        vars: ['meshPickCoverage', 'meshPickNearFade', 'meshPickCutout', 'meshPickBrightness'],
      },
      // These two are declared `brightnessShared().toVar(name)` — the RHS is the real
      // computation, so there is no dead initialiser at all and the read-order half is
      // fully live for them, not just for a dropped prologue call.
      { shader: 'line-pick', vars: ['lineBrightness'] },
      { shader: 'line-capsule-pick-sideon', vars: ['lineCapsulePickBrightness'] },
      // `point-pick` is deliberately absent: its shared brightness is an UNNAMED
      // `.toVar()`, so it surfaces as `nodeVarN` in the generated code and there is no
      // stable name to key on. Naming it there would let it join this list.
    ] as const;

    for (const { shader, vars } of shared) {
      const result = await runTSL(page, shader);
      assertAssignedInUnconditionalFlow(shader, result.fragmentShader, vars);
    }
  });
});
