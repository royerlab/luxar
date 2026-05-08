/**
 * dependency-cruiser config — enforces the package layering documented
 * in `src/CONVENTIONS.md`:
 *
 *   types → config → cache → rendering → data → scene → input → ui → core
 *
 * (Note: `rendering` sits BELOW `data` because rendering primitives —
 * materials, geometries, GPU buffer pools — are foundational
 * lower-level building blocks that the data layer assembles into
 * meshes. The original plan had data before rendering; this order
 * matches the actual dependency direction in the codebase.)
 *
 * Each layer may only import from layers to its left (plus `utils`,
 * `themes`, `wasm`, `workers`, `profiling`, `controls`, which are
 * cross-cutting and may be used anywhere).
 *
 * Type-only imports (`import type {...}`) are excluded — they're
 * erased at compile time so they don't create runtime coupling.
 *
 * The `no-circular` rule rounds out the structural guarantees and
 * is now severity `error` (Phase 13.16) — zero cycles existed at
 * the time of the ratchet, and any future cycle should fail the
 * build rather than warn-and-be-ignored.
 *
 * (No `no-orphans` rule — the project's many type-only files would
 * dominate the report and require a brittle allowlist; a type-aware
 * tool like `ts-prune` is a better dead-code seam.)
 *
 * Run with `pnpm check:layers`. CI also runs this in pre-commit.
 */
// All known pre-existing layer violations have been cleared. The list
// is kept for the rare case a future regression needs a temporary
// downgrade while a follow-up commit lands. Add a path here, drop the
// entry once the underlying coupling is cured.
const KNOWN_LAYER_EXCEPTIONS = [];

module.exports = {
  forbidden: [
    {
      name: 'no-circular',
      severity: 'error',
      comment:
        'Circular dependencies are usually a smell — a leaf module ' +
        'requiring its consumer suggests the data flow is upside-down. ' +
        'Phase 13.16: ratcheted from warn to error after a clean ' +
        '`pnpm check:layers --output-type json` showed zero cycles. ' +
        'Any new cycle now fails the build.',
      from: {},
      to: { circular: true },
    },
    // ─── Layer order ──────────────────────────────────────────────
    // Each rule says: "if `from` matches src/X/, `to` must NOT match
    // src/{higher_layer}/". dependencyTypes filter excludes type-only
    // imports (TS `import type {...}`) so re-exports don't trigger.
    //
    // Severity is `error` — the build fails on any new violation. The
    // Any path listed in KNOWN_LAYER_EXCEPTIONS is exempt — wire a
    // corresponding warn-only rule below if you add one. Currently
    // empty: every layer crossing has been resolved.
    layerRule('types', [
      'config',
      'cache',
      'rendering',
      'data',
      'scene',
      'input',
      'ui',
      'core',
    ]),
    layerRule('config', ['cache', 'rendering', 'data', 'scene', 'input', 'ui', 'core']),
    layerRule('cache', ['rendering', 'data', 'scene', 'input', 'ui', 'core']),
    layerRule('rendering', ['data', 'scene', 'input', 'ui', 'core']),
    layerRule('data', ['scene', 'input', 'ui', 'core']),
    layerRule('scene', ['input', 'ui', 'core']),
    layerRule('input', ['ui', 'core']),
    layerRule('ui', ['core']),

    // Per-file warn rules previously here have been removed —
    // input-handler → ui (Phase 8.6.d) and scene-loader → ui (Phase
    // 8.6.e) are now properly dependency-inverted. To re-enable a
    // pinpoint warn rule for a future regression, list the file in
    // KNOWN_LAYER_EXCEPTIONS and add a corresponding rule here.
  ],
  options: {
    doNotFollow: { path: 'node_modules' },
    exclude: {
      path: '(\\.test\\.tsx?$|tests/|/dist/|node_modules)',
    },
    tsConfig: { fileName: './tsconfig.json' },
    tsPreCompilationDeps: false, // Ignore type-only imports
    enhancedResolveOptions: {
      exportsFields: ['exports'],
      conditionNames: ['import', 'require', 'node', 'default'],
    },
    reporterOptions: {
      text: { highlightFocused: true },
    },
  },
};

/**
 * Build a forbidden rule banning imports from `fromLayer` to any of
 * `forbiddenHigherLayers`. Severity is `error` — the build fails on
 * any new violation. Add a path to KNOWN_LAYER_EXCEPTIONS and a
 * matching warn-only rule above to temporarily downgrade a specific
 * file while a follow-up commit lands the fix.
 */
function layerRule(fromLayer, forbiddenHigherLayers) {
  const exceptions = KNOWN_LAYER_EXCEPTIONS.map((p) => `^${p}$`).join('|');
  return {
    name: `layer-${fromLayer}-no-upward`,
    severity: 'error',
    comment:
      `Modules in src/${fromLayer}/ must not import from higher layers ` +
      `(${forbiddenHigherLayers.join(', ')}). See src/CONVENTIONS.md ` +
      `for the full layer order. Type-only imports are exempt.`,
    from: {
      path: `^src/${fromLayer}/`,
      // Empty when KNOWN_LAYER_EXCEPTIONS is empty — depcruiser treats
      // an empty regex as "match nothing", which is what we want.
      ...(exceptions ? { pathNot: exceptions } : {}),
    },
    to: {
      path: `^src/(${forbiddenHigherLayers.join('|')})/`,
      pathNot: '\\.d\\.ts$',
    },
  };
}
