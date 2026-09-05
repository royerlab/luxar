/**
 * dependency-cruiser config — enforces the package layering documented
 * in `CONVENTIONS.md`:
 *
 *   types → config → cache → rendering → data → scene → input → ui → core
 *
 * (Note: `rendering` sits below `data` because rendering primitives —
 * materials, geometries, GPU buffer pools — are foundational
 * lower-level building blocks that the data layer assembles into
 * meshes.)
 *
 * Each layer may only import from layers to its left (plus `utils`,
 * `themes`, `wasm`, `workers`, `profiling`, `controls`, which are
 * cross-cutting and may be used anywhere).
 *
 * Type-only imports (`import type {...}`) are excluded — they're
 * erased at compile time so they don't create runtime coupling.
 *
 * The `no-circular` rule rounds out the structural guarantees at
 * severity `error` — any cycle should fail the build rather than
 * warn-and-be-ignored.
 *
 * (No `no-orphans` rule — the project's many type-only files would
 * dominate the report and require a brittle allowlist; a type-aware
 * tool like `ts-prune` is a better dead-code seam.)
 *
 * Run with `pnpm check:layers`. CI also runs this in pre-commit.
 */
// Keep this list empty unless a reviewed layer exception needs a clear
// owner and removal condition.
const KNOWN_LAYER_EXCEPTIONS = [];

module.exports = {
  forbidden: [
    {
      name: 'no-circular',
      severity: 'error',
      comment:
        'Circular dependencies are usually a smell — a leaf module ' +
        'requiring its consumer suggests the data flow is upside-down. ' +
        'Severity is `error`, so any new cycle fails the build.',
      from: {},
      to: { circular: true },
    },
    {
      name: 'input-private-modules',
      severity: 'error',
      comment:
        'Modules outside src/input/ must import through src/input/index.ts, ' +
        'not reach into the private input/input-handler/ implementation tree.',
      from: { path: '^src/(?!input/)' },
      to: { path: '^src/input/input-handler(?:\\.ts|/)' },
    },
    {
      name: 'input-no-data-runtime',
      severity: 'error',
      comment:
        'The input package owns interaction wiring, not data loading. ' +
        'Runtime loading orchestration belongs in the scene or data layer.',
      from: { path: '^src/input/' },
      to: { path: '^src/data/' },
    },
    // ─── Layer order ──────────────────────────────────────────────
    // Each rule says: "if `from` matches src/X/, `to` must NOT match
    // src/{higher_layer}/". dependencyTypes filter excludes type-only
    // imports (TS `import type {...}`) so re-exports don't trigger.
    //
    // Severity is `error` — the build fails on any new violation.
    // Paths listed in KNOWN_LAYER_EXCEPTIONS are exempt; add a
    // matching warn-only rule below for each reviewed exception.
    layerRule('types', ['config', 'cache', 'rendering', 'data', 'scene', 'input', 'ui', 'core']),
    layerRule('config', ['cache', 'rendering', 'data', 'scene', 'input', 'ui', 'core']),
    layerRule('cache', ['rendering', 'data', 'scene', 'input', 'ui', 'core']),
    layerRule('rendering', ['data', 'scene', 'input', 'ui', 'core']),
    layerRule('data', ['scene', 'input', 'ui', 'core']),
    layerRule('scene', ['input', 'ui', 'core']),
    layerRule('input', ['ui', 'core']),
    layerRule('ui', ['core']),

    // No per-file warn rules are active. If KNOWN_LAYER_EXCEPTIONS
    // gains an entry, add the corresponding warn-only rule here.
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
 * any new violation. Paths in KNOWN_LAYER_EXCEPTIONS are skipped by
 * this rule and should have a matching warn-only rule above.
 */
function layerRule(fromLayer, forbiddenHigherLayers) {
  const exceptions = KNOWN_LAYER_EXCEPTIONS.map((p) => `^${p}$`).join('|');
  return {
    name: `layer-${fromLayer}-no-upward`,
    severity: 'error',
    comment:
      `Modules in src/${fromLayer}/ must not import from higher layers ` +
      `(${forbiddenHigherLayers.join(', ')}). See src/CONVENTIONS.md ` +
      'for the full layer order. Type-only imports are exempt.',
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
