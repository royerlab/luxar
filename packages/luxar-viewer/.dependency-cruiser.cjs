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
 * No-circular and no-orphans rules round out the structural
 * guarantees.
 *
 * Run with `pnpm check:layers`. CI also runs this in pre-commit.
 */
// Files known to contain pre-existing layer violations. Listed by
// path so the layer rules below can downgrade `severity` for these
// specific edges to `warn` while everything else hits `error`.
//
// These two crossings need bigger refactors than the Phase 8.6 bus
// pattern delivered (the bus handles toggle commands and FPS pubs;
// these are construction ownership and granular provider wiring).
// Cleared in a future pass — at which point the entry is deleted
// from this list.
const KNOWN_LAYER_EXCEPTIONS = [
  // input-handler still constructs DimensionSliders directly. Needs
  // ownership migration to ui/ (similar to the Phase 8.6.b/1 +
  // 8.6.c PerformanceMonitor / DebugConsole moves).
  'src/input/input-handler.ts',
  // scene-loader pushes granular providers into DataMonitorManager
  // (cache stats, accumulators, profiler, scene graph). Needs an
  // event-bus migration of those providers.
  'src/data/scene-loader.ts',
];

module.exports = {
  forbidden: [
    {
      name: 'no-circular',
      severity: 'warn',
      comment:
        'Circular dependencies are usually a smell — a leaf module ' +
        'requiring its consumer suggests the data flow is upside-down. ' +
        'Severity is `warn` while we work down the existing list.',
      from: {},
      to: { circular: true },
    },
    // ─── Layer order ──────────────────────────────────────────────
    // Each rule says: "if `from` matches src/X/, `to` must NOT match
    // src/{higher_layer}/". dependencyTypes filter excludes type-only
    // imports (TS `import type {...}`) so re-exports don't trigger.
    //
    // Severity is `error` — the build fails on any new violation. The
    // two known exceptions in KNOWN_LAYER_EXCEPTIONS are pinned to
    // `warn` via the per-rule `pathNot` filter on `from`.
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

    // Per-file warn for the two known exceptions. Same upward
    // restrictions as the strict rules (input → ui|core; data →
    // scene|input|ui|core), but at warn severity so the build stays
    // green while we work the cleanup backlog.
    {
      name: 'layer-input-known-exception',
      severity: 'warn',
      comment:
        'Pre-existing input → ui violation — input-handler still ' +
        'constructs DimensionSliders directly. See KNOWN_LAYER_EXCEPTIONS.',
      from: { path: '^src/input/input-handler\\.ts$' },
      to: { path: '^src/(ui|core)/', pathNot: '\\.d\\.ts$' },
    },
    {
      name: 'layer-data-known-exception',
      severity: 'warn',
      comment:
        'Pre-existing data → ui violation — scene-loader pushes ' +
        'granular providers into DataMonitorManager. See ' +
        'KNOWN_LAYER_EXCEPTIONS.',
      from: { path: '^src/data/scene-loader\\.ts$' },
      to: { path: '^src/(scene|input|ui|core)/', pathNot: '\\.d\\.ts$' },
    },
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
 * `forbiddenHigherLayers`. Severity defaults to `warn` — preexisting
 * violations stay surfaced without breaking the build, and we ratchet
 * to `error` after each cleanup pass.
 */
function layerRule(fromLayer, forbiddenHigherLayers) {
  return {
    name: `layer-${fromLayer}-no-upward`,
    severity: 'error',
    comment:
      `Modules in src/${fromLayer}/ must not import from higher layers ` +
      `(${forbiddenHigherLayers.join(', ')}). See src/CONVENTIONS.md ` +
      `for the full layer order. Type-only imports are exempt. ` +
      `Pre-existing violations are pinned to 'warn' via the ` +
      `layer-known-exception rule below.`,
    from: {
      path: `^src/${fromLayer}/`,
      // Don't fire on the two paths still working through the cleanup
      // backlog. They get caught by the warn-only known-exception rule.
      pathNot: KNOWN_LAYER_EXCEPTIONS.map((p) => `^${p}$`).join('|'),
    },
    to: {
      path: `^src/(${forbiddenHigherLayers.join('|')})/`,
      pathNot: '\\.d\\.ts$',
    },
  };
}
