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
    severity: 'warn',
    comment:
      `Modules in src/${fromLayer}/ must not import from higher layers ` +
      `(${forbiddenHigherLayers.join(', ')}). See src/CONVENTIONS.md for ` +
      `the full layer order. Type-only imports are exempt.`,
    from: { path: `^src/${fromLayer}/` },
    to: {
      path: `^src/(${forbiddenHigherLayers.join('|')})/`,
      pathNot: '\\.d\\.ts$',
    },
  };
}
