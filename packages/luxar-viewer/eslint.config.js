import js from '@eslint/js';
import typescript from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import globals from 'globals';

export default [
  js.configs.recommended,
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      parser: typescriptParser,
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
      },
      parserOptions: {
        project: './tsconfig.json',
      },
    },
    plugins: {
      '@typescript-eslint': typescript,
    },
    rules: {
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      'semi': ['error', 'always'],
      'quotes': ['error', 'single', { avoidEscape: true }],
      'indent': 'off', // Handled by prettier — eslint indent conflicts with prettier ternary formatting
      'no-undef': 'off', // TypeScript handles this
      // Channel all logging through src/utils/log.ts. The two exceptions
      // (utils/log.ts itself and console-interceptor.ts which monkey-patches
      // console.*) opt out via the per-file override below.
      'no-console': 'error',
      // Keep `three/webgpu` and `three/tsl` off the eager critical path.
      //
      // These two subpaths pull the three.js node system — ~173 kB gzipped as
      // its own chunk — and the production default backend is WebGL
      // (`selectBackend()` in renderer-setup.ts). A single value import from a
      // module the entry point reaches is enough to pin that chunk into the
      // eager graph, which is what shipped for a long time and cost every
      // WebGL user a renderer they never ran (issue #1679).
      //
      // The whole cone now hangs off ONE dynamic import in
      // `rendering/tsl/load.ts`; the `*-tsl` / `*.tsl` modules that
      // `rendering/tsl/registry.ts` gathers are the only production modules
      // allowed to name these specifiers directly (see the override below).
      // Everything else reaches the classes through `rendering/tsl/slot.ts`.
      //
      // `allowTypeImports` keeps `import type { WebGPURenderer }` legal —
      // types are erased and cost nothing (renderer-capabilities.ts relies on
      // this, and documents why).
      //
      // NOTE: this deliberately lives in ESLint rather than
      // `.dependency-cruiser.cjs`. That config excludes `node_modules` from
      // the graph entirely, so a dep-cruiser rule targeting a third-party
      // specifier can never fire — it would look like a gate and enforce
      // nothing.
      '@typescript-eslint/no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: 'three/webgpu',
              message:
                'Import three/webgpu only inside the lazy cone rooted at ' +
                'src/rendering/tsl/registry.ts — that is, from the *-tsl / *.tsl modules it ' +
                'gathers, not from registry.ts itself. Elsewhere, get the ' +
                'class from requireTslMaterials() in rendering/tsl/slot.ts (loaded by ' +
                'loadTslMaterials()), or use `import type` if you only need the type. See #1679.',
              allowTypeImports: true,
            },
            {
              name: 'three/tsl',
              message:
                'Import three/tsl only from the *-tsl / *.tsl modules inside the lazy cone ' +
                'rooted at src/rendering/tsl/registry.ts. A value import from anywhere the ' +
                'entry point reaches eagerly pulls the ' +
                'three-webgpu chunk into the initial payload. See #1679.',
              allowTypeImports: true,
            },
          ],
        },
      ],
    },
  },
  {
    // The lazy cone itself. `registry.ts` is the designated entry to the cone
    // and imports neither subpath itself; the `*-tsl` / `*.tsl` modules it pulls
    // in are the TSL implementations and necessarily name `three/webgpu` /
    // `three/tsl` and `NodeMaterial`.
    // They are reachable ONLY through registry.ts's dynamic-import boundary,
    // which is the property that actually matters and is verified against the
    // built bundle by `scripts/check-eager-chunks.mjs`.
    // `*-tsl-*.ts` is not redundant with `*-tsl.ts`: the capsule variants
    // (`shader-tsl-capsule.ts`) match only the former, and were caught by this
    // rule the first time it ran.
    files: [
      'src/rendering/tsl/registry.ts',
      'src/rendering/**/*-tsl.ts',
      'src/rendering/**/*-tsl-*.ts',
      'src/rendering/**/*.tsl.ts',
      'src/rendering/materials/_shared/tsl-helpers.ts',
      'src/rendering/materials/_shared/erf-tsl.ts',
    ],
    rules: {
      '@typescript-eslint/no-restricted-imports': 'off',
    },
  },
  {
    // Tests and the TSL↔GLSL parity harness are not shipped, so they cannot
    // affect the initial payload — the thing this rule protects. The harness in
    // particular exists to drive the WebGPU path directly.
    files: ['src/tests/**/*.{ts,tsx}'],
    rules: {
      '@typescript-eslint/no-restricted-imports': 'off',
    },
  },
  {
    // log.ts wraps console.*. console-interceptor.ts intentionally
    // monkey-patches console.log/warn/error/info/debug for the debug-console
    // overlay. Both are the legitimate exception sites.
    files: ['src/utils/log.ts', 'src/utils/console-interceptor.ts'],
    rules: {
      'no-console': 'off',
    },
  },
  {
    // Tests, benchmarks, screenshot drivers and mocks are tooling — they
    // legitimately use console.* for diagnostic output that doesn't need
    // to flow through the in-app debug console.
    files: [
      'src/tests/**/*.ts',
      'src/tests/**/*.tsx',
      'src/**/*.test.ts',
      'src/**/*.spec.ts',
    ],
    rules: {
      'no-console': 'off',
    },
  },
  {
    // Production code (excluding tests) — warn on `any` usage to nudge new
    // code toward typed alternatives without forcing a sweep of every existing
    // site. Browser-API gaps (Float16Array, OPFS keys/entries, performance.memory),
    // three.js library internals (bloomEffect private fields, composer.multisampling),
    // and similar runtime-feature casts can stay; mark them with a
    // `// eslint-disable-next-line @typescript-eslint/no-explicit-any`
    // and a one-line justification when intentional.
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/tests/**', 'src/**/*.test.ts', 'src/**/*.spec.ts'],
    plugins: {
      '@typescript-eslint': typescript,
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
  {
    ignores: ['dist/**', 'node_modules/**', 'coverage/**', '*.config.js', '*.config.ts'],
  },
];
