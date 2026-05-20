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
