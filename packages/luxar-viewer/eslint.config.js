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
    },
  },
  {
    // Production code (excluding tests) — warn on `any` usage to nudge new
    // code toward typed alternatives without forcing a sweep of every legacy
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
