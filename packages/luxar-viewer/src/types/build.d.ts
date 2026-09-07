/**
 * The compile-time build stamp injected by Vite's `define:`.
 *
 * Declared as a global rather than imported because that is what `define`
 * produces: a bare identifier substituted at build time. It is ABSENT in
 * vitest, in ts-node tooling, and in any consumer that bundles `src/` with its
 * own config — so the only legal way to touch it is behind a `typeof` guard.
 * `config/build-info.ts` is that guard; nothing else should name this symbol.
 *
 * The value is a JSON *string* (see `buildDefine` in `tools/build-identity.ts`
 * for why it is double-encoded), not an object.
 */
declare const __LUXAR_BUILD__: string;
