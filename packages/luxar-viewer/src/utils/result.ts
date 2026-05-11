/**
 * Result<T, E> — discriminated-union return type for fallible
 * operations.
 *
 * Many places in the data layer return `T | undefined` or `T | null`
 * to mean both "missing optional thing" and "real failure" — the
 * caller can't tell the cases apart, and the error context is lost.
 * `Result` makes the distinction explicit:
 *
 *   const r = await store.get(key);
 *   if (isOk(r)) use(r.value);
 *   else if (r.error === 'Missing') skipQuietly();
 *   else surfaceError(r.error);
 *
 * The error type defaults to `string` (suitable for tagged unions
 * like `'Missing' | 'Network' | 'Corrupt'`) but can be any type —
 * including a structured error object that carries a cause / stack.
 *
 * @module utils/result
 */

export type Ok<T> = { readonly ok: true; readonly value: T };
export type Err<E> = { readonly ok: false; readonly error: E };

/**
 * Discriminated union of success or failure. Narrow with
 * {@link isOk} / {@link isErr} or with the `ok` field directly.
 */
export type Result<T, E = string> = Ok<T> | Err<E>;

/** Construct a successful result. */
export function ok<T, E = string>(value: T): Result<T, E> {
  return { ok: true, value };
}

/** Construct a failed result. */
export function err<E, T = never>(error: E): Result<T, E> {
  return { ok: false, error };
}

/** Type guard for a successful result. */
export function isOk<T, E>(r: Result<T, E>): r is Ok<T> {
  return r.ok === true;
}

/** Type guard for a failed result. */
export function isErr<T, E>(r: Result<T, E>): r is Err<E> {
  return r.ok === false;
}

/**
 * Pattern-match a result to a value with separate handlers for the
 * two branches.
 *
 * ```ts
 * const display = match(result, {
 *   ok: (v) => `loaded ${v.length} bytes`,
 *   err: (e) => `failed: ${e}`,
 * });
 * ```
 */
export function match<T, E, R>(
  r: Result<T, E>,
  handlers: { ok: (value: T) => R; err: (error: E) => R }
): R {
  return r.ok ? handlers.ok(r.value) : handlers.err(r.error);
}

/**
 * Apply `f` to the success value (no-op on err). Mirrors
 * `Promise.then`'s shape but synchronous.
 */
export function mapOk<T, U, E>(r: Result<T, E>, f: (v: T) => U): Result<U, E> {
  return r.ok ? ok(f(r.value)) : r;
}

/**
 * Apply `f` to the error (no-op on ok). Useful for translating
 * lower-level errors into the caller's preferred type.
 */
export function mapErr<T, E, F>(r: Result<T, E>, f: (e: E) => F): Result<T, F> {
  return r.ok ? r : err(f(r.error));
}

/**
 * Extract the success value, throwing if the result is an error.
 * Use sparingly — prefer `match` or explicit branches; `unwrap` is
 * a code-smell signal that the caller hasn't decided how to handle
 * the error case.
 */
export function unwrap<T, E>(r: Result<T, E>): T {
  if (r.ok) return r.value;
  throw new Error(
    `unwrap on Err: ${typeof r.error === 'string' ? r.error : JSON.stringify(r.error)}`
  );
}

/**
 * Extract the success value, returning a fallback on error.
 *
 * ```ts
 * const config = unwrapOr(loadConfig(), DEFAULT_CONFIG);
 * ```
 */
export function unwrapOr<T, E>(r: Result<T, E>, fallback: T): T {
  return r.ok ? r.value : fallback;
}

/**
 * Run an async function and capture any thrown error in an `Err`.
 * Useful for converting throw-based APIs into Result-based ones at
 * a single call site.
 *
 * ```ts
 * const r = await tryAsync(() => zarrStore.get(key), (e) => `fetch failed: ${e}`);
 * ```
 */
export async function tryAsync<T, E = string>(
  fn: () => Promise<T>,
  mapError: (e: unknown) => E
): Promise<Result<T, E>> {
  try {
    return ok(await fn());
  } catch (e) {
    return err(mapError(e));
  }
}
