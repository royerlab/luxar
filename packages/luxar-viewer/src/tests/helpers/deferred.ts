/**
 * A promise plus the handles that settle it, for tests that order async
 * steps explicitly (`Promise.withResolvers` is outside the ES2022 lib this
 * package compiles against).
 */
export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

/** A fresh {@link Deferred}; `deferred()` settles a `Promise<void>`. */
export function deferred<T = void>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
