/**
 * Race a promise against a timeout. Throws `Error('OPFS timeout: <label> exceeded <ms>ms')`
 * if the timeout fires first. Used to bound individual OPFS I/O calls
 * so a hung browser handle cannot stall the cache indefinitely.
 */
export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  label: string
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(new Error(`OPFS timeout: ${label} exceeded ${timeoutMs}ms`)),
          timeoutMs
        );
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
