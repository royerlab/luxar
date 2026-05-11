/**
 * Type declarations for Float16Array
 *
 * Float16Array is supported in modern browsers (2024+) but TypeScript
 * doesn't have built-in type definitions for it yet.
 *
 * Browser support as of 2024:
 * - Chrome 122 (Feb 2024)
 * - Edge 122 (Feb 2024)
 * - Firefox 127 (Jun 2024)
 * - Safari 17 (Sep 2023)
 * - Opera 108 (Mar 2024)
 */

interface Float16ArrayConstructor {
  readonly prototype: Float16Array;
  new (length: number): Float16Array;
  new (array: ArrayLike<number>): Float16Array;
  new (buffer: ArrayBuffer, byteOffset?: number, length?: number): Float16Array;
  readonly BYTES_PER_ELEMENT: number;
}

interface Float16Array extends ArrayBufferView {
  readonly buffer: ArrayBuffer;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly length: number;
  readonly BYTES_PER_ELEMENT: number;

  [index: number]: number;

  set(array: ArrayLike<number>, offset?: number): void;
  subarray(begin?: number, end?: number): Float16Array;
  copyWithin(target: number, start: number, end?: number): this;
  fill(value: number, start?: number, end?: number): this;
  forEach(
    callbackfn: (value: number, index: number, array: Float16Array) => void,
    thisArg?: unknown
  ): void;
  indexOf(searchElement: number, fromIndex?: number): number;
  lastIndexOf(searchElement: number, fromIndex?: number): number;
  slice(start?: number, end?: number): Float16Array;
}

declare global {
  var Float16Array: Float16ArrayConstructor;
}
