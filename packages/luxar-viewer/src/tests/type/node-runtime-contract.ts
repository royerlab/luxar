// Retarget this probe to the next Node-only API, or delete it, when .nvmrc advances; it complements scripts/check-node-types-version.mjs.
// @ts-expect-error Node 22 does not provide the Node 26-only `node:vfs` module.
import type { VirtualFileSystem } from 'node:vfs';

export type Node26OnlyVirtualFileSystem = VirtualFileSystem;
