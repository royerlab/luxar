// @ts-expect-error Node 22 does not provide the Node 26-only `node:vfs` module.
import type { VirtualFileSystem } from 'node:vfs';

export type Node26OnlyVirtualFileSystem = VirtualFileSystem;
