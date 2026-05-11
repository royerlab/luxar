/**
 * SceneGraphBuilder - Builds the scene graph structure from Zarr metadata.
 *
 * This module is responsible for:
 * - Enumerating zarr store contents
 * - Building the hierarchical SceneNode structure
 * - Parsing node attributes and types
 * - URL normalization for zarr access
 *
 * The actual THREE.js scene construction remains in SceneLoader as it requires
 * complex loader integration and view state management.
 *
 * Extracted from SceneLoader to reduce its complexity.
 *
 * @module data/scene-graph-builder
 */

import * as zarr from '../zarr';
import { SceneNode } from '../data-loader-types';
import { ZarrNodeAttrs, hasContentsMethod } from '../../types/zarr';
import { log, Modules } from '../../utils/log';

/**
 * Entry in the store listing.
 */
export interface StoreEntry {
  /** Path in the store */
  path: string;
  /** Kind of entry (group, array) */
  kind: string;
}

/**
 * Builds scene graph structures from Zarr store metadata.
 *
 * This class handles the pure graph-building logic without the
 * THREE.js scene construction, which remains in SceneLoader.
 *
 * @example
 * ```typescript
 * const builder = new SceneGraphBuilder(store);
 *
 * // Enumerate store contents
 * const entries = await builder.enumerateStore();
 *
 * // Build scene graph from root location
 * const sceneGraph = await builder.buildSceneGraph(rootLoc, rootAttrs);
 * ```
 */
export class SceneGraphBuilder {
  private store: zarr.Readable;

  constructor(store: zarr.Readable) {
    this.store = store;
  }

  /**
   * Enumerate all entries in the zarr store.
   *
   * Uses consolidated metadata if available, otherwise falls back
   * to a minimal enumeration.
   *
   * @returns Array of store entries with path and kind
   */
  async enumerateStore(): Promise<StoreEntry[]> {
    // Try to use consolidated metadata
    if (hasContentsMethod(this.store)) {
      const contents = await this.store.contents();
      log.custom('📋', Modules.SCENE_LOADER, `Found ${contents.length} items in store`);
      return contents;
    }

    // Fallback enumeration
    log.warning(Modules.SCENE_LOADER, 'Store does not support contents(), using fallback');
    return [{ path: '/', kind: 'group' }];
  }

  /**
   * Build the scene graph structure from zarr metadata.
   *
   * Creates a hierarchical SceneNode structure by:
   * 1. Enumerating all groups in the store
   * 2. Sorting by path depth (parents before children)
   * 3. Opening each group to read its attributes
   * 4. Building parent-child relationships
   *
   * @param rootLoc - Root zarr location
   * @param rootAttrs - Root node attributes
   * @returns Root SceneNode with nested children
   */
  async buildSceneGraph(
    rootLoc: zarr.Location<zarr.Readable>,
    rootAttrs: Record<string, unknown>
  ): Promise<SceneNode> {
    // Enumerate all groups in the store
    const listing = await this.enumerateStore();

    // Build hierarchical structure
    const root: SceneNode = {
      path: '/',
      type: 'scene',
      attrs: rootAttrs,
      hasSpatialIndex: false,
      children: [],
    };

    // Build node map
    const nodeMap = new Map<string, SceneNode>();
    nodeMap.set('/', root);

    // Sort by path depth to ensure parents are created before children
    const sortedPaths = listing
      .filter((e) => e.kind === 'group' && e.path !== '/')
      .sort((a, b) => a.path.split('/').length - b.path.split('/').length);

    for (const entry of sortedPaths) {
      const loc = rootLoc.resolve(entry.path.slice(1)); // Remove leading /
      const group = await zarr.open(loc, { kind: 'group' });
      const attrs = group.attrs as ZarrNodeAttrs;

      // We no longer check for spatial index here - PointsSpatialIndexLoader handles it
      const node: SceneNode = {
        path: entry.path,
        type: attrs?.type || 'group',
        attrs: attrs || {},
        hasSpatialIndex: false, // Will be determined by the loader
        children: [],
      };

      // Log if extend_to_all is present
      if (attrs?.extend_to_all) {
        log.data(
          Modules.SCENE_LOADER,
          `Node ${entry.path} has extend_to_all: ${attrs.extend_to_all.join(', ')}`
        );
      }

      // Find parent and add as child
      const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/')) || '/';
      const parent = nodeMap.get(parentPath);
      if (parent) {
        parent.children = parent.children || [];
        parent.children.push(node);
      }

      nodeMap.set(entry.path, node);
    }

    return root;
  }

  /**
   * Normalize URL for zarr store access.
   *
   * Handles both absolute HTTP URLs and relative paths,
   * ensuring trailing slashes for proper zarr access.
   *
   * @param url - URL or path to normalize
   * @returns Normalized URL with trailing slash
   */
  static normalizeURL(url: string): string {
    // Case-insensitive scheme match. Without it, mixed-case URLs like
    // `HTTPS://Example.com/data.zarr` get treated as relative paths and
    // window.location.origin is prepended. Mirrors
    // `data/scene-loader/url-normalization.ts`.
    if (/^https?:\/\//i.test(url)) {
      return url.endsWith('/') ? url : url + '/';
    }

    // For relative paths, we need window context
    if (typeof window === 'undefined') {
      // Node.js environment - just add trailing slash
      return url.endsWith('/') ? url : url + '/';
    }

    const baseUrl = window.location.origin;
    const cleanPath = url.startsWith('/') ? url : '/' + url;
    return baseUrl + cleanPath + (cleanPath.endsWith('/') ? '' : '/');
  }

  /**
   * Get the total count of nodes by type in a scene graph.
   *
   * Useful for logging and statistics.
   *
   * @param root - Root scene node
   * @returns Object with counts per node type
   */
  static countNodeTypes(root: SceneNode): Record<string, number> {
    const counts: Record<string, number> = {};

    function traverse(node: SceneNode): void {
      counts[node.type] = (counts[node.type] || 0) + 1;
      if (node.children) {
        for (const child of node.children) {
          traverse(child);
        }
      }
    }

    traverse(root);
    return counts;
  }

  /**
   * Find all nodes of a specific type in the scene graph.
   *
   * @param root - Root scene node
   * @param type - Node type to find
   * @returns Array of matching nodes
   */
  static findNodesByType(root: SceneNode, type: string): SceneNode[] {
    const results: SceneNode[] = [];

    function traverse(node: SceneNode): void {
      if (node.type === type) {
        results.push(node);
      }
      if (node.children) {
        for (const child of node.children) {
          traverse(child);
        }
      }
    }

    traverse(root);
    return results;
  }

  /**
   * Find a node by path in the scene graph.
   *
   * @param root - Root scene node
   * @param path - Path to find
   * @returns Matching node or undefined
   */
  static findNodeByPath(root: SceneNode, path: string): SceneNode | undefined {
    function traverse(node: SceneNode): SceneNode | undefined {
      if (node.path === path) {
        return node;
      }
      if (node.children) {
        for (const child of node.children) {
          const found = traverse(child);
          if (found) return found;
        }
      }
      return undefined;
    }

    return traverse(root);
  }
}
