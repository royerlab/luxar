/**
 * Scene-graph tree, LOD, and draw-order templates.
 */

import type {
  LODProgressState,
  NodeDensityState,
  NodeDrawOrder,
  SceneGraphNode,
  SceneGraphState,
} from '../../../types/data-monitor-types';
import { GEOMETRY_TYPES, LOADER_TYPES } from '../../../types/format-contract';
import { escapeHtml } from '../../../utils/escape-html';
import { MONITOR_ICONS } from './primitives';
import { formatNumber } from './format';

// Scene graph tree rendering

/**
 * Get icon for scene graph node type
 */
function getNodeTypeIcon(type: string): string {
  const icons: Record<string, string> = {
    scene: MONITOR_ICONS.nodeScene,
    group: MONITOR_ICONS.nodeGroup,
    points: MONITOR_ICONS.nodePoints,
    lines: MONITOR_ICONS.nodeLines,
    gsplats: MONITOR_ICONS.nodeGsplats,
    mesh: MONITOR_ICONS.nodeMesh,
  };
  return icons[type] || MONITOR_ICONS.dot;
}

/**
 * Get CSS class for node type color
 */
function getNodeTypeColorClass(type: string): string {
  const typeClasses: Record<string, string> = {
    scene: 'luxar-scene-graph__name--scene',
    group: 'luxar-scene-graph__name--group',
    points: 'luxar-scene-graph__name--points',
    lines: 'luxar-scene-graph__name--lines',
    gsplats: 'luxar-scene-graph__name--gsplats',
    mesh: 'luxar-scene-graph__name--mesh',
  };
  return typeClasses[type] || '';
}

/**
 * Pick the tree icon for a node. Specialized groups (`kind=lod` /
 * `kind=partition`) get their own glyph so they read distinctly from
 * plain containers; everything else falls back to its geometry type.
 */
function getSceneGraphIcon(node: SceneGraphNode): string {
  if (node.kind === 'lod') return MONITOR_ICONS.kindLod;
  if (node.kind === 'partition') return MONITOR_ICONS.kindPartition;
  return getNodeTypeIcon(node.type);
}

/**
 * Render the kind badge (`K LODs` / `N parts`) for a specialized group.
 * Mirrors the Layers-panel `--kind` badge so the two panels read
 * consistently. Returns `''` for non-specialized nodes.
 */
function renderKindBadge(node: SceneGraphNode): string {
  if (node.kind === 'lod' && (node.lodGroupChildCount ?? 0) > 0) {
    const title = `Substitutive LOD group · ${node.lodGroupChildCount} levels (one rendered at a time)`;
    return `<span class="luxar-scene-graph__badge luxar-scene-graph__badge--kind" title="${escapeHtml(title)}">${node.lodGroupChildCount} LODs</span>`;
  }
  if (node.kind === 'partition' && (node.partCount ?? 0) > 0) {
    const title = `Partition group · ${node.partCount} disjoint BSP parts — all parts are rendered; the GPU frustum-culls each part at draw time`;
    return `<span class="luxar-scene-graph__badge luxar-scene-graph__badge--kind" title="${escapeHtml(title)}">${node.partCount} parts</span>`;
  }
  return '';
}

/**
 * Render the live LOD-progress chip for a node, driven by the
 * {@link LODProgressState} snapshot:
 *   - substitutive (`kind=lod`): "L{active+1}/{count}" active-level chip.
 *   - additive (`additiveSublods`): "LOD {loaded}/{total}" with a ⏳ while
 *     refinement is in progress and a residency dot (● cached / ◌ streaming).
 * Renders a structural slot even before the first provider poll so the
 * incremental patcher (`updateSceneGraphBadges`) can fill it in-place.
 * Returns `''` for nodes with no LOD dimension.
 */
export function lodChipContent(
  node: SceneGraphNode,
  state: LODProgressState | undefined
): { text: string; title: string } | null {
  if (node.kind === 'lod') {
    const count = state?.levelCount ?? node.lodGroupChildCount ?? 0;
    if (count <= 0) return null;
    // No live registry state yet: show "–" instead of guessing level 1.
    if (!state || state.activeLevel === undefined) {
      return {
        text: `L–/${count}`,
        title: `Substitutive LOD group with ${count} levels — active level not yet reported`,
      };
    }
    const active = state.activeLevel + 1;
    const sel = state.selector && state.selector !== 'auto' ? ` (${state.selector})` : '';
    return {
      text: `L${active}/${count}`,
      title: `Active substitutive level ${active} of ${count}${sel} — only this level is rendered`,
    };
  }

  if (node.additiveSublods && node.additiveSublods > 1) {
    const total = state?.total ?? node.additiveSublods;
    // No live loader state: the node's progressive loader isn't streaming
    // (typically an inactive substitutive level). Show "–" rather than a
    // fabricated 0 so "not active" doesn't read as "stalled at zero".
    if (!state) {
      return {
        text: `LOD –/${total}`,
        title: `Additive LOD — ${total} detail levels available; not streaming (level not active)`,
      };
    }
    const loaded = state.loaded ?? 0;
    const refining = state.refining === true;
    const residency =
      state.lastAllResident === false ? ' ◌' : state.lastAllResident === true ? ' ●' : '';
    const spinner = refining ? ' ⏳' : '';
    // Always spell out what the residency dot means — the ● typically
    // appears exactly when refinement has finished, so the explanation
    // must not be gated on `refining`.
    const residencyNote =
      state.lastAllResident === true
        ? ' · ● = fully cache-resident (no network needed)'
        : state.lastAllResident === false
          ? ' · ◌ = streaming from network'
          : '';
    // Committed energy fraction e(k) — how much of the ladder's total
    // self-energy is already on screen (quality stamps; absent on legacy
    // unstamped datasets). Far more informative than the raw level count:
    // energy-ordered streaming front-loads the visually important elements,
    // so e.g. 2/6 levels can already carry ~70% of the energy.
    const energyStr =
      typeof state.energy === 'number' ? ` ~${Math.round(state.energy * 100)}%` : '';
    const energyNote =
      typeof state.energy === 'number'
        ? ` · ~${Math.round(state.energy * 100)}% of the level's total energy already on screen (energy-ordered streaming loads the visually important elements first)`
        : '';
    const base = refining
      ? `Additive LOD refining — ${loaded}/${total} levels loaded`
      : `Additive LOD — ${loaded}/${total} levels loaded`;
    return {
      text: `LOD ${loaded}/${total}${energyStr}${residency}${spinner}`,
      title: `${base}${energyNote}${residencyNote}`,
    };
  }

  return null;
}

function renderLodChip(node: SceneGraphNode, state: LODProgressState | undefined): string {
  const content = lodChipContent(node, state);
  if (!content) return '';
  return `<span class="luxar-scene-graph__lod" data-lod-path="${escapeHtml(node.path)}" title="${escapeHtml(content.title)}">${escapeHtml(content.text)}</span>`;
}

/**
 * A live chip's content: an optional inline SVG glyph (trusted markup from
 * `MONITOR_ICONS`), the short text beside it, and the full explanation as a
 * tooltip. The glyph carries what used to be a word (`transparent`, `drawn`)
 * so a row of chips stays short; the tooltip is where the meaning lives.
 */
export interface ChipContent {
  icon?: string;
  text: string;
  title: string;
}

/** Inner HTML for a chip: glyph markup + escaped text; `''` for no content. */
export function chipInnerHtml(content: ChipContent | null): string {
  if (!content) return '';
  return `${content.icon ?? ''}${escapeHtml(content.text)}`;
}

/**
 * Chip content for a node's live draw-order state: a bucket glyph (two
 * overlapping outlines = `transparent`, a filled square = `opaque`), the
 * resolved `renderOrder` (drawn ascending) with any authored layer order in
 * front, and the full explanation — bucket, depthWrite, order — as the
 * tooltip. Returns `null` when no draw-order state is known for the node
 * (no live mesh — a group, or before the first provider poll). Shared by the
 * initial render and the monitor's incremental patcher so both agree.
 */
export function drawOrderChipContent(state: NodeDrawOrder | undefined): ChipContent | null {
  if (!state) return null;
  const dw = state.depthWrite ? 'depthWrite on' : 'depthWrite off';
  // An authored band is prefixed (`O2 #2 transparent`) because it EXPLAINS the
  // resolved integer beside it: with a level, the layer's position is stated and
  // camera-independent; without one it is inferred from the geometry each frame.
  // `O`, not `L`: the LOD chip immediately to the left of this one already
  // renders `L{active}/{count}`, so an `L3` here reads as an LOD level rather
  // than a layer order — the exact misreading this prefix has to avoid.
  const order = state.layerOrder === undefined ? '' : `O${state.layerOrder} `;
  // `transparent` / `opaque` names a RENDER BUCKET, not an opacity. Spelling
  // that out matters because the word invites the wrong reading: a fully opaque
  // additive layer still sits in the `transparent` bucket, and a layer at 5%
  // opacity in `opaque` mode still sits in the `opaque` one.
  const bucket =
    state.bucket === 'opaque'
      ? "'opaque' render bucket — NOT an opacity: it means this mesh is drawn in " +
        "THREE's depth-first pass, before every transparent mesh in the scene " +
        'regardless of renderOrder. `opaque` is the only Luxar blending mode that ' +
        'lands here, and a backdrop must use it to composite under the content in ' +
        'front of it.'
      : "'transparent' render bucket — NOT an opacity: it means this mesh is drawn " +
        "in THREE's blended pass, after every opaque mesh, and its order there is " +
        'what renderOrder decides. Every blending mode except `opaque` lands here ' +
        '(additive, volumetric, normal, max, luminous), however opaque the pixels ' +
        'themselves look.';

  return {
    icon: state.bucket === 'opaque' ? MONITOR_ICONS.bucketOpaque : MONITOR_ICONS.bucketTransparent,
    text: `${order}#${state.renderOrder}`,
    title:
      // Bucket first: it is what the glyph stands for.
      `${bucket} ${dw}. ` +
      (state.layerOrder === undefined
        ? 'Layer order: none authored, so the cross-layer order is INFERRED from the ' +
          'geometry each frame (mean view depth, then bounding-sphere containment). '
        : `Layer order ${state.layerOrder}: the cross-layer order is STATED by the author ` +
          'and does not change with the camera. Higher draws nearer the viewer, like a ' +
          'CSS z-index. ') +
      `renderOrder ${state.renderOrder} — compared ascending, so lower is drawn first, ` +
      'and only ever compared against meshes in the SAME bucket.',
  };
}

/**
 * The viewer-drawable node types (the loader set) that get a draw-order chip
 * slot. Matches the provider's classification in
 * `data/scene-loader/monitor/draw-order-provider.ts`.
 */
const DRAWABLE_NODE_TYPES: ReadonlySet<string> = new Set<string>(LOADER_TYPES);

function renderDrawOrderChip(node: SceneGraphNode, state: NodeDrawOrder | undefined): string {
  const content = drawOrderChipContent(state);
  // A drawable node gets a (possibly empty) chip slot even without live
  // state: the per-tick updater only patches EXISTING elements, and a node
  // hidden at structural-render time (toggled-off layer, inactive
  // substitutive-LOD level) has no provider state yet — without the empty
  // slot its chip could never appear once the node becomes visible.
  if (!content && !DRAWABLE_NODE_TYPES.has(node.type)) return '';
  const title = content ? escapeHtml(content.title) : '';
  return `<span class="luxar-scene-graph__draworder" data-draworder-path="${escapeHtml(node.path)}" title="${title}">${chipInnerHtml(content)}</span>`;
}

/**
 * Chip content for a node's live density-guard state: a thinned-lattice glyph
 * plus `1/K` while the guard thins the node, `null` otherwise (unthinned,
 * off-screen, no record, guard off). Deliberately silent at keep 1 so the row
 * only gains a chip when something is actually being left out of the draw.
 * The LOD chip to its left is unaffected by thinning — every resident element
 * stays resident — which is exactly the misreading this chip is here to
 * prevent, and the tooltip says so. Shared by the initial render and the
 * incremental patcher.
 */
export function densityChipContent(state: NodeDensityState | undefined): ChipContent | null {
  if (!state || !(state.keep < 1) || !state.onScreen) return null;
  const k = Math.round(1 / state.keep);
  const epp = state.elementsPerPixel;
  const eppStr = epp >= 10 ? Math.round(epp).toLocaleString() : epp.toFixed(1);
  return {
    icon: MONITOR_ICONS.densityThinned,
    text: `1/${k}`,
    title:
      `Drawn 1/${k} of the resident elements. Density guard: this node projects ` +
      `${eppStr} resident elements per pixel, so the shader draws a hashed 1/${k} of ` +
      `them and brightens each ×${k} — the composited brightness is unchanged and ` +
      'nothing is unloaded (the LOD chip still counts every resident level). ' +
      'Zooming in restores the full draw step by step; the Density Guard toggle ' +
      'in the Performance popover turns thinning off.',
  };
}

function renderDensityChip(node: SceneGraphNode, state: NodeDensityState | undefined): string {
  const content = densityChipContent(state);
  // Persistent (possibly empty) slot for every drawable node, for the same
  // reason as the draw-order chip: the per-tick patcher only fills existing
  // elements, and thinning comes and goes with the camera.
  if (!content && !DRAWABLE_NODE_TYPES.has(node.type)) return '';
  const title = content ? escapeHtml(content.title) : '';
  return `<span class="luxar-scene-graph__density" data-density-path="${escapeHtml(node.path)}" title="${title}">${chipInnerHtml(content)}</span>`;
}

/**
 * Role of a node that is a direct child of a substitutive `kind=lod`
 * group: `active` = the level currently rendered, `inactive` = a level
 * present in the file but not rendered right now. `undefined` when the
 * node is not a substitutive level (or the active level is unknown).
 */
interface LevelContext {
  parentPath: string;
  index: number;
  role: 'active' | 'inactive' | undefined;
}

/**
 * Stats-badge content (element / child count + tooltip) for a tree node.
 * Shared by the initial render and the incremental badge patcher so text
 * and tooltip always agree. Per-type visible counts (after nD slicing)
 * are appended symmetrically for all four geometry types when known and
 * different from the dataset count. Returns `null` for nodes with no
 * stats badge (leaves without counts; specialized groups, whose kind
 * badge already carries the child count).
 */
export function nodeStatsContent(node: SceneGraphNode): { text: string; title: string } | null {
  const visibleSuffix = (visible: number | undefined, total: number): string =>
    visible !== undefined && visible !== total
      ? ` (${visible.toLocaleString()} visible after slicing)`
      : '';

  if (node.type === 'points' && node.pointCount !== undefined) {
    return {
      text: formatNumber(node.pointCount),
      title: `${node.pointCount.toLocaleString()} points in this layer${visibleSuffix(node.visiblePointCount, node.pointCount)}`,
    };
  }
  if (node.type === 'lines' && node.segmentCount !== undefined) {
    let title = `${node.segmentCount.toLocaleString()} line segments`;
    if (node.vertexCount !== undefined) {
      title += `, ${node.vertexCount.toLocaleString()} vertices`;
    }
    title += visibleSuffix(node.visibleSegmentCount, node.segmentCount);
    return { text: formatNumber(node.segmentCount), title };
  }
  if (node.type === 'gsplats' && node.splatCount !== undefined) {
    return {
      text: formatNumber(node.splatCount),
      title: `${node.splatCount.toLocaleString()} Gaussian splats${visibleSuffix(node.visibleSplatCount, node.splatCount)}`,
    };
  }
  if (node.type === 'mesh' && node.faceCount !== undefined) {
    // Triangles, matching the drawn-primitive convention the arms above use — `lines`
    // reports segments rather than vertices for the same reason. Vertices ride along in
    // the tooltip exactly as they do for lines, since for a mesh both numbers are
    // interesting (the vertex:face ratio is what tells a welded surface from a soup).
    //
    // `visibleSuffix` applies here as it does for the other three, from the count the
    // visible-counts walk already measured per path (`monitor/visible-counts.ts` →
    // `SceneGraphModel.syncVisibleCountsIntoTree`). For a mesh it reads as "how much of
    // this surface the current nD slab indexes" rather than a streaming residency: the
    // node is resident in full either way.
    let title = `${node.faceCount.toLocaleString()} triangles`;
    if (node.vertexCount !== undefined) {
      title += `, ${node.vertexCount.toLocaleString()} vertices`;
    }
    title += visibleSuffix(node.visibleFaceCount, node.faceCount);
    return { text: formatNumber(node.faceCount), title };
  }
  if (node.type === 'group' && node.children.length > 0 && !node.kind) {
    // Plain groups show child count. Specialized groups (kind=lod /
    // kind=partition) skip it — their kind badge ("K LODs" / "N parts")
    // already carries the same number.
    return {
      text: `${node.children.length}`,
      title: `${node.children.length} child node${node.children.length !== 1 ? 's' : ''}`,
    };
  }
  return null;
}

/** Human-readable suffix for a substitutive level's row tooltip. */
export function levelRoleTitleSuffix(role: 'active' | 'inactive' | undefined): string {
  if (role === 'active') return ' — active substitutive level (currently rendered)';
  if (role === 'inactive') return ' — inactive substitutive level (not rendered)';
  return '';
}

/**
 * Render a single scene graph tree node
 */
function renderSceneGraphNode(
  node: SceneGraphNode,
  expandedNodes: ReadonlySet<string>,
  depth: number = 0,
  lodStates?: ReadonlyMap<string, LODProgressState>,
  drawOrderStates?: ReadonlyMap<string, NodeDrawOrder>,
  levelCtx?: LevelContext,
  densityStates?: ReadonlyMap<string, NodeDensityState>
): string {
  const hasChildren = node.children.length > 0;
  const isExpanded = expandedNodes.has(node.path);

  // Node stats and tooltips (shared with the incremental badge patcher).
  const stats = nodeStatsContent(node);
  const statsText = stats?.text ?? '';
  const statsTooltip = stats?.title ?? '';

  // Build tooltip for the whole node
  const nodeTypeDescriptions: Record<string, string> = {
    scene: 'Root scene node',
    group: 'Container for organizing nodes',
    points: 'Point cloud layer',
    lines: 'Line segments layer',
    gsplats: 'Gaussian splats layer',
    mesh: 'Mesh geometry',
  };
  const kindDescriptions: Partial<Record<string, string>> = {
    lod: 'Substitutive LOD group (one level rendered at a time)',
    partition: 'Partition group (disjoint BSP parts)',
  };
  const baseDesc = node.kind
    ? kindDescriptions[node.kind] || node.kind
    : nodeTypeDescriptions[node.type] || node.type;
  const baseTooltip = `${baseDesc}${node.hasSpatialIndex ? ' (indexed)' : ''}`;
  const nodeTooltip = `${baseTooltip}${levelRoleTitleSuffix(levelCtx?.role)}`;

  const kindBadge = renderKindBadge(node);
  const lodChip = renderLodChip(node, lodStates?.get(node.path));
  const drawOrderChip = renderDrawOrderChip(node, drawOrderStates?.get(node.path));
  const densityChip = renderDensityChip(node, densityStates?.get(node.path));

  // Expand/collapse toggle
  const toggleIcon = hasChildren ? (isExpanded ? '▼' : '▶') : '•';
  const toggleClass = hasChildren
    ? 'luxar-scene-graph__toggle--clickable'
    : 'luxar-scene-graph__toggle--disabled';

  // Indent using CSS custom property for dynamic depth
  const indentStyle = `style="--node-depth: ${depth}; padding-left: calc(var(--node-depth) * 16px);"`;

  // Substitutive-level rows carry data attributes so the incremental
  // patcher can re-mark active/inactive when the LOD selector switches
  // levels between structural rebuilds (`data-base-title` lets it
  // re-derive the tooltip without re-rendering).
  const levelClass =
    levelCtx?.role === 'active'
      ? ' luxar-scene-graph__node-row--active-level'
      : levelCtx?.role === 'inactive'
        ? ' luxar-scene-graph__node-row--inactive-level'
        : '';
  const levelAttrs = levelCtx
    ? ` data-level-of="${escapeHtml(levelCtx.parentPath)}" data-level-index="${levelCtx.index}" data-base-title="${escapeHtml(baseTooltip)}"`
    : '';

  return `
    <div class="luxar-scene-graph__node">
      <div class="luxar-scene-graph__node-row${levelClass}" ${indentStyle} title="${escapeHtml(nodeTooltip)}"${levelAttrs}>
        <!-- Toggle -->
        <span
          class="luxar-scene-graph__toggle ${toggleClass}"
          ${hasChildren ? `data-action="toggleNode" data-node-path="${escapeHtml(node.path)}"` : ''}
          ${hasChildren ? `title="${isExpanded ? 'Collapse' : 'Expand'} ${escapeHtml(node.name)}"` : ''}
        >${toggleIcon}</span>

        <!-- Icon & Name -->
        <span class="luxar-scene-graph__icon">${getSceneGraphIcon(node)}</span>
        <span class="luxar-scene-graph__name ${getNodeTypeColorClass(node.displayType ?? node.type)}">
          ${escapeHtml(node.name)}
        </span>

        <!-- Stats badge -->
        ${
          statsText
            ? `<span class="luxar-scene-graph__badge" data-node-path="${escapeHtml(node.path)}" title="${escapeHtml(statsTooltip)}">${statsText}</span>`
            : ''
        }

        <!-- Kind badge (K LODs / N parts) -->
        ${kindBadge}

        <!-- Live LOD-progress chip (also carries the ⏳ refining marker) -->
        ${lodChip}

        <!-- Live draw-order chip (blending bucket / depthWrite / renderOrder) -->
        ${drawOrderChip}

        <!-- Live density-guard chip (keep fraction while thinned) -->
        ${densityChip}
      </div>

      <!-- Children (if expanded) -->
      ${
        isExpanded && hasChildren
          ? node.children
              .map((child, i) =>
                renderSceneGraphNode(
                  child,
                  expandedNodes,
                  depth + 1,
                  lodStates,
                  drawOrderStates,
                  node.kind === 'lod'
                    ? {
                        parentPath: node.path,
                        index: i,
                        role: activeLevelRole(lodStates?.get(node.path), i),
                      }
                    : undefined,
                  densityStates
                )
              )
              .join('')
          : ''
      }
    </div>
  `;
}

/**
 * Resolve a substitutive level's role from its parent group's live state.
 * Unknown (no state yet) → `undefined`, so rows aren't mis-marked before
 * the first provider poll. Shared by the initial render and the monitor's
 * incremental level-row patcher so both derive the role identically.
 */
export function activeLevelRole(
  state: LODProgressState | undefined,
  index: number
): 'active' | 'inactive' | undefined {
  const active = state?.kind === 'lod' ? state.activeLevel : undefined;
  if (active === undefined) return undefined;
  return index === active ? 'active' : 'inactive';
}

/**
 * Render scene graph tree component
 */
export function renderSceneGraphTree(
  state: SceneGraphState,
  expandedNodes: ReadonlySet<string>,
  lodStates?: ReadonlyMap<string, LODProgressState>,
  drawOrderStates?: ReadonlyMap<string, NodeDrawOrder>,
  densityStates?: ReadonlyMap<string, NodeDensityState>
): string {
  if (!state.root) {
    return `
      <div class="luxar-scene-graph__empty">
        No scene loaded
      </div>
    `;
  }

  // Header with stats. "layers" (node counts), not element counts — the
  // tooltip disambiguates, since "5 gsplats" otherwise reads as 5 splats.
  const headerStats = GEOMETRY_TYPES.map((t) =>
    state.nodesByType[t] > 0 ? `${state.nodesByType[t]} ${t}` : null
  )
    .filter(Boolean)
    .join(', ');
  const headerStatsTooltip =
    'Layer (node) counts per geometry type — not element counts. ' +
    'Each level of a substitutive LOD group counts as its own layer, ' +
    'even though only one level renders at a time.';

  // Summarise LOD/partition activity across the scene so the user sees it
  // without expanding the tree: how many substitutive-LOD groups, how many
  // additive nodes streaming/refining, out of how many additive nodes total.
  const lodSummary = summariseLodStates(lodStates, countAdditiveNodes(state.root));
  const lodSummaryTooltip =
    'LOD activity: substitutive = groups that swap between K resolutions of the same data ' +
    '(one rendered at a time) · additive = layers refined by streaming extra detail levels ' +
    'on top of a base ("x/y active" = levels with a live streaming loader out of all additive layers) · ' +
    'partition = groups of disjoint spatial parts (all rendered, frustum-culled per part) · ' +
    'refining = additive layers still loading detail.';

  return `
    <div class="luxar-scene-graph">
      <div class="luxar-scene-graph__header">
        <h4 class="luxar-scene-graph__title" title="Hierarchy of scene nodes (groups, points, lines, gsplats, mesh) with per-node element counts">SCENE GRAPH</h4>
        ${headerStats ? `<span class="luxar-scene-graph__stats" title="${escapeHtml(headerStatsTooltip)}">${headerStats}</span>` : ''}
      </div>
      ${lodSummary ? `<div class="luxar-scene-graph__lod-summary" data-field="lod-summary" title="${escapeHtml(lodSummaryTooltip)}">${lodSummary}</div>` : ''}
      <div class="luxar-scene-graph__container">
        ${renderSceneGraphNode(state.root, expandedNodes, 0, lodStates, drawOrderStates, undefined, densityStates)}
      </div>
    </div>
  `;
}

/**
 * Count additive-LOD layers (nodes with `additiveSublods > 1`) in the tree.
 * Used to reconcile the header summary (which counts *live* streaming
 * loaders) with the tree (which renders a chip slot for every additive
 * layer): "1/5 additive active" instead of a bare "1 additive" that
 * contradicts five visible chips.
 */
export function countAdditiveNodes(root: SceneGraphNode | null | undefined): number {
  if (!root) return 0;
  let count = (root.additiveSublods ?? 0) > 1 ? 1 : 0;
  for (const child of root.children) count += countAdditiveNodes(child);
  return count;
}

/**
 * One-line summary of LOD/partition activity for the scene-graph header,
 * e.g. "2 substitutive · 1/5 additive active · refining 1". Returns `''`
 * when no LOD/partition state is present.
 *
 * @param additiveTotal - total additive layers in the *tree* (see
 *   {@link countAdditiveNodes}). The live count only covers layers with a
 *   streaming loader attached (typically just the active substitutive
 *   level); showing "live/total" keeps the summary consistent with the
 *   number of additive chips visible in the tree.
 */
export function summariseLodStates(
  lodStates?: ReadonlyMap<string, LODProgressState>,
  additiveTotal?: number
): string {
  if (!lodStates || lodStates.size === 0) return '';
  let lod = 0;
  let additive = 0;
  let partition = 0;
  let refining = 0;
  for (const s of lodStates.values()) {
    if (s.kind === 'lod') lod++;
    else if (s.kind === 'additive') {
      additive++;
      if (s.refining) refining++;
    } else if (s.kind === 'partition') partition++;
  }
  const parts: string[] = [];
  // "substitutive" (not the generic "LOD group") so it reads in parallel
  // with "additive" — both are LOD kinds; naming only one "LOD" was the
  // ambiguous wording.
  if (lod > 0) parts.push(`${lod} substitutive`);
  if (additiveTotal !== undefined && additiveTotal > additive) {
    parts.push(`${additive}/${additiveTotal} additive active`);
  } else if (additive > 0) {
    parts.push(`${additive} additive`);
  }
  if (partition > 0) parts.push(`${partition} partition${partition !== 1 ? 's' : ''}`);
  if (refining > 0) parts.push(`refining ${refining}`);
  return parts.join(' · ');
}
