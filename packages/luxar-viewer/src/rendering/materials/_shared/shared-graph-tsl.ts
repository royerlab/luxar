/**
 * One TSL node graph per material CONFIGURATION, shared by every material
 * of that configuration.
 *
 * three's WebGPU renderer caches a node build (`NodeManager.getForRender` →
 * `NodeBuilder.build`, 11–28 ms for a gsplat graph) under a key that hashes
 * every node of the material's graph by its `id`. A wrapper that builds its
 * own graph over its own `uniform()` / `texture()` leaves therefore never
 * shares a build, so N materials of one configuration cost N builds — 92 of
 * them in a 6 s playback pass of a 50-timepoint partition (#2956's gating
 * activates each timepoint's parts on first visit). WebGL caches programs by
 * source and never paid this.
 *
 * Here the graph is built ONCE per configuration key, over FORWARDING
 * leaves: each forwarding leaf is a fresh `uniform()` / `texture()` of the
 * same type whose `updateBefore` (object granularity — run for every draw,
 * before the draw's bindings upload) copies the value of the same-named leaf
 * of the material being drawn. The wrapper keeps its own leaves as plain
 * value holders, so its `uniforms` proxies, clone and setter paths are
 * unchanged; the cached `NodeBuilderState` (whose bindings point at the
 * shared forwarding leaves) is then valid for every material of the key,
 * and each draw still uploads its own material's values into its own
 * per-object binding clone. This is the pattern three's built-in node
 * materials use for `map` / `color` (`materialReference`), done generically
 * over a whole leaf set so it also covers `.load()` texture fetches, which a
 * `ReferenceNode` cannot express.
 *
 * The graph slots shared are `vertexNode`, `colorNode` and `depthNode` (the
 * last written only by the pick factories).
 *
 * Contract for a caller:
 *
 *   - Everything the factory reads at graph-construction OR build time that
 *     selects code must be in `key` (flags, the baked element-texture width,
 *     …). Leaf presence and each leaf's type (texture type/format) are added
 *     automatically.
 *   - The factory must read leaf VALUES only through the graph (never bake a
 *     `.value` into code) except through `key`.
 *   - Material state the factory tail writes (blending, `toneMapped`) is
 *     per-material and must be re-applied by the caller — the factory runs
 *     against a scratch material, once per key.
 *   - Two texture leaves of one material must never hold the same texture.
 *     three merges texture nodes of one texture into one binding at build
 *     time, and a build three repeats for another render context sees the
 *     values the forwarding leaves last held: the two leaves would then share
 *     one binding for every material of the key. No wrapper does this today
 *     (each has at most an element/base-colour texture and a colormap).
 *
 * Kept out of `tsl-helpers.ts` (it needs `three/webgpu`; see
 * `live-texture-tsl.ts` for the import-cone reason).
 *
 * @module rendering/materials/_shared/shared-graph-tsl
 */

import * as THREE from 'three';
import { texture, uniform } from 'three/tsl';
import { NodeMaterial, NodeUpdateType } from 'three/webgpu';
import { SOFT_DISPOSE_FLAG } from '../../material-manager/soft-dispose-flag';
import type { TSLNode } from './tsl-helpers';

/** A material's leaf set: name → TSL leaf (`uniform()` or `texture()`), or absent. */
export type TSLLeafSet = object;

/** The graph slots a factory writes, shared across one configuration. */
export interface SharedTSLGraph {
  readonly vertexNode: TSLNode;
  readonly colorNode: TSLNode;
  /**
   * The fragment depth output, for graphs that write one (the pick factories'
   * brightness-as-depth / real-depth convention); null for the visual ones.
   */
  readonly depthNode: TSLNode | null;
  /** The forwarding leaves the graph was built over (introspection/tests). */
  readonly inputs: Readonly<Record<string, TSLNode>>;
  /** The full configuration key. */
  readonly key: string;
}

interface FrameLike {
  readonly material?: object | null;
}

/** Drawn material → its own leaf set (the forwarding source). */
const leavesByMaterial = new WeakMap<object, Record<string, TSLNode | undefined>>();
const registeredMaterials = new WeakSet<NodeMaterial>();
const forwardedInputsByMaterial = new WeakMap<NodeMaterial, Set<TSLNode>>();
const ownerByInput = new WeakMap<TSLNode, string>();
const standInByInput = new WeakMap<TSLNode, unknown>();
/** Configuration key → shared graph. Bounded by the configuration space. */
const graphs = new Map<string, SharedTSLGraph>();

function isTextureLeaf(node: TSLNode): boolean {
  return (node as { isTextureNode?: boolean } | null)?.isTextureNode === true;
}

function leafEntries(leaves: TSLLeafSet): [string, TSLNode][] {
  return Object.entries(leaves as Record<string, TSLNode | undefined>)
    .filter((entry): entry is [string, TSLNode] => entry[1] !== undefined && entry[1] !== null)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

/**
 * The type signature of one leaf: what the build derives code or a binding
 * layout from. A texture's sample type follows its data type/format (and
 * depth-ness), so two leaves with different textures of one signature
 * generate the same shader. Every texture field `standInTexture` copies is
 * here, since the build sees the first material's values through it.
 */
function leafSignature(name: string, node: TSLNode): string {
  if (isTextureLeaf(node)) {
    const tex = (node.value ?? {}) as Partial<THREE.DepthTexture> & { isDepthTexture?: boolean };
    const depth = tex.isDepthTexture === true ? `d${tex.compareFunction}` : 'c';
    return `${name}:tex:${tex.type}:${tex.format}:${depth}:${tex.colorSpace}:${tex.minFilter}:${tex.magFilter}:${tex.wrapS}:${tex.wrapT}`;
  }
  return `${name}:${(node as { nodeType?: string | null }).nodeType ?? '?'}`;
}

/** The full cache key of a configuration over a leaf set. */
export function sharedTSLGraphKey(family: string, key: unknown, leaves: TSLLeafSet): string {
  const sig = leafEntries(leaves)
    .map(([name, node]) => leafSignature(name, node))
    .join(',');
  return `${family}|${JSON.stringify(key)}|${sig}`;
}

function cloneValue(value: unknown): unknown {
  const v = value as { clone?: () => unknown; isTexture?: boolean } | null;
  // A texture is a REFERENCE (the binding); a Vector/Matrix/Color is a value.
  if (v && typeof v === 'object' && v.isTexture !== true && typeof v.clone === 'function') {
    return v.clone();
  }
  return value;
}

/** A small, independent image with the same sample type as the real texture. */
function standInTexture(source: THREE.Texture): THREE.Texture {
  let standIn: THREE.Texture;
  if (source instanceof THREE.DepthTexture) {
    standIn = new THREE.DepthTexture(1, 1, source.type);
  } else if (source instanceof THREE.DataTexture) {
    const pixels = source.image.data;
    const ArrayType = pixels?.constructor as Uint8ArrayConstructor | undefined;
    const data = ArrayType ? new ArrayType(4) : new Uint8Array(4);
    standIn = new THREE.DataTexture(data, 1, 1, source.format as THREE.PixelFormat, source.type);
  } else {
    standIn = new THREE.Texture();
  }
  standIn.type = source.type;
  standIn.format = source.format;
  standIn.colorSpace = source.colorSpace;
  standIn.minFilter = source.minFilter;
  standIn.magFilter = source.magFilter;
  standIn.wrapS = source.wrapS;
  standIn.wrapT = source.wrapT;
  standIn.generateMipmaps = source.generateMipmaps;
  if (source instanceof THREE.DepthTexture && standIn instanceof THREE.DepthTexture) {
    standIn.compareFunction = source.compareFunction;
  }
  return standIn;
}

/**
 * Forwarding twin of one leaf: same type, and before every draw it takes the
 * value of the drawn material's same-named leaf. A material with no
 * registered leaf of that name (not one of ours) leaves the value as is.
 *
 * `updateBefore`, not `update`: every `.sample()` / `.load()` tap of a
 * texture is a CLONE reading the base through `referenceNode`, and the
 * renderer runs every `updateBefore` ahead of any `update` — so all taps
 * see one texture (`live-texture-tsl.ts` documents the failure otherwise).
 */
function forwardingLeaf(name: string, template: TSLNode): TSLNode {
  // `uniform` is typed per value kind; the leaf's own value picks the kind.
  const makeUniform = uniform as (value: unknown, type?: string | null) => TSLNode;
  const standIn = isTextureLeaf(template)
    ? standInTexture(template.value as THREE.Texture)
    : cloneValue(template.value);
  const node: TSLNode = isTextureLeaf(template)
    ? texture(standIn as THREE.Texture)
    : makeUniform(standIn, (template as { nodeType?: string | null }).nodeType);
  standInByInput.set(node, standIn);
  node.updateBeforeType = NodeUpdateType.OBJECT;
  node.updateBefore = (frame: FrameLike): undefined => {
    const material = frame.material;
    if (!material) return;
    const source = leavesByMaterial.get(material)?.[name];
    if (source) {
      node.value = source.value;
      ownerByInput.set(node, (material as NodeMaterial).uuid);
      let forwarded = forwardedInputsByMaterial.get(material as NodeMaterial);
      if (!forwarded) {
        forwarded = new Set<TSLNode>();
        forwardedInputsByMaterial.set(material as NodeMaterial, forwarded);
      }
      forwarded.add(node);
    }
  };
  return node;
}

/** Forwarding twins of a whole leaf set (absent leaves stay absent). */
export function forwardTSLLeaves<T extends TSLLeafSet>(leaves: T): T {
  const out: Record<string, TSLNode> = {};
  for (const [name, node] of leafEntries(leaves)) out[name] = forwardingLeaf(name, node);
  return out as T;
}

/**
 * Point `material` at the shared graph of its configuration, building it on
 * first use, and register `leaves` as the values its draws forward.
 *
 * `leaves` is held by reference: a later in-place leaf swap (a fresh
 * `texture()` node assigned into the same record) is picked up by the next
 * draw, but a swap that changes a leaf's SIGNATURE or the key must call this
 * again (the wrappers do, from `rebuildGraph`).
 *
 * @param build - runs the geometry's factory over the forwarding leaves into
 *   the given scratch material; called once per key.
 * @returns the shared graph now on `material`.
 */
export function applySharedTSLGraph<T extends TSLLeafSet>(
  material: NodeMaterial,
  family: string,
  key: unknown,
  leaves: T,
  build: (inputs: T, scratch: NodeMaterial) => void
): SharedTSLGraph {
  const fullKey = sharedTSLGraphKey(family, key, leaves);
  let graph = graphs.get(fullKey);
  if (!graph) {
    const inputs = forwardTSLLeaves(leaves);
    const scratch = new NodeMaterial();
    build(inputs, scratch);
    graph = {
      vertexNode: scratch.vertexNode,
      colorNode: scratch.colorNode,
      depthNode: (scratch.depthNode as TSLNode | null | undefined) ?? null,
      inputs: inputs as Record<string, TSLNode>,
      key: fullKey,
    };
    graphs.set(fullKey, graph);
  }
  leavesByMaterial.set(material, leaves as Record<string, TSLNode | undefined>);
  if (!registeredMaterials.has(material)) {
    registeredMaterials.add(material);
    material.addEventListener('dispose', () => {
      if ((material as unknown as Record<symbol, boolean>)[SOFT_DISPOSE_FLAG]) return;
      leavesByMaterial.delete(material);
      for (const input of forwardedInputsByMaterial.get(material) ?? []) {
        if (ownerByInput.get(input) === material.uuid) {
          input.value = standInByInput.get(input);
          ownerByInput.delete(input);
        }
      }
      forwardedInputsByMaterial.delete(material);
    });
  }
  material.vertexNode = graph.vertexNode;
  material.colorNode = graph.colorNode;
  material.depthNode = graph.depthNode;
  return graph;
}

/** Number of distinct shared graphs built so far (tests / diagnostics). */
export function sharedTSLGraphCount(): number {
  return graphs.size;
}

/** The shared graph a material was last pointed at, if any (tests / diagnostics). */
export function getSharedTSLGraph(material: NodeMaterial): SharedTSLGraph | undefined {
  for (const graph of graphs.values()) {
    if (graph.colorNode === material.colorNode && graph.vertexNode === material.vertexNode) {
      return graph;
    }
  }
  return undefined;
}
