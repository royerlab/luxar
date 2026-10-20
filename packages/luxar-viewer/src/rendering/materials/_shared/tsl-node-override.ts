/** Per-material overrides for Luxar's shared TSL geometry graphs. */
import type { Material } from 'three';
import type { Node, NodeMaterial } from 'three/webgpu';

export type TSLMaterialNodes = Readonly<Pick<NodeMaterial, 'vertexNode' | 'depthNode'>> & {
  /** Luxar's built-in graph always produces a four-component output. */
  readonly colorNode: Node<'vec4'>;
};

/** Return replacement output nodes; omitted outputs retain the Luxar graph. */
export type TSLNodeOverride = (base: TSLMaterialNodes) => Partial<TSLMaterialNodes>;

const baseNodes = new WeakMap<NodeMaterial, TSLMaterialNodes>();
const overrides = new WeakMap<NodeMaterial, TSLNodeOverride>();

function applyOverride(
  material: NodeMaterial,
  base: TSLMaterialNodes,
  override?: TSLNodeOverride
): void {
  // Resolve the callback before changing any output, so a throwing override
  // leaves the material on its complete, unmodified Luxar graph.
  const replacement = override?.(base);
  material.vertexNode =
    replacement?.vertexNode === undefined ? base.vertexNode : replacement.vertexNode;
  material.colorNode =
    replacement?.colorNode === undefined ? base.colorNode : replacement.colorNode;
  material.depthNode =
    replacement?.depthNode === undefined ? base.depthNode : replacement.depthNode;
}

/** Called by the shared graph builder whenever a material changes variant. */
export function bindTSLNodeOverride(material: NodeMaterial, base: TSLMaterialNodes): void {
  baseNodes.set(material, base);
  try {
    applyOverride(material, base, overrides.get(material));
  } catch (error) {
    // A failed extension must not leave the previous variant's graph bound
    // after Luxar has already switched the material's configuration.
    applyOverride(material, base);
    throw error;
  }
}

/**
 * Replace output nodes of a built-in visual TSL Points, Lines, GSplats, or Mesh
 * material (excluding physical meshes and picking materials).
 * The callback sees the latest Luxar graph on registration and every rebuild;
 * return only the outputs to replace. Passing null restores the base graph.
 * The callback is retained when the material is cloned.
 * If the callback throws during a rebuild, that variant uses its base graph;
 * the callback stays registered and runs again on the next rebuild.
 * Vertex and depth overrides affect the visual pass only: picking continues
 * with Luxar's built-in positions and depth.
 */
export function setTSLNodeOverride(material: Material, override: TSLNodeOverride | null): void {
  const nodeMaterial = material as NodeMaterial;
  const base = baseNodes.get(nodeMaterial);
  if (!base) throw new Error('setTSLNodeOverride requires a Luxar shared-graph TSL material');
  applyOverride(nodeMaterial, base, override ?? undefined);
  if (override) overrides.set(nodeMaterial, override);
  else overrides.delete(nodeMaterial);
  nodeMaterial.needsUpdate = true;
}

/** Carry an opt-in override onto a wrapper's newly constructed clone. */
export function copyTSLNodeOverride(source: NodeMaterial, target: NodeMaterial): void {
  const override = overrides.get(source);
  if (override) setTSLNodeOverride(target, override);
}
