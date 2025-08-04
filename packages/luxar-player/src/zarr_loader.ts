import * as zarr from "zarrita";
import { get } from "zarrita";
import * as THREE from "three";
import { createGaussianPointMaterial } from "./shader-manager";

/* ------------------------------------------------------------------ utils */
function toURL(path: string) {
  const abs = path.startsWith("http")
    ? path
    : new URL(path.replace(/^\/?/, "/"), window.location.origin).toString();
  return abs.endsWith("/") ? abs : abs + "/";
}

type ObjRecord = { obj: THREE.Object3D; path: string };

interface ZarrGroupAttrs {
  type?: string;
  transform?: number[];
}

/* ------------------------------------------------------------------ main */
export async function loadScene(src: string): Promise<THREE.Group> {
  /* 1. open the store (use consolidated metadata if present) */
  const rawStore = new zarr.FetchStore(toURL(src));
  const store = await zarr.tryWithConsolidated(rawStore); // falls back gracefully

  /* 2. enumerate every node once */
  const listing = await store.contents(); // [{ path: "/", kind: "group" }, …]

  const rootLoc   = zarr.root(store);
  const rootThree = new THREE.Group();
  const lookup    = new Map<string, ObjRecord>([["/", { obj: rootThree, path: "/" }]]);

  /* 3. create Three.js objects in path-depth order */
  const groups = listing.filter((e: { kind: string; path: string }) => e.kind === "group" && e.path !== "/")
                        .sort((a: { path: string }, b: { path: string }) => a.path.split("/").length - b.path.split("/").length);

  for (const entry of groups) {
    const loc   = rootLoc.resolve(entry.path.slice(1));   // drop leading "/"
    const grp   = await zarr.open(loc, { kind: "group" });
    const attrs = grp.attrs as ZarrGroupAttrs;

    /* build renderable */
    let obj: THREE.Object3D;
    if (attrs?.type === "points") {
      obj = await buildPoints(loc);
    } else {
      obj = new THREE.Group();
    }

    if (Array.isArray(attrs?.transform) && attrs.transform.length === 16) {
      obj.applyMatrix4(new THREE.Matrix4().fromArray(attrs.transform));
    }

    /* attach to parent in Three.js graph */
    const parentPath = entry.path.substring(0, entry.path.lastIndexOf("/")) || "/";
    lookup.get(parentPath)!.obj.add(obj);
    lookup.set(entry.path, { obj, path: entry.path });
  }

  return rootThree;
}

/* ---------------------------------------------------------------- geometry */
async function buildPoints(loc: zarr.Location<zarr.Readable>): Promise<THREE.Points> {
  const posArr = await zarr.open(loc.resolve("positions"), { kind: "array" });
  const pos    = (await get(posArr)).data as Float32Array;

  let col: Uint8Array | undefined;
  try {
    const colArr = await zarr.open(loc.resolve("colors"), { kind: "array" });
    col = (await get(colArr)).data as Uint8Array;
  } catch {/* optional */}

  let radii: Float32Array | undefined;
  try {
    const radiiArr = await zarr.open(loc.resolve("radii"), { kind: "array" });
    radii = (await get(radiiArr)).data as Float32Array;
  } catch {/* optional */}

  let sharpness: Float32Array | undefined;
  try {
    const sharpnessArr = await zarr.open(loc.resolve("sharpness"), { kind: "array" });
    sharpness = (await get(sharpnessArr)).data as Float32Array;
  } catch {/* optional */}

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  
  if (col) {
    // Convert Uint8Array colors (0-255) to Float32Array (0.0-1.0) for shader compatibility
    const floatColors = new Float32Array(col.length);
    for (let i = 0; i < col.length; i++) {
      floatColors[i] = col[i] / 255.0;
    }
    geom.setAttribute("color", new THREE.BufferAttribute(floatColors, 3));
  } else {
    // Provide default white colors if no color data is available
    // This ensures the vertex shader always has color data to work with
    const numVertices = pos.length / 3;
    const defaultColors = new Float32Array(numVertices * 3);
    defaultColors.fill(1.0); // All white (RGB = 1.0, 1.0, 1.0)
    geom.setAttribute("color", new THREE.BufferAttribute(defaultColors, 3));
  }

  if (radii) {
    // Add radius attribute for custom shaders
    geom.setAttribute("radius", new THREE.BufferAttribute(radii, 1));
  } else {
    // Provide default radii if not specified
    const numVertices = pos.length / 3;
    const defaultRadii = new Float32Array(numVertices);
    defaultRadii.fill(0.1); // Default radius
    geom.setAttribute("radius", new THREE.BufferAttribute(defaultRadii, 1));
  }

  if (sharpness) {
    // Add sharpness attribute for custom shaders
    geom.setAttribute("sharpness", new THREE.BufferAttribute(sharpness, 1));
  } else {
    // Provide default sharpness if not specified
    const numVertices = pos.length / 3;
    const defaultSharpness = new Float32Array(numVertices);
    defaultSharpness.fill(2.0); // Default sharpness (quadratic falloff)
    geom.setAttribute("sharpness", new THREE.BufferAttribute(defaultSharpness, 1));
  }

  // Create advanced Gaussian point material with HDR output and bloom effects
  // This replaces the basic PointsMaterial with custom shaders for smooth, natural-looking points
  const material = createGaussianPointMaterial();
  
  return new THREE.Points(geom, material);
}
