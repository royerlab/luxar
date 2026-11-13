# rendering/gpu-depth-sort

> Back-to-front depth ordering on the GPU for native WebGPU (#3162)

## Purpose

Order-dependent blending (normal, volumetric) draws instanced elements back to
front through the ordering pair `aSortedIndex` / `aSortedIndexB`
(`rendering/element-storage.ts`). On WebGL that order comes from the CPU
SortWorker (`workers/sort-worker/`) and is uploaded on every re-sort: 40 MB per
sort for a 10M-element node. On WebGPU the upload, the worker round trip, and
the garbage they leave are what made large sorted scenes slower than WebGL
(#3162: frame p95 about 2x WebGL's, CPU time about 2x).

This module computes the same order on the GPU instead. It reads each
element's center from the element texture the geometry already uploaded, and
writes the order into the ordering attribute's own GPU buffer, so a re-sort
moves no data across the CPU.

The sort itself is a standalone WebGPU module that knows nothing about
three.js or the depth-sort coordinator. Two pieces connect it:

- `three-webgpu-bridge.ts` is the only file that reaches into three's
  WebGPURenderer. It gets the device, and brings a texture or an attribute up
  to date through three's own update paths before handing over its GPU object,
  so the sort reads the committed texels and three never re-uploads over its
  result.
- `depth-sort-coordinator/gpu-sort-producer.ts` is the coordinator's GPU
  producer of orderings, alongside the SortWorker.

The GPU sort is off by default (`config.depthSort.gpuSort`), and `?gpuSort=1`
or `?gpuSort=0` overrides it. It runs only on a native WebGPU device. Mesh
nodes, nodes the device cannot hold, and every other backend keep the
SortWorker, as do all nodes while the pipelines compile or after a device
loss.

## File map

```
gpu-depth-sort/
├── gpu-depth-sorter.ts — GpuDepthSorter: compile once per device, sort(request), scratch, dispose
├── three-webgpu-bridge.ts — The device and up-to-date GPU resources of a three.js WebGPURenderer
├── pipelines.ts        — The compute pipelines and their explicit bind-group layouts
├── scratch.ts          — Scratch buffers, uniform slots and bind groups for one capacity
├── plan.ts             — Pure sizing: blocks, scan levels, device limits, uniform packing
├── wgsl.ts             — The WGSL of every pass
└── gpu-types.ts        — The WebGPU API surface used, typed structurally
```

## Algorithm

All passes run in one compute pass and one submit per sort:

1. **depth**: the view-space z of each element, plus the min/max over the
   elements in front of the camera.
2. **keys**: the 16-bit back-to-front key.
3. **Four 4-bit radix passes**, least significant digit first. Each pass:
   - a block histogram;
   - a multi-level exclusive scan of the digit-major histograms;
   - a stable scatter.

   In the scatter, each invocation owns 16 consecutive elements. An element's
   destination is its block's offset for its digit, plus the count of that
   digit in earlier invocations, plus its count among the invocation's own
   earlier elements. A three-barrier scan of packed 16-bit counters gives the
   earlier-invocation counts. Because each invocation owns consecutive
   elements, the scatter is stable without sorting inside the workgroup.

4. **copy**: the final order is copied into the target ordering attribute. The
   attribute therefore keeps three's own buffer usage (VERTEX, COPY_DST) and
   needs no storage binding.

Blocks hold 4,096 elements. Scratch for N elements is four 4N-byte buffers plus
a small histogram: about 160 MB at 10M. One sorter's scratch is shared by every
node it sorts. Each sort writes its uniforms and submits before the next sort,
and the queue runs them in order. Scratch grows by doubling.

**Measured cost:** about 9 ms of GPU time for a 10M-element sort on an RTX 3070
(`gpu-depth-sort.spec.ts`, back-to-back sorts), against 30–100 ms for the
SortWorker plus 12–28 ms of main-thread upload.

## Parity with the CPU kernel

The keys follow `wasm/typescript/depth-sort.ts`:

- view z is `m2·x + m6·y + m10·z + m14`;
- `zMin` / `zMax` are taken over z < 0;
- a degenerate range keys every element 0, so the stable sort returns the
  identity order, like the CPU's identity fallback;
- elements behind the camera get key 0.

WGSL may fuse a multiply-add, and its division is accurate to 2.5 ULP (refined
here by one Newton step), so a key near a bucket edge can land one bucket away
from the CPU's.

The contract is therefore:

- the GPU order is an exact, stable sort of the GPU's own keys;
- each GPU key is within ±1 bucket of the CPU key;
- the result is deterministic for a given device.

A NaN center is implementation-defined. On the RTX 3070, about 3 in 10,000
elements of a 10M sort sit one bucket from the CPU's position.

## Limits

`planScratch` refuses a capacity whose largest binding exceeds
`maxStorageBufferBindingSize` (33.5M elements at the 128 MiB default), or whose
widest dispatch exceeds `maxComputeWorkgroupsPerDimension`.
`GpuDepthSorter.accepts(count)` reports whether a node fits. A node that does
not fit keeps the SortWorker.

## Tests

- `tests/unit/rendering/gpu-depth-sort/plan.test.ts`: the sizing and packing
  math.
- `tests/unit/rendering/gpu-depth-sort/gpu-depth-sorter.test.ts`: the
  orchestration against a recording fake device. It covers:
  - the dispatch sequence and widths;
  - that the WGSL bindings match the layouts;
  - the copy, the uniform writes, scratch growth and disposal;
  - compiler-message reporting.
- `tests/e2e/gpu-depth-sort.spec.ts` (native WebGPU only,
  `pnpm test:e2e:webgpu`): sorts on a real device through
  `/gpu-sort-harness.html` and checks the contract above against
  `sort_splats_by_depth`. It covers:
  - sizes from 1 to 10M, including block and scan edges;
  - the three texel layouts;
  - straddling, behind-camera and flat scenes.
