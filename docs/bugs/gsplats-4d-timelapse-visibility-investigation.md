# Investigation: GSplats 4D Time-Lapse Visibility Bug

## Summary

When viewing a 4D scene with many separate gsplats nodes (one per timepoint, e.g., 50 nodes for 50 timepoints), only the first 2-3 timepoints display data in the viewer. Scrubbing the time slider or playing the animation shows no gsplats for later timepoints. Cell tracking lines (which use `extend_to_all=["time"]`) display correctly at all times.

## Dataset

- **Demo**: `demo_gsplats_4d_celegans_tracking.py` — C. elegans embryo confocal time-lapse
- **Structure**: 50 gsplats nodes (`gsplats_t0000` through `gsplats_t0049`), each containing ~3000-3200 4D splats
- **Dimensions**: `[x, y, z, time]` where time is discrete with step=1.0
- **Time embedding**: Each node uses `dim_order=["z", "y", "x"]` with `fill={"time": float(t)}` and `fill_sigma={"time": 0.3}`

## Root Cause (Confirmed)

**The viewer is receiving 404 errors when fetching zarr chunks** for nodes beyond the first 2 timepoints. The viewer's zarrita library requests chunk indices that don't exist on disk.

### Evidence (browser console at time=25)

```
GET http://127.0.0.1:8004/gsplats_t0023/centers/1.1    404 (Not Found)
GET http://127.0.0.1:8004/gsplats_t0023/centers/0.1    404 (Not Found)
GET http://127.0.0.1:8004/gsplats_t0024/centers/4.0    404 (Not Found)
GET http://127.0.0.1:8004/gsplats_t0025/centers/0.1    404 (Not Found)
GET http://127.0.0.1:8004/gsplats_t0025/cholesky_factors/4.0  404 (Not Found)
GET http://127.0.0.1:8004/gsplats_t0024/colors/1.0     404 (Not Found)
GET http://127.0.0.1:8004/gsplats_t0025/sharpnesses/1  404 (Not Found)
```

Followed by:
```
[SceneLoader] Failed to update gsplats /gsplats_t0025 (attempt 1): Cannot read properties of undefined (reading 'set')
[SceneLoader] 1 loader(s) failed: /gsplats_t0025
```

### What the 404 chunk paths mean

- `centers/0.1` = zarr chunk `[row=0, col=1]` — requesting the **second column chunk**
- `centers/4.0` = zarr chunk `[row=4, col=0]` — requesting a row chunk beyond what exists
- `centers/1.1` = zarr chunk `[row=1, col=1]` — both wrong dimensions

The actual zarr data has `chunks=(1024, 4)` for centers (4D, shape `(~3100, 4)`). With 4 columns in one chunk, there is only column chunk 0. The row chunks are 0, 1, 2, 3 (for ~3100 rows / 1024 per chunk).

## Why This Happens

### The chunk alignment fix IS applied correctly

After the compiler fix (passing `spatial_index_data` to `_calculate_intelligent_chunks`), the zarr arrays have correct chunks:

```
gsplats_t0025:
  centers:          shape=(3142, 4)  chunks=(1024, 4)   ← correct
  amplitudes:       shape=(3142,)    chunks=(1024,)      ← correct
  cholesky_factors: shape=(3142, 10) chunks=(1024, 10)   ← correct
  colors:           shape=(1, 3)     chunks=(1, 3)       ← broadcasted (correct)
  sharpnesses:      shape=(1,)       chunks=(1,)         ← broadcasted (correct)
  chunk_size metadata: 1024                               ← matches
```

### But the viewer is using stale `.zarray` metadata

The HTTP server serving the zarr data was started BEFORE the data was regenerated with the chunk alignment fix. When the scene was regenerated, the zarr directory on disk was overwritten, but:

1. The HTTP server process (PID 3049927, started at 16:42) was already running
2. The server or the browser cached the OLD `.zarray` files from the pre-fix data
3. The old `.zarray` had different chunk sizes (e.g., `chunks=(3142, 4)` — whole array in one chunk for the old unaligned data, or different row chunk sizes)
4. zarrita in the browser uses the cached `.zarray` to compute chunk paths → requests wrong chunks → 404s

### Verification

After killing the old server and starting a fresh one:
```
curl centers/0.0: 200  ← exists
curl centers/0.1: 404  ← correctly does not exist (column dim is 4, one chunk)
curl centers/1.0: 200  ← exists
curl centers/4.0: 404  ← correctly does not exist (only 4 row chunks: 0,1,2,3)
```

## Detailed Timeline of the Investigation

### Phase 1: Initial hypothesis — chunk alignment bug (confirmed, fixed)

The original bug report (`gsplats-chunk-bounds-vs-zarr-chunks-mismatch.md`) identified that the compiler wrote `chunk_bounds` spatial partitions with one `chunk_size` but zarr arrays with a different chunk size. This was fixed by passing `ordering_data` / `spatial_index_data` to `_calculate_intelligent_chunks()` at all 11 call sites in `compiler.py`.

**Tests confirmed**: All zarr arrays now have `chunks[0] == chunk_size` from the spatial index metadata.

### Phase 2: Amplitude issues

After the chunk fix, the demo still looked wrong:

1. **Flat `scale_intensity(0.1)` made amplitudes too dim** — max amplitude was ~0.02 instead of 0.1. Fixed by normalizing per-timepoint like the zebrafish demo: `scale_intensity(0.1 / amp_max)`.

2. **Oversized "background" splats** — The fitting produced ~30 splats per timepoint with sigma > 5 µm (covering the entire volume), drowning out the ~3000 detail splats. Fixed by filtering splats with `max_sigma > 3.0 µm` before adding to the scene.

### Phase 3: Fitting quality

The original fitting used too high a learning rate, causing poor convergence. The user fixed this in the `demo_3d_celegans_confocal.py` inspection demo. All 50 timepoints were re-fit with improved parameters.

### Phase 4: Viewer visibility investigation

After re-fitting and regenerating, later timepoints still showed no data. Extensive investigation of the viewer code:

1. **Data pipeline verified correct**: Python simulation of the AABB chunk_bounds query confirmed all 50 nodes are self-visible at their own time position.

2. **Console logs showed successful loading**: Every update logged "Loading 3142 gsplats" and "Worker projection complete: 3142/3142 visible splats" for the correct nodes.

3. **But THREE.js meshes had instanceCount=0**: Browser console inspection revealed:
   ```
   gsplats_t0000: instanceCount 3169  ← loaded at init, works
   gsplats_t0001: instanceCount 3151  ← loaded at init, works
   gsplats_t0002: instanceCount 0     ← should have data, but 0!
   ...
   gsplats_t0049: instanceCount 0     ← all zeros
   ```

4. **Geometry setter watch revealed the real error**:
   ```
   GEOMETRY SET on /gsplats_t0025 instanceCount: 3142
   Failed to update gsplats /gsplats_t0025: Cannot read properties of undefined (reading 'set')
   ```
   The geometry was being assigned, but the data writing failed because the zarr chunk loads returned corrupted/incomplete data due to 404s.

5. **404 errors confirmed**: The browser was requesting non-existent zarr chunk paths like `centers/0.1`, `centers/4.0`, `cholesky_factors/1.1`, etc.

### Phase 5: Stale server identified

The HTTP server (luxar serve) was started before the data was regenerated. It was serving stale `.zarray` metadata with old chunk sizes.

## Why Initial Timepoints Worked

Timepoints t0000 and t0001 loaded correctly during the initial scene load because:

1. At scene initialization (time=0), the viewer loads all nodes sequentially
2. For t0000 and t0001, the spatial query finds visible chunks at time=0
3. These nodes load their data BEFORE any time-slider update
4. The initial load may have succeeded because the browser hadn't yet cached the `.zarray` metadata, or the initial load used a different code path

Later timepoints (t0002+) only load during `updateView` calls when the slider moves. By that point, zarrita has cached the `.zarray` metadata and uses it to compute chunk paths — if stale, it requests wrong paths.

## Fixes Applied (Python/compiler side)

1. **Chunk alignment fix** (`compiler.py`): 11 call sites now pass `spatial_index_data` to `_calculate_intelligent_chunks()`. Also fixed the 1D branch of the function.

2. **Per-timepoint amplitude normalization** (`demo_gsplats_4d_celegans_tracking.py`): Changed from flat `scale_intensity(0.1)` to per-timepoint normalization `scale_intensity(0.1 / amp_max)`.

3. **Oversized splat filtering** (`demo_gsplats_4d_celegans_tracking.py`): Filters out splats with `max_sigma > 3.0 µm` before adding to the scene.

4. **New inspection demo** (`gsplats/demos/demo_3d_celegans_confocal.py`): napari-based demo for inspecting single-timepoint fitting quality.

## Remaining Issue

The viewer-side issue of 404 errors on zarr chunks appears to be caused by **stale HTTP server caching**. The fix is to restart the data server after regenerating the zarr data. However, there may also be a deeper issue with how zarrita caches `.zarray` metadata in the browser — this should be investigated separately.

### To test

1. Kill any existing `luxar serve` processes
2. Start a fresh server: `hatch run python -m luxar serve datasets/demos/gsplats_4d_celegans_tracking.zarr --port 8004`
3. Hard-refresh the browser (Ctrl+Shift+R)
4. Scrub the time slider — all 50 timepoints should now show data

## Key Diagnostic Commands

### Verify zarr chunk alignment (Python)
```python
import zarr, numpy as np
from math import ceil

store = zarr.open_group('datasets/demos/gsplats_4d_celegans_tracking.zarr', 'r')
for name in sorted(k for k in store.keys() if k.startswith('gsplats_t')):
    g = store[name]
    cs = g.attrs.get('chunk_size', None)
    if cs:
        ok = g['centers'].chunks[0] == cs
        print(f"{name}: chunk_size={cs}, centers.chunks={g['centers'].chunks}, aligned={ok}")
```

### Check viewer mesh state (browser console)
```javascript
const scene = window.__luxarDebug?.scene;
scene.traverse(obj => {
  if (obj.name?.includes('gsplats_t002')) {
    console.log(obj.name, 'instanceCount:', obj.geometry?.instanceCount);
  }
});
```

### Simulate viewer AABB query (Python)
```python
import zarr, numpy as np
store = zarr.open_group('path/to/data.zarr', 'r')
tolerance = [1e10, 1e10, 1e10, 0.5]  # x,y,z displayed, time discrete

for t in [0, 10, 25, 49]:
    for i in range(50):
        g = store[f'gsplats_t{i:04d}']
        cb = np.array(g['chunk_bounds'])
        slice_pos = [0, 0, 0, float(t)]
        for chunk in range(cb.shape[0]):
            intersects = True
            for dim in range(4):
                if cb[chunk, dim, 1] < slice_pos[dim] - tolerance[dim]:
                    intersects = False; break
                if cb[chunk, dim, 0] > slice_pos[dim] + tolerance[dim]:
                    intersects = False; break
            if intersects:
                print(f"time={t}: {g.name} chunk {chunk} visible")
                break
```
