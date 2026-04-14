#!/usr/bin/env python3
"""
Luxar Viewer Performance Benchmark
===================================

Measures real rendering performance of the Luxar web viewer using Playwright.

Metrics collected per dataset:
  - Time to first render (page load -> scene initialized with geometry visible)
  - Idle FPS (scene loaded, no interaction, continuous rAF loop for 3 s)
  - Interaction FPS (programmatic camera orbit for 3 s)
  - JS heap memory usage (via performance.memory, Chromium only)

Usage:
    python run_benchmark.py                       # Run all datasets
    python run_benchmark.py --datasets small       # Preset: small datasets only
    python run_benchmark.py --datasets large       # Preset: large datasets only
    python run_benchmark.py --repeats 5            # 5 repetitions per dataset
    python run_benchmark.py --headed               # Show browser window
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LUXAR_ROOT = Path(__file__).resolve().parents[3]  # .../luxar
DEMOS_DIR = LUXAR_ROOT / "datasets" / "demos"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Datasets to benchmark, ordered roughly by element count.
# Each tuple: (zarr_name, label, expected_geometry_type)
DATASET_CATALOG = [
    # Small gsplats (< 10K)
    ("gsplats_3d_cells3d_multichannel.zarr", "cells3d-multichannel (8K gsplats)", "gsplats"),
    # Medium gsplats (10-50K)
    ("gsplats_3d_tribolium_embryo.zarr", "tribolium-embryo (44K gsplats)", "gsplats"),
    # Large gsplats (50-100K)
    ("gsplats_3d_kidney_multichannel_layers.zarr", "kidney-layers (68K gsplats)", "gsplats"),
    # Very large gsplats (100K+)
    ("gsplats_3d_opencell_map4.zarr", "opencell-map4 (102K gsplats)", "gsplats"),
    # Points (500K)
    ("lorenz.zarr", "lorenz-attractor (500K points)", "points"),
    ("spiral_galaxy.zarr", "spiral-galaxy (500K points)", "points"),
    # Mixed points + lines
    ("forest.zarr", "forest (672K pts+lines)", "mixed"),
    # Mega gsplats (10M)
    ("storm_3d_microtubules.zarr", "storm-microtubules (10M gsplats)", "gsplats"),
]

PRESETS = {
    "small": [d for d in DATASET_CATALOG if "8K" in d[1] or "44K" in d[1]],
    "medium": [d for d in DATASET_CATALOG if "68K" in d[1] or "102K" in d[1]],
    "large": [d for d in DATASET_CATALOG if "500K" in d[1] or "672K" in d[1] or "10M" in d[1]],
    "gsplats": [d for d in DATASET_CATALOG if d[2] == "gsplats"],
    "points": [d for d in DATASET_CATALOG if d[2] in ("points", "mixed")],
    "all": DATASET_CATALOG,
    "quick": DATASET_CATALOG[:4],  # First 4 for fast iteration
}

# Benchmark timing parameters (seconds)
IDLE_MEASURE_DURATION_S = 3.0
INTERACTION_MEASURE_DURATION_S = 3.0
WARMUP_FRAMES = 30  # Frames to skip before measuring


@dataclass
class BenchmarkResult:
    dataset: str
    label: str
    geometry_type: str
    total_elements: int = 0
    total_gsplats: int = 0
    total_points: int = 0
    load_time_ms: float = 0.0
    # rAF-based FPS (vsync-capped, reflects real user experience)
    idle_fps_mean: float = 0.0
    idle_fps_std: float = 0.0
    idle_fps_min: float = 0.0
    idle_fps_max: float = 0.0
    idle_fps_median: float = 0.0
    idle_frame_times_ms: list = field(default_factory=list)
    interaction_fps_mean: float = 0.0
    interaction_fps_std: float = 0.0
    interaction_fps_min: float = 0.0
    interaction_fps_max: float = 0.0
    interaction_fps_median: float = 0.0
    interaction_frame_times_ms: list = field(default_factory=list)
    # Uncapped render throughput (bypasses vsync, shows true GPU capacity)
    idle_throughput_fps: float = 0.0
    idle_render_time_ms_mean: float = 0.0
    idle_render_time_ms_median: float = 0.0
    interaction_throughput_fps: float = 0.0
    interaction_render_time_ms_mean: float = 0.0
    interaction_render_time_ms_median: float = 0.0
    # Memory and GPU info
    js_heap_used_mb: float = 0.0
    js_heap_total_mb: float = 0.0
    webgl_renderer: str = ""
    error: Optional[str] = None


def find_free_port(start: int = 18000, end: int = 19000) -> int:
    """Find an available port in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


def start_luxar_server(zarr_path: Path, data_port: int, viewer_port: int) -> subprocess.Popen:
    """Start `luxar serve <path> --viewer` and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "luxar", "serve",
        str(zarr_path),
        "--viewer",
        "--port", str(data_port),
        "--viewer-port", str(viewer_port),
        "--host", "127.0.0.1",
    ]
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(LUXAR_ROOT),
        preexec_fn=os.setsid,  # So we can kill the whole process group
    )
    return proc


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Poll until the HTTP server is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def kill_server(proc: subprocess.Popen) -> None:
    """Kill the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def compute_fps_stats(frame_times_ms: list[float]) -> dict:
    """Compute FPS statistics from per-frame durations in milliseconds."""
    if not frame_times_ms:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

    fps_values = [1000.0 / ft for ft in frame_times_ms if ft > 0]
    if not fps_values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

    return {
        "mean": statistics.mean(fps_values),
        "std": statistics.stdev(fps_values) if len(fps_values) > 1 else 0,
        "min": min(fps_values),
        "max": max(fps_values),
        "median": statistics.median(fps_values),
    }


# ---------------------------------------------------------------------------
# JavaScript snippets executed in the browser
# ---------------------------------------------------------------------------

JS_WAIT_FOR_READY = """
() => {
    const d = window.__luxarDebug;
    return d && d.runtimeReady && d.getState && d.getState().initialized;
}
"""

JS_WAIT_FOR_GEOMETRY = """
() => {
    const d = window.__luxarDebug;
    if (!d || !d.getState) return false;
    const s = d.getState();
    return s.totalElements > 0;
}
"""

JS_GET_STATE = """
() => {
    const d = window.__luxarDebug;
    return d.getState();
}
"""

JS_GET_MEMORY = """
() => {
    if (performance.memory) {
        return {
            usedJSHeapSize: performance.memory.usedJSHeapSize,
            totalJSHeapSize: performance.memory.totalJSHeapSize,
        };
    }
    return null;
}
"""

# Measure idle FPS using rAF timestamps (vsync-capped, shows real user experience)
JS_MEASURE_IDLE_FPS_RAF = """
(durationMs) => {
    return new Promise((resolve) => {
        const debug = window.__luxarDebug;
        const frameTimes = [];
        let lastTime = null;
        let warmupCount = 0;
        const WARMUP = %d;
        const startTime = performance.now();

        function onFrame(now) {
            if (warmupCount < WARMUP) {
                warmupCount++;
                lastTime = now;
                debug.renderOnce();
                requestAnimationFrame(onFrame);
                return;
            }

            if (lastTime !== null) {
                frameTimes.push(now - lastTime);
            }
            lastTime = now;

            if (performance.now() - startTime < durationMs + WARMUP * 16.7) {
                debug.renderOnce();
                requestAnimationFrame(onFrame);
            } else {
                resolve(frameTimes);
            }
        }

        debug.renderOnce();
        requestAnimationFrame(onFrame);
    });
}
""" % WARMUP_FRAMES

# Measure uncapped render throughput: tight synchronous loop bypassing vsync.
# Calls renderer.render() directly and uses gl.finish() to ensure GPU completes.
JS_MEASURE_RENDER_THROUGHPUT = """
(durationMs) => {
    const debug = window.__luxarDebug;
    const renderer = debug.renderer;
    const scene = debug.scene;
    const camera = debug.camera;
    const postProcessing = debug.postProcessing;
    const gl = renderer.getContext();

    // Warm up
    for (let i = 0; i < 10; i++) {
        if (postProcessing && postProcessing.render) {
            postProcessing.render();
        } else {
            renderer.render(scene, camera);
        }
        gl.finish();
    }

    // Measure: tight render loop, no rAF, no vsync gate
    const frameTimes = [];
    const startTime = performance.now();
    let frames = 0;

    while (performance.now() - startTime < durationMs) {
        const t0 = performance.now();
        if (postProcessing && postProcessing.render) {
            postProcessing.render();
        } else {
            renderer.render(scene, camera);
        }
        gl.finish();  // Force GPU to complete before timing
        const t1 = performance.now();
        frameTimes.push(t1 - t0);
        frames++;
    }

    const elapsedMs = performance.now() - startTime;
    return {
        frameTimes: frameTimes,
        totalFrames: frames,
        elapsedMs: elapsedMs,
        throughputFps: frames / (elapsedMs / 1000.0),
    };
}
"""

# Measure interaction throughput: tight render loop with camera orbit
JS_MEASURE_INTERACTION_THROUGHPUT = """
(durationMs) => {
    const debug = window.__luxarDebug;
    const renderer = debug.renderer;
    const scene = debug.scene;
    const camera = debug.camera;
    const postProcessing = debug.postProcessing;
    const controls = debug.controls;
    const gl = renderer.getContext();

    const orbitRadius = camera.position.length();
    const orbitSpeed = 2.0;  // radians per second
    let angle = Math.atan2(camera.position.x, camera.position.z);

    // Warm up
    for (let i = 0; i < 10; i++) {
        angle += 0.01;
        camera.position.x = orbitRadius * Math.sin(angle);
        camera.position.z = orbitRadius * Math.cos(angle);
        camera.lookAt(0, 0, 0);
        if (postProcessing && postProcessing.render) {
            postProcessing.render();
        } else {
            renderer.render(scene, camera);
        }
        gl.finish();
    }

    const frameTimes = [];
    const startTime = performance.now();
    let lastTime = startTime;
    let frames = 0;

    while (performance.now() - startTime < durationMs) {
        const now = performance.now();
        const dt = (now - lastTime) / 1000.0;
        lastTime = now;

        angle += orbitSpeed * Math.max(dt, 0.001);
        camera.position.x = orbitRadius * Math.sin(angle);
        camera.position.z = orbitRadius * Math.cos(angle);
        camera.lookAt(0, 0, 0);
        if (controls && controls.update) {
            controls.update();
        }

        const t0 = performance.now();
        if (postProcessing && postProcessing.render) {
            postProcessing.render();
        } else {
            renderer.render(scene, camera);
        }
        gl.finish();
        const t1 = performance.now();
        frameTimes.push(t1 - t0);
        frames++;
    }

    const elapsedMs = performance.now() - startTime;
    return {
        frameTimes: frameTimes,
        totalFrames: frames,
        elapsedMs: elapsedMs,
        throughputFps: frames / (elapsedMs / 1000.0),
    };
}
"""

# Measure rAF-based interaction FPS (vsync-capped)
JS_MEASURE_INTERACTION_FPS_RAF = """
(durationMs) => {
    return new Promise((resolve) => {
        const debug = window.__luxarDebug;
        const camera = debug.camera;
        const controls = debug.controls;
        const frameTimes = [];
        let lastTime = null;
        let warmupCount = 0;
        const WARMUP = %d;
        const startTime = performance.now();

        const orbitRadius = camera.position.length();
        const orbitSpeed = 0.5;
        let angle = Math.atan2(camera.position.x, camera.position.z);

        function onFrame(now) {
            if (warmupCount < WARMUP) {
                warmupCount++;
                lastTime = now;
                debug.renderOnce();
                requestAnimationFrame(onFrame);
                return;
            }

            if (lastTime !== null) {
                const dt = (now - lastTime) / 1000.0;
                angle += orbitSpeed * dt;
                camera.position.x = orbitRadius * Math.sin(angle);
                camera.position.z = orbitRadius * Math.cos(angle);
                camera.lookAt(0, 0, 0);
                if (controls && controls.update) {
                    controls.update();
                }
                frameTimes.push(now - lastTime);
            }
            lastTime = now;

            if (performance.now() - startTime < durationMs + WARMUP * 16.7) {
                debug.renderOnce();
                requestAnimationFrame(onFrame);
            } else {
                resolve(frameTimes);
            }
        }

        debug.renderOnce();
        requestAnimationFrame(onFrame);
    });
}
""" % WARMUP_FRAMES

# Get WebGL renderer info
JS_GET_WEBGL_INFO = """
() => {
    const debug = window.__luxarDebug;
    const renderer = debug.renderer;
    const gl = renderer.getContext();

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    return {
        vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'unknown',
        renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown',
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxViewportDims: Array.from(gl.getParameter(gl.MAX_VIEWPORT_DIMS)),
    };
}
"""


def run_single_benchmark(
    page,
    dataset_name: str,
    label: str,
    geom_type: str,
    viewer_url: str,
    data_url: str,
) -> BenchmarkResult:
    """Run benchmark for a single dataset using an existing Playwright page."""
    result = BenchmarkResult(dataset=dataset_name, label=label, geometry_type=geom_type)

    try:
        # Navigate to dataset
        full_url = f"{viewer_url}/?src={data_url}&debug"
        print(f"    Navigating to: {full_url}")

        t0 = time.time()
        page.goto(full_url, timeout=60000)

        # Wait for Luxar debug interface to be ready
        page.wait_for_function(JS_WAIT_FOR_READY, timeout=60000)

        # Wait for geometry to appear
        page.wait_for_function(JS_WAIT_FOR_GEOMETRY, timeout=120000)

        load_time_ms = (time.time() - t0) * 1000.0
        result.load_time_ms = load_time_ms
        print(f"    Load time: {load_time_ms:.0f} ms")

        # Get scene state
        state = page.evaluate(JS_GET_STATE)
        result.total_elements = state.get("totalElements", 0)
        result.total_gsplats = state.get("totalGSplats", 0)
        result.total_points = state.get("totalPoints", 0)
        print(f"    Elements: {result.total_elements} (gsplats={result.total_gsplats}, points={result.total_points})")

        # Measure memory
        mem = page.evaluate(JS_GET_MEMORY)
        if mem:
            result.js_heap_used_mb = mem["usedJSHeapSize"] / (1024 * 1024)
            result.js_heap_total_mb = mem["totalJSHeapSize"] / (1024 * 1024)
            print(f"    JS Heap: {result.js_heap_used_mb:.1f} MB used / {result.js_heap_total_mb:.1f} MB total")

        # Get WebGL renderer info
        webgl_info = page.evaluate(JS_GET_WEBGL_INFO)
        result.webgl_renderer = webgl_info.get("renderer", "unknown")
        print(f"    WebGL: {webgl_info.get('vendor', '?')} / {webgl_info.get('renderer', '?')}")

        # -- rAF-based FPS (vsync-capped, real user experience) --
        print(f"    Measuring idle FPS via rAF ({IDLE_MEASURE_DURATION_S}s)...")
        idle_frame_times = page.evaluate(
            JS_MEASURE_IDLE_FPS_RAF,
            IDLE_MEASURE_DURATION_S * 1000,
        )
        result.idle_frame_times_ms = idle_frame_times
        idle_stats = compute_fps_stats(idle_frame_times)
        result.idle_fps_mean = idle_stats["mean"]
        result.idle_fps_std = idle_stats["std"]
        result.idle_fps_min = idle_stats["min"]
        result.idle_fps_max = idle_stats["max"]
        result.idle_fps_median = idle_stats["median"]
        print(f"    Idle FPS (rAF): {idle_stats['mean']:.1f} +/- {idle_stats['std']:.1f}")

        print(f"    Measuring interaction FPS via rAF ({INTERACTION_MEASURE_DURATION_S}s)...")
        interaction_frame_times = page.evaluate(
            JS_MEASURE_INTERACTION_FPS_RAF,
            INTERACTION_MEASURE_DURATION_S * 1000,
        )
        result.interaction_frame_times_ms = interaction_frame_times
        interaction_stats = compute_fps_stats(interaction_frame_times)
        result.interaction_fps_mean = interaction_stats["mean"]
        result.interaction_fps_std = interaction_stats["std"]
        result.interaction_fps_min = interaction_stats["min"]
        result.interaction_fps_max = interaction_stats["max"]
        result.interaction_fps_median = interaction_stats["median"]
        print(f"    Interaction FPS (rAF): {interaction_stats['mean']:.1f} +/- {interaction_stats['std']:.1f}")

        # -- Uncapped render throughput (true GPU performance) --
        print(f"    Measuring idle render throughput ({IDLE_MEASURE_DURATION_S}s, uncapped)...")
        idle_tp = page.evaluate(
            JS_MEASURE_RENDER_THROUGHPUT,
            IDLE_MEASURE_DURATION_S * 1000,
        )
        result.idle_throughput_fps = idle_tp["throughputFps"]
        if idle_tp["frameTimes"]:
            result.idle_render_time_ms_mean = statistics.mean(idle_tp["frameTimes"])
            result.idle_render_time_ms_median = statistics.median(idle_tp["frameTimes"])
        print(f"    Idle throughput: {idle_tp['throughputFps']:.1f} FPS ({result.idle_render_time_ms_mean:.2f} ms/frame)")

        print(f"    Measuring interaction render throughput ({INTERACTION_MEASURE_DURATION_S}s, uncapped)...")
        interaction_tp = page.evaluate(
            JS_MEASURE_INTERACTION_THROUGHPUT,
            INTERACTION_MEASURE_DURATION_S * 1000,
        )
        result.interaction_throughput_fps = interaction_tp["throughputFps"]
        if interaction_tp["frameTimes"]:
            result.interaction_render_time_ms_mean = statistics.mean(interaction_tp["frameTimes"])
            result.interaction_render_time_ms_median = statistics.median(interaction_tp["frameTimes"])
        print(f"    Interaction throughput: {interaction_tp['throughputFps']:.1f} FPS ({result.interaction_render_time_ms_mean:.2f} ms/frame)")

        # Measure memory after all tests
        mem_after = page.evaluate(JS_GET_MEMORY)
        if mem_after:
            result.js_heap_used_mb = mem_after["usedJSHeapSize"] / (1024 * 1024)
            result.js_heap_total_mb = mem_after["totalJSHeapSize"] / (1024 * 1024)

    except Exception as e:
        result.error = str(e)
        print(f"    ERROR: {e}")

    return result


def run_benchmarks(
    datasets: list[tuple[str, str, str]],
    repeats: int = 3,
    headed: bool = False,
    viewport_width: int = 1280,
    viewport_height: int = 720,
) -> list[dict]:
    """Run benchmarks for all selected datasets."""
    from playwright.sync_api import sync_playwright

    all_results = []

    with sync_playwright() as p:
        # Launch browser with GPU acceleration and expose performance.memory
        # Note: headless Chromium uses software rendering (SwiftShader/SwANGLE).
        # For GPU-accelerated results, use --headed on a machine with a display.
        # On Linux with a GPU, headless=False + Xvfb can also work.
        launch_args = [
            "--enable-gpu-rasterization",
            "--enable-webgl",
            "--enable-webgl2",
            "--ignore-gpu-blocklist",
            "--enable-precise-memory-info",  # Expose performance.memory
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
        ]
        if not headed:
            # Use SwANGLE (software GL) in headless mode -- this is the default
            # and the most reliable path for headless WebGL rendering.
            pass
        browser = p.chromium.launch(
            headless=not headed,
            args=launch_args,
        )

        for ds_name, ds_label, ds_geom in datasets:
            zarr_path = DEMOS_DIR / ds_name
            if not zarr_path.exists():
                print(f"\n  SKIP {ds_label}: {zarr_path} does not exist")
                continue

            print(f"\n{'='*70}")
            print(f"  Dataset: {ds_label}")
            print(f"  Path: {zarr_path}")
            print(f"{'='*70}")

            # Find ports and start server
            data_port = find_free_port(18000, 18500)
            viewer_port = find_free_port(18500, 19000)

            print(f"  Starting server (data={data_port}, viewer={viewer_port})...")
            server_proc = start_luxar_server(zarr_path, data_port, viewer_port)

            try:
                # Wait for both data and viewer servers
                if not wait_for_server(data_port, timeout=30):
                    print(f"  ERROR: Data server did not start on port {data_port}")
                    kill_server(server_proc)
                    continue
                if not wait_for_server(viewer_port, timeout=30):
                    print(f"  ERROR: Viewer server did not start on port {viewer_port}")
                    kill_server(server_proc)
                    continue

                print("  Servers ready.")

                viewer_url = f"http://127.0.0.1:{viewer_port}"
                data_url = f"http://127.0.0.1:{data_port}"

                repeat_results = []
                for rep in range(repeats):
                    print(f"\n  --- Repeat {rep + 1}/{repeats} ---")

                    # Create a fresh browser context per repeat to avoid caching
                    context = browser.new_context(
                        viewport={"width": viewport_width, "height": viewport_height},
                    )
                    page = context.new_page()

                    result = run_single_benchmark(
                        page, ds_name, ds_label, ds_geom,
                        viewer_url, data_url,
                    )
                    repeat_results.append(result)

                    page.close()
                    context.close()

                # Aggregate results across repeats
                valid_results = [r for r in repeat_results if r.error is None]
                if valid_results:
                    aggregated = {
                        "dataset": ds_name,
                        "label": ds_label,
                        "geometry_type": ds_geom,
                        "total_elements": valid_results[0].total_elements,
                        "total_gsplats": valid_results[0].total_gsplats,
                        "total_points": valid_results[0].total_points,
                        "repeats": len(valid_results),
                        "viewport": f"{viewport_width}x{viewport_height}",
                        "load_time_ms": {
                            "mean": statistics.mean([r.load_time_ms for r in valid_results]),
                            "std": statistics.stdev([r.load_time_ms for r in valid_results]) if len(valid_results) > 1 else 0,
                            "min": min(r.load_time_ms for r in valid_results),
                            "max": max(r.load_time_ms for r in valid_results),
                            "values": [r.load_time_ms for r in valid_results],
                        },
                        "idle_fps": {
                            "mean": statistics.mean([r.idle_fps_mean for r in valid_results]),
                            "std": statistics.stdev([r.idle_fps_mean for r in valid_results]) if len(valid_results) > 1 else 0,
                            "per_repeat_mean": [r.idle_fps_mean for r in valid_results],
                            "per_repeat_median": [r.idle_fps_median for r in valid_results],
                        },
                        "interaction_fps": {
                            "mean": statistics.mean([r.interaction_fps_mean for r in valid_results]),
                            "std": statistics.stdev([r.interaction_fps_mean for r in valid_results]) if len(valid_results) > 1 else 0,
                            "per_repeat_mean": [r.interaction_fps_mean for r in valid_results],
                            "per_repeat_median": [r.interaction_fps_median for r in valid_results],
                        },
                        "idle_throughput_fps": {
                            "mean": statistics.mean([r.idle_throughput_fps for r in valid_results]),
                            "std": statistics.stdev([r.idle_throughput_fps for r in valid_results]) if len(valid_results) > 1 else 0,
                            "values": [r.idle_throughput_fps for r in valid_results],
                        },
                        "idle_render_time_ms": {
                            "mean": statistics.mean([r.idle_render_time_ms_mean for r in valid_results]),
                            "median": statistics.mean([r.idle_render_time_ms_median for r in valid_results]),
                        },
                        "interaction_throughput_fps": {
                            "mean": statistics.mean([r.interaction_throughput_fps for r in valid_results]),
                            "std": statistics.stdev([r.interaction_throughput_fps for r in valid_results]) if len(valid_results) > 1 else 0,
                            "values": [r.interaction_throughput_fps for r in valid_results],
                        },
                        "interaction_render_time_ms": {
                            "mean": statistics.mean([r.interaction_render_time_ms_mean for r in valid_results]),
                            "median": statistics.mean([r.interaction_render_time_ms_median for r in valid_results]),
                        },
                        "js_heap_used_mb": {
                            "mean": statistics.mean([r.js_heap_used_mb for r in valid_results]),
                            "values": [r.js_heap_used_mb for r in valid_results],
                        },
                        "js_heap_total_mb": {
                            "mean": statistics.mean([r.js_heap_total_mb for r in valid_results]),
                            "values": [r.js_heap_total_mb for r in valid_results],
                        },
                        "webgl_renderer": valid_results[0].webgl_renderer,
                    }
                    all_results.append(aggregated)

                    print(f"\n  SUMMARY for {ds_label}:")
                    print(f"    Elements: {aggregated['total_elements']}")
                    print(f"    Load time: {aggregated['load_time_ms']['mean']:.0f} +/- {aggregated['load_time_ms']['std']:.0f} ms")
                    print(f"    Idle FPS (rAF): {aggregated['idle_fps']['mean']:.1f} +/- {aggregated['idle_fps']['std']:.1f}")
                    print(f"    Idle throughput: {aggregated['idle_throughput_fps']['mean']:.1f} +/- {aggregated['idle_throughput_fps']['std']:.1f} FPS (uncapped)")
                    print(f"    Interaction FPS (rAF): {aggregated['interaction_fps']['mean']:.1f} +/- {aggregated['interaction_fps']['std']:.1f}")
                    print(f"    Interaction throughput: {aggregated['interaction_throughput_fps']['mean']:.1f} +/- {aggregated['interaction_throughput_fps']['std']:.1f} FPS (uncapped)")
                    print(f"    Render time (idle): {aggregated['idle_render_time_ms']['mean']:.2f} ms/frame")
                    print(f"    Render time (orbit): {aggregated['interaction_render_time_ms']['mean']:.2f} ms/frame")
                    print(f"    JS Heap: {aggregated['js_heap_used_mb']['mean']:.1f} MB")
                else:
                    print(f"  ALL REPEATS FAILED for {ds_label}")
                    errors = [r.error for r in repeat_results if r.error]
                    all_results.append({
                        "dataset": ds_name,
                        "label": ds_label,
                        "error": errors[0] if errors else "Unknown error",
                    })

            finally:
                kill_server(server_proc)

        browser.close()

    return all_results


def print_summary_table(results: list[dict]) -> None:
    """Print a nicely formatted summary table."""
    W = 150
    print("\n")
    print("=" * W)
    print("VIEWER PERFORMANCE BENCHMARK RESULTS")
    print("=" * W)
    header = (
        f"{'Dataset':<40s} {'Elements':>9s} {'Load(ms)':>9s} "
        f"{'rAF FPS':>8s} {'Idle TP':>9s} {'Orbit TP':>9s} "
        f"{'Idle ms':>8s} {'Orbit ms':>9s} {'Heap MB':>8s}"
    )
    print(header)
    print("-" * W)

    for r in results:
        if "error" in r and r.get("error"):
            print(f"{r['label']:<40s}  {'ERROR':>9s}  {r.get('error', '')}")
            continue

        elements = r.get("total_elements", 0)
        load_ms = r["load_time_ms"]["mean"]
        raf_fps = r["idle_fps"]["mean"]
        idle_tp = r["idle_throughput_fps"]["mean"]
        orbit_tp = r["interaction_throughput_fps"]["mean"]
        idle_ms = r["idle_render_time_ms"]["mean"]
        orbit_ms = r["interaction_render_time_ms"]["mean"]
        heap_mb = r["js_heap_used_mb"]["mean"]

        print(
            f"{r['label']:<40s} {elements:>9d} {load_ms:>9.0f} "
            f"{raf_fps:>8.1f} {idle_tp:>9.1f} {orbit_tp:>9.1f} "
            f"{idle_ms:>8.2f} {orbit_ms:>9.2f} {heap_mb:>8.1f}"
        )

    print("=" * W)
    print("  rAF FPS  = vsync-capped frame rate (real user experience)")
    print("  Idle TP  = uncapped idle render throughput (FPS)")
    print("  Orbit TP = uncapped camera-orbit render throughput (FPS)")
    print("  Idle ms  = mean render time per frame, idle (ms)")
    print("  Orbit ms = mean render time per frame, orbiting (ms)")
    print("  Heap MB  = JS heap memory used (MB)")


def main():
    parser = argparse.ArgumentParser(description="Luxar Viewer Performance Benchmark")
    parser.add_argument(
        "--datasets", "-d",
        default="all",
        help=f"Dataset preset or comma-separated zarr names. Presets: {', '.join(PRESETS.keys())}",
    )
    parser.add_argument("--repeats", "-r", type=int, default=3, help="Repetitions per dataset (default: 3)")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--width", type=int, default=1280, help="Viewport width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Viewport height (default: 720)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON filename (default: auto-generated)")
    args = parser.parse_args()

    # Select datasets
    if args.datasets in PRESETS:
        datasets = PRESETS[args.datasets]
    else:
        # Interpret as comma-separated zarr names
        names = [n.strip() for n in args.datasets.split(",")]
        datasets = [(n, n, "unknown") for n in names if (DEMOS_DIR / n).exists()]

    if not datasets:
        print("No valid datasets found. Check --datasets argument.")
        sys.exit(1)

    # Filter to only existing datasets
    existing = [(n, label, g) for n, label, g in datasets if (DEMOS_DIR / n).exists()]
    if len(existing) < len(datasets):
        missing = set(d[0] for d in datasets) - set(d[0] for d in existing)
        print(f"Warning: Missing datasets: {missing}")
    datasets = existing

    print("Luxar Viewer Performance Benchmark")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Repeats: {args.repeats}")
    print(f"  Viewport: {args.width}x{args.height}")
    print(f"  Headed: {args.headed}")
    print()

    results = run_benchmarks(
        datasets,
        repeats=args.repeats,
        headed=args.headed,
        viewport_width=args.width,
        viewport_height=args.height,
    )

    # Print summary
    print_summary_table(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = RESULTS_DIR / args.output
    else:
        out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    meta = {
        "timestamp": timestamp,
        "viewport": f"{args.width}x{args.height}",
        "repeats": args.repeats,
        "headed": args.headed,
        "luxar_root": str(LUXAR_ROOT),
    }
    output = {"metadata": meta, "results": results}

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
