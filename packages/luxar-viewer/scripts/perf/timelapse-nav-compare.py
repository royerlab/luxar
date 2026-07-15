#!/usr/bin/env python3
"""Compare before/after bench JSONs and print a markdown report."""
import json
import statistics
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/perfbench"


def load(name):
    try:
        return json.load(open(f"{BASE}/{name}.json"))
    except FileNotFoundError:
        return None


def wall_stats(run):
    walls = [tp["wallMs"] for tp in run["perTp"]]
    return statistics.median(walls), sorted(walls)[int(len(walls) * 0.9)]


def fmt_pool(delta):
    return ", ".join(f"{k}={v:+d}" for k, v in delta.items() if v)


def pool_delta(b, a):
    keys = ("allocations", "reuses", "evictions", "capacityGrowths", "deferredEvictions")
    return {k: a[k] - b[k] for k in keys if k in a and k in b}


def report(pair):
    before, after = load(f"before-{pair}"), load(f"after-{pair}")
    if not (before and after):
        print(f"### {pair}: MISSING RUNS ({'before ' if not before else ''}{'after' if not after else ''})")
        return
    print(f"### {pair}")
    print()
    print("| metric | before | after |")
    print("|---|---|---|")
    il_b, il_a = before["initialLoad"], after["initialLoad"]
    print(f"| initial load wall (ms) | {il_b['wallMs']} | {il_a['wallMs']} |")
    fb, fa = il_b["frameStats"], il_a["frameStats"]
    if fb and fa:
        print(f"| initial frames p95/p99/max (ms) | {fb['p95Ms']}/{fb['p99Ms']}/{fb['maxMs']} | {fa['p95Ms']}/{fa['p99Ms']}/{fa['maxMs']} |")
    for i in range(min(len(before["runs"]), len(after["runs"]))):
        rb, ra = before["runs"][i], after["runs"][i]
        mb, pb90 = wall_stats(rb)
        ma, pa90 = wall_stats(ra)
        tag = "cold" if i == 0 else "warm"
        print(f"| scrub{i+1} ({tag}) per-tp wall median/p90 (ms) | {mb:.0f}/{pb90} | {ma:.0f}/{pa90} |")
        fb, fa = rb["frameStats"], ra["frameStats"]
        if fb and fa:
            print(f"| scrub{i+1} frames p95/max (ms) | {fb['p95Ms']}/{fb['maxMs']} | {fa['p95Ms']}/{fa['maxMs']} |")
    # memory across scrubs (webgpu only — webgl info.memory lacks buffers)
    def mem_line(d):
        il = d["initialLoad"]["memory"] or {}
        last = d["runs"][-1]["memory"] or {}
        churn = (d.get("churn") or {}).get("memory") or {}
        return il.get("attributesSize"), last.get("attributesSize"), churn.get("attributesSize"), il.get("attributes"), last.get("attributes"), churn.get("attributes")
    b0, b1, b2, ba0, ba1, ba2 = mem_line(before)
    a0, a1, a2, aa0, aa1, aa2 = mem_line(after)
    if b0 is not None and a0 is not None:
        print(f"| attributesSize load→scrubs→churn (MB) | {b0/1e6:.1f} → {b1/1e6:.1f} → {b2/1e6 if b2 else 0:.1f} | {a0/1e6:.1f} → {a1/1e6:.1f} → {a2/1e6 if a2 else 0:.1f} |")
        print(f"| attribute count load→scrubs→churn | {ba0} → {ba1} → {ba2} | {aa0} → {aa1} → {aa2} |")
    cb, ca = before.get("churn"), after.get("churn")
    if cb and ca:
        print(f"| churn ({cb['switches']} switches) wall (ms) | {cb['wallMs']} | {ca['wallMs']} |")
        print(f"| churn pool delta | {fmt_pool(pool_delta(cb['poolBefore'], cb['poolAfter']))} | {fmt_pool(pool_delta(ca['poolBefore'], ca['poolAfter']))} |")
        fb, fa = cb["frameStats"], ca["frameStats"]
        if fb and fa:
            print(f"| churn frames p95/max (ms) | {fb['p95Ms']}/{fb['maxMs']} | {fa['p95Ms']}/{fa['maxMs']} |")
    print()


for pair in ("webgl", "webgpu"):
    report(pair)
