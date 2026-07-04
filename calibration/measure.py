"""Calibration harness: measure the synchronization-difficulty distribution.

Headline metric: **sync-premium rate** — the fraction of instances where the
per-joint time-optimal shortcut (each joint's fastest individual profile, take
the max) underestimates the true minimum cycle T*. Those are the instances
where the engineer's back-of-envelope answer is wrong. Secondary: the
**sync premium** (T* - T_naive)/T_naive on that subset — how much coordination
costs; and the raw-draw rejection rate (sample-bias indicator).

    python calibration/measure.py --n 200
"""

import sys, os, json, argparse, random
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_cycle_time import (
    generate_instance, build_instance, solve_min_cycle, naive_per_joint_T,
    naive_schedule, check_feasibility, DEFAULT_GEN_CONFIG,
)


def measure_instance(inst):
    sol = solve_min_cycle(inst)
    if sol is None:
        return None
    t_naive = naive_per_joint_T(inst)
    ok, _ = check_feasibility(inst, naive_schedule(inst))
    prem = (sol["T_star"] - t_naive) / t_naive
    return {"premium_pos": prem > 0, "premium": max(0.0, prem),
            "naive_infeasible": not ok, "T_star": sol["T_star"]}


def rejection_rate(cfg, k=200, seed_base=900_000):
    cap = {**DEFAULT_GEN_CONFIG, **cfg}["max_tstar"]
    fails = 0
    for i in range(k):
        rng = random.Random(seed_base + i)
        inst = build_instance(rng, **cfg)
        sol = solve_min_cycle(inst)
        if sol is None or sol["T_star"] > cap:
            fails += 1
    return fails / k


def measure_config(cfg, n):
    rows = [m for m in (measure_instance(generate_instance(s, **cfg))
                        for s in range(n)) if m]
    pos = [r for r in rows if r["premium_pos"]]
    return {
        "n": len(rows),
        "sync_premium_rate": len(pos) / len(rows),
        "naive_schedule_infeasible_rate": sum(r["naive_infeasible"] for r in rows) / len(rows),
        "median_premium": median([r["premium"] for r in pos]) if pos else 0.0,
        "mean_T_star": mean(r["T_star"] for r in rows),
        "rejection_rate": rejection_rate(cfg),
    }


def fmt(label, r):
    return (f"{label:<44} | {r['sync_premium_rate']*100:6.1f}% "
            f"| {r['naive_schedule_infeasible_rate']*100:6.1f}% "
            f"| {r['median_premium']*100:6.1f}% "
            f"| {r['mean_T_star']:4.1f} | {r['rejection_rate']*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("N_INSTANCES", "200")))
    args = ap.parse_args()

    header = f"{'config':<44} | sync-prem | naive-inf | med prem | T*   | reject"
    print("\n  sync-prem = % instances where the per-joint shortcut underestimates T*")
    print("  naive-inf = % where the shortcut's actual schedule fails the grader")
    print(f"\n{header}\n{'-' * len(header)}")

    results = {}
    base = measure_config({}, args.n)
    results["default"] = {"config": {}, "metrics": base}
    print(fmt("default (shipped)", base))

    for detour in ((0.1, 0.7), (0.2, 1.2), (0.4, 1.6)):
        for vd in ((2.4, 4.5), (3.0, 6.0)):
            cfg = {"waypoint_detour": detour, "v_distal": vd}
            r = measure_config(cfg, args.n)
            key = f"detour={detour}_vdistal={vd}"
            results[key] = {"config": cfg, "metrics": r}
            print(fmt(key, r))

    out = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(out, "w") as f:
        json.dump({"n_per_config": args.n, "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
