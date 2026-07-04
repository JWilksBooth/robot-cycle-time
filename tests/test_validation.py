"""Validation harness: dual-formulation T* cross-check + red-team gates."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_cycle_time import (
    generate_instance, solve_min_cycle, solve_min_cycle_crosscheck,
    naive_per_joint_T, naive_schedule, check_feasibility,
    reward_format, reward_feasibility, reward_cycle_time, MAX_T,
)

N = int(os.environ.get("N_INSTANCES", "200"))


def run_crosscheck():
    mism, xc_fail, sync_binding = 0, 0, 0
    for i in range(N):
        inst = generate_instance(i)
        sol = solve_min_cycle(inst)
        assert sol is not None, f"seed {i}: generator produced infeasible instance"

        if naive_per_joint_T(inst) < sol["T_star"]:
            sync_binding += 1  # the per-joint shortcut underestimates the true cycle

        xc = solve_min_cycle_crosscheck(inst)
        if xc is None:
            xc_fail += 1
            continue
        if xc != sol["T_star"]:
            mism += 1
            print(f"  MISMATCH seed {i}: bisection/position T*={sol['T_star']} "
                  f"vs scan/velocity T*={xc}")

    print(f"cross-check: {N} instances | T* mismatches={mism} | "
          f"crosscheck non-converged={xc_fail}")
    print(f"difficulty:  {sync_binding}/{N} instances have a synchronization premium "
          f"({100*sync_binding/N:.0f}%) — per-joint time-optimal reasoning underestimates T*")
    assert mism == 0, "formulation disagreement — do not ship"
    assert xc_fail <= 0.02 * N, "crosscheck failing — vacuous pass risk"
    assert sync_binding >= 0.2 * N, "too few sync-premium instances — increase heterogeneity/detour"
    return sync_binding


def run_gate_tests():
    # find an instance where the naive schedule is infeasible (the signature case)
    inst = sol = None
    for s in range(500):
        cand = generate_instance(seed=s)
        nsch = naive_schedule(cand)
        ok, _ = check_feasibility(cand, nsch)
        if not ok:
            inst, sol = cand, solve_min_cycle(cand)
            break
    assert inst is not None
    J = len(inst["joints"])

    def answer(X):
        return json.dumps({"trajectory": [[round(v, 4) for v in row] for row in X]})

    # 1. ground-truth optimum scores full credit
    a = answer(sol["trajectory"])
    assert reward_format(a, inst) == 1.0
    assert reward_feasibility(a, inst) == 1.0
    assert reward_cycle_time(a, inst, sol["T_star"]) > 0.99
    print("gate 1 (T* optimum): PASS")

    # 2. THE signature attack: the per-joint shortcut's schedule — each joint
    # time-optimal alone, waypoint crossings unsynchronized. Must score 0.
    a = answer(naive_schedule(inst))
    assert reward_feasibility(a, inst) == 0.0
    assert reward_cycle_time(a, inst, sol["T_star"]) == 0.0
    print("gate 2 (unsynchronized per-joint schedule): PASS — feasibility 0.0")

    # 3. garbage
    assert reward_format("forty-two", inst) == 0.0
    assert reward_cycle_time("forty-two", inst, sol["T_star"]) == 0.0
    print("gate 3 (garbage): PASS")

    # 4. feasible but slow: optimum padded with rest steps — partial credit
    padded = sol["trajectory"] + [sol["trajectory"][-1][:] for _ in range(5)]
    a = answer(padded)
    assert reward_feasibility(a, inst) == 1.0
    r = reward_cycle_time(a, inst, sol["T_star"])
    assert 0.0 < r < 1.0, f"expected partial credit, got {r}"
    print(f"gate 4 (feasible padded +5 steps): PASS — partial credit {r:.3f}")

    # 5. NaN/Infinity/huge-int/bracket-bomb — all rewards 0, no crash
    row_of = lambda bad: "[" + ", ".join([bad] * J) + "]"
    for bad in ("NaN", "Infinity", "-Infinity", "9" * 5000):
        a = '{"trajectory": [' + ", ".join([row_of(bad)] * 6) + "]}"
        assert reward_format(a, inst) == 0.0
        assert reward_feasibility(a, inst) == 0.0
        assert reward_cycle_time(a, inst, sol["T_star"]) == 0.0
    bomb = '{"trajectory": ' + "[" * 2000 + "]" * 2000 + "}"
    assert reward_format(bomb, inst) == 0.0 and reward_cycle_time(bomb, inst, sol["T_star"]) == 0.0
    print("gate 5 (NaN/Infinity/huge-int/bracket-bomb): PASS — all rewards 0.0, no crash")

    # 6. teleport attack: skip the physics, jump near the target early, dwell.
    # Velocity/accel gates must catch the jump.
    tele = [list(inst["target"]) for _ in range(max(4, sol["T_star"] - 2))]
    a = answer(tele)
    assert reward_feasibility(a, inst) == 0.0, "teleport passed the kinematic gate!"
    print("gate 6 (teleport): PASS — feasibility 0.0")

    # 7. certificate integrity: no schedule shorter than T* may pass the gate.
    # Construct the strongest candidate we can (ground-truth LP at T*-1 is
    # infeasible by certificate; simulate an attacker truncating the optimum).
    trunc = sol["trajectory"][:-1]
    if len(trunc) >= 1:
        a = answer(trunc)
        assert not (reward_feasibility(a, inst) == 1.0), (
            "truncated schedule passed — certificate broken")
    print("gate 7 (below-T* certificate): PASS — truncation rejected")

    # 8. arity/cap: oversized schedule rejected at parse
    huge = [[0.0] * J for _ in range(MAX_T + 1)]
    assert reward_format(answer(huge), inst) == 0.0
    print("gate 8 (schedule-length cap): PASS")


if __name__ == "__main__":
    run_gate_tests()
    run_crosscheck()
    print("ALL VALIDATION PASSED")
