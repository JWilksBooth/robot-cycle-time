"""robot-cycle-time: minimum-cycle-time robot motion scheduling RL environment.

First rung of a robotics vertical built with the validation discipline of the
power-systems environments (economic-dispatch -> dcopf-grid-verifiers ->
multiperiod-dispatch -> n1-contingency-dispatch). The skill axis is
SYNCHRONIZED TIME-OPTIMAL MOTION: an N-joint arm must move rest-to-rest from a
start pose to a target pose, passing through a synchronized intermediate
waypoint (all joints simultaneously), with per-joint velocity and acceleration
limits — in the fewest timesteps. Cycle time is the metric industrial robotics
is paid in.

The naive shortcut this environment defeats is the engineer's classic
back-of-envelope: compute each joint's individual time-optimal profile and
take the maximum. Heterogeneous axes (slow heavy proximal joints, fast light
distal joints) plus the synchronized waypoint make that schedule infeasible —
each joint wants to pass the waypoint at a different time — and make the true
minimum cycle longer than the per-joint bound (the "synchronization premium").

Validation discipline (factory standard):
- Ground truth: T* = minimum horizon with a feasible profile. Found by
  bisection over T; feasibility at each T is a position-space LP over joint
  trajectories, with the waypoint step free (any interior step; feasibility is
  monotone in T because a rest step can always be appended).
- Cross-check: an INDEPENDENTLY assembled velocity-space formulation (variables
  are per-step joint velocities; positions enter as cumulative sums), searched
  by linear scan from a per-joint lower bound, solved with a different HiGHS
  algorithm. Both must certify the same T* (achievable at T*, infeasible at
  T*-1) across the dataset.
- Grading and ground truth share the SAME tolerance envelope (pose +/-0.01 rad,
  0.5% kinematic grace), so tolerance-rent cannot buy a shorter certified
  cycle: T*-1 is proven infeasible under the grader's own tolerances.
- Parsers reject non-finite values, oversized integers, and pathological
  nesting; all attack classes from the sibling environments are regression-
  gated from day one.
"""

from __future__ import annotations

import json
import math
import random
import re

__version__ = "0.1.0"

DEFAULT_NUM_EXAMPLES = 300

DT = 0.1            # seconds per timestep
POS_TOL = 0.01      # rad tolerance on waypoint/target poses (grader AND ground truth)
LIMIT_GRACE = 1.005 # 0.5% kinematic grace (grader AND ground truth)
MAX_T = 60          # hard cap on schedule length accepted from a model

# Quantization epsilons: answers are reported to ~3 decimals (0.0005 rad
# rounding), which propagates to 0.005 rad/s velocity and 0.05 rad/s^2 accel
# noise at dt=0.1. These margins are folded into the EFFECTIVE limits used
# identically by the ground truth, the cross-check, and the grader — so the
# T*-1 infeasibility certificate holds under the grader's exact tolerances
# and rounding can neither fail an honest optimum nor buy a shorter cycle.
EPS_P = 5e-4        # rad
EPS_V = 0.02        # rad/s
EPS_A = 0.2         # rad/s^2
_POS_EFF = POS_TOL + EPS_P


def _v_eff(joint: dict) -> float:
    return joint["v_max"] * LIMIT_GRACE + EPS_V


def _a_eff(joint: dict) -> float:
    return joint["a_max"] * LIMIT_GRACE + EPS_A

# --- Calibration knobs (measured; see calibration/measure.py) ----------------
# Joint envelopes are position-tiered with real industrial texture: proximal
# joints are slow and strong (they carry the arm), distal joints are fast and
# light. Ranges bracket published specs of common 6-axis industrial arms
# (UR/Franka-class collaborative envelopes); exact tiers are stylization to be
# reviewed by a practitioner, not vendor data.
DEFAULT_GEN_CONFIG: dict = {
    "n_joints_range": (3, 6),
    "v_proximal": (1.4, 2.2),    # rad/s, slowest (base/shoulder) tier
    "v_distal": (3.0, 6.0),      # rad/s, fastest (wrist) tier
    "a_proximal": (3.0, 7.0),    # rad/s^2
    "a_distal": (10.0, 25.0),
    "travel_range": (0.6, 2.6),  # |target - start| per joint, rad
    "waypoint_detour": (0.2, 1.2),  # waypoint offset from the straight path, rad
    "max_tstar": 45,             # reject draws needing more than this many steps
}


def _tiered(rng: random.Random, j: int, J: int, lo_band, hi_band) -> float:
    """Interpolate a joint's limit between proximal and distal bands by position."""
    frac = j / max(1, J - 1)
    lo = lo_band[0] + (hi_band[0] - lo_band[0]) * frac
    hi = lo_band[1] + (hi_band[1] - lo_band[1]) * frac
    return round(rng.uniform(lo, hi), 3)


def build_instance(rng: random.Random, **cfg) -> dict:
    p = {**DEFAULT_GEN_CONFIG, **cfg}
    J = rng.randint(*p["n_joints_range"])
    joints = []
    for j in range(J):
        joints.append({
            "name": f"J{j+1}",
            "v_max": _tiered(rng, j, J, p["v_proximal"], p["v_distal"]),
            "a_max": _tiered(rng, j, J, p["a_proximal"], p["a_distal"]),
        })
    start, waypoint, target = [], [], []
    for j in range(J):
        s = round(rng.uniform(-math.pi, math.pi), 3)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        travel = rng.uniform(*p["travel_range"]) * sign
        g = round(s + travel, 3)
        # waypoint near mid-path with a detour, sign chosen per joint so the
        # synchronized crossing is genuinely awkward for heterogeneous axes
        detour = rng.uniform(*p["waypoint_detour"]) * (1.0 if rng.random() < 0.5 else -1.0)
        w = round(s + 0.5 * travel + detour, 3)
        start.append(s); waypoint.append(w); target.append(g)
    return {"joints": joints, "start": start, "waypoint": waypoint,
            "target": target, "dt": DT}


def generate_instance(seed: int, max_attempts: int = 200, **cfg) -> dict:
    """Deterministic per seed; retry stream offset by (seed+1)."""
    p = {**DEFAULT_GEN_CONFIG, **cfg}
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        inst = build_instance(rng, **cfg)
        tstar = solve_min_cycle(inst)
        if tstar is not None and tstar["T_star"] <= p["max_tstar"]:
            return inst
        rng = random.Random((seed + 1) * 1_000_003 + attempt)
    raise RuntimeError(f"no feasible instance for seed {seed}")


# ---------------- ground truth: bisection + position-space LP ----------------

def _feasible_position_lp(inst: dict, T: int, t_w: int, margin: float = 1.0):
    """Feasibility LP in position space for horizon T with waypoint at step t_w.
    Variables theta[j,t], t=1..T. Rest-to-rest. margin scales the quantization
    epsilons: margin=1.0 is exactly the grader's envelope (used for T* and the
    T*-1 infeasibility certificate); margin<1.0 solves slightly INTERIOR so the
    returned reference trajectory survives decimal rounding when re-graded."""
    from scipy.optimize import linprog
    import numpy as np

    J = len(inst["joints"])
    dt = inst["dt"]
    n = J * T
    idx = lambda j, t: j * T + (t - 1)
    pos_eff = POS_TOL + margin * EPS_P

    A, b = [], []
    for j, jt in enumerate(inst["joints"]):
        vmax = (jt["v_max"] * LIMIT_GRACE + margin * EPS_V) * dt
        amax = (jt["a_max"] * LIMIT_GRACE + margin * EPS_A) * dt * dt
        s = inst["start"][j]
        for t in range(1, T + 1):
            prev = None if t == 1 else idx(j, t - 1)
            # velocity |theta_t - theta_{t-1}| <= vmax
            row = np.zeros(n); row[idx(j, t)] = 1.0
            if prev is not None: row[prev] = -1.0
            A.append(row);  b.append(vmax + (s if prev is None else 0.0))
            A.append(-row); b.append(vmax - (s if prev is None else 0.0))
        for t in range(1, T):
            # accel |theta_{t+1} - 2 theta_t + theta_{t-1}| <= amax
            row = np.zeros(n)
            row[idx(j, t + 1)] = 1.0; row[idx(j, t)] = -2.0
            base = 0.0
            if t == 1: base = s
            else: row[idx(j, t - 1)] = 1.0
            A.append(row);  b.append(amax - base)
            A.append(-row); b.append(amax + base)
        # initial accel from rest: |theta_1 - s| <= amax
        row = np.zeros(n); row[idx(j, 1)] = 1.0
        A.append(row);  b.append(amax + s)
        A.append(-row); b.append(amax - s)
        # final decel to rest: |theta_T - theta_{T-1}| <= amax
        row = np.zeros(n); row[idx(j, T)] = 1.0
        if T >= 2: row[idx(j, T - 1)] = -1.0
        A.append(row);  b.append(amax + (s if T < 2 else 0.0))
        A.append(-row); b.append(amax - (s if T < 2 else 0.0))

    bounds = [(None, None)] * n
    for j in range(J):
        w, g = inst["waypoint"][j], inst["target"][j]
        bounds[idx(j, t_w)] = (w - pos_eff, w + pos_eff)
        bounds[idx(j, T)] = (g - pos_eff, g + pos_eff)

    res = linprog(np.zeros(n), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=bounds, method="highs-ds")
    if not res.success:
        return None
    return [[round(float(res.x[idx(j, t)]), 5) for j in range(J)]
            for t in range(1, T + 1)]


def _feasible_T(inst: dict, T: int):
    """Any interior waypoint step make horizon T feasible? Returns (t_w, traj) or None."""
    if T < 4:
        return None
    for t_w in range(2, T - 1):
        traj = _feasible_position_lp(inst, T, t_w)
        if traj is not None:
            return (t_w, traj)
    return None


def solve_min_cycle(inst: dict, t_cap: int = MAX_T) -> dict | None:
    """T* by bisection (feasibility is monotone in T: a rest step can always be
    appended to a feasible schedule)."""
    lo, hi = 4, t_cap
    if _feasible_T(inst, hi) is None:
        return None
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        got = _feasible_T(inst, mid)
        if got is not None:
            best = (mid, got)
            hi = mid - 1
        else:
            lo = mid + 1
    T_star, (t_w, traj) = best
    # reference trajectory: re-solve slightly interior so decimal rounding
    # cannot push it outside the grader's envelope (fall back outward if the
    # instance is only feasible in the outer shell)
    for m in (0.25, 0.5, 0.75):
        interior = _feasible_position_lp(inst, T_star, t_w, margin=m)
        if interior is not None:
            traj = interior
            break
    return {"T_star": T_star, "waypoint_step": t_w, "trajectory": traj,
            "cycle_s": round(T_star * inst["dt"], 2)}


# -------- independent cross-check: velocity-space LP, linear scan ------------

def _feasible_velocity_lp(inst: dict, T: int, t_w: int) -> bool:
    """Independently assembled feasibility test: variables are per-step joint
    velocities v[j,t]; positions are cumulative sums. Different variable space,
    different assembly, different HiGHS algorithm."""
    from scipy.optimize import linprog
    import numpy as np

    J, dt = len(inst["joints"]), inst["dt"]
    n = J * T
    idx = lambda j, t: j * T + (t - 1)
    A, b = [], []
    bounds = []
    for j, jt in enumerate(inst["joints"]):
        vm = _v_eff(jt)
        for t in range(1, T + 1):
            bounds.append((-vm, vm))
        am = _a_eff(jt) * dt
        for t in range(1, T):
            row = np.zeros(n); row[idx(j, t + 1)] = 1.0; row[idx(j, t)] = -1.0
            A.append(row); b.append(am)
            A.append(-row); b.append(am)
        for edge_t in (1, T):  # from rest / to rest
            row = np.zeros(n); row[idx(j, edge_t)] = 1.0
            A.append(row); b.append(am)
            A.append(-row); b.append(am)
        # waypoint and target as cumulative-sum bands
        for (upto, pose) in ((t_w, inst["waypoint"][j]), (T, inst["target"][j])):
            row = np.zeros(n)
            for t in range(1, upto + 1):
                row[idx(j, t)] = dt
            delta = pose - inst["start"][j]
            A.append(row); b.append(delta + _POS_EFF)
            A.append(-row); b.append(-(delta - _POS_EFF))
    res = linprog(np.zeros(n), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=bounds, method="highs-ipm")
    return bool(res.success)


def solve_min_cycle_crosscheck(inst: dict, t_cap: int = MAX_T) -> int | None:
    """Linear scan upward from a provable arithmetic lower bound (per-joint
    path length divided by top speed — pure arithmetic, no shared solver code,
    so the cross-check stays independent of the position-space assembly)."""
    lb = 4
    for j, jt in enumerate(inst["joints"]):
        path = (abs(inst["waypoint"][j] - inst["start"][j])
                + abs(inst["target"][j] - inst["waypoint"][j]))
        lb = max(lb, math.ceil(path / (_v_eff(jt) * inst["dt"])))
    for T in range(lb, t_cap + 1):
        for t_w in range(2, T - 1):
            if _feasible_velocity_lp(inst, T, t_w):
                return T
    return None


# ---------------- the naive shortcut (difficulty metric + gate attack) --------

def _solo_joint_T(inst: dict, j: int, t_cap: int = MAX_T):
    """Minimum horizon for joint j ALONE (waypoint step free) — the per-joint
    time-optimal bound an engineer computes on the back of an envelope."""
    solo = {"joints": [inst["joints"][j]], "start": [inst["start"][j]],
            "waypoint": [inst["waypoint"][j]], "target": [inst["target"][j]],
            "dt": inst["dt"]}
    for T in range(4, t_cap + 1):
        got = _feasible_T(solo, T)
        if got is not None:
            return T, got[1]
    return None, None


def naive_per_joint_T(inst: dict) -> int:
    """max over joints of the solo minimum — the classic shortcut's cycle estimate."""
    return max(_solo_joint_T(inst, j)[0] or MAX_T for j in range(len(inst["joints"])))


def naive_schedule(inst: dict) -> list[list[float]]:
    """The shortcut's actual schedule: each joint runs its own solo-optimal
    profile independently (padded with rest at its goal), waypoint crossings
    unsynchronized. This is what per-joint reasoning produces."""
    J = len(inst["joints"])
    T_n = naive_per_joint_T(inst)
    cols = []
    for j in range(J):
        _, traj = _solo_joint_T(inst, j)
        col = [row[0] for row in traj] if traj else [inst["target"][j]]
        col += [inst["target"][j]] * (T_n - len(col))
        cols.append(col[:T_n])
    return [[cols[j][t] for j in range(J)] for t in range(T_n)]


# ---------------- parsing, physics gate, rewards ------------------------------

def _is_finite_number(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def parse_trajectory(completion: str, inst: dict) -> list[list[float]] | None:
    J = len(inst["joints"])
    for raw in reversed(re.findall(r'\{[^{}]*"trajectory"[^{}]*\}', completion, re.DOTALL)):
        try:
            obj = json.loads(raw)
        # ValueError covers JSONDecodeError AND CPython's >4300-digit int limit;
        # RecursionError covers nested-bracket bombs. Skip, never crash.
        except (ValueError, RecursionError):
            continue
        rows = obj.get("trajectory")
        if (isinstance(rows, list) and 1 <= len(rows) <= MAX_T
                and all(isinstance(r, list) and len(r) == J
                        and all(_is_finite_number(v) for v in r) for r in rows)):
            return [[float(v) for v in r] for r in rows]
    return None


def check_feasibility(inst: dict, X: list[list[float]]) -> tuple[bool, list[str]]:
    """Positive-proof kinematic gate, tolerance-identical to the ground truth:
    velocity and acceleration limits every step (incl. from-rest and to-rest
    edges), synchronized waypoint at SOME step, target at the final step."""
    J, dt = len(inst["joints"]), inst["dt"]
    T = len(X)
    v: list[str] = []
    prev = list(inst["start"])
    vel_prev = [0.0] * J
    for t, row in enumerate(X, start=1):
        for j, jt in enumerate(inst["joints"]):
            vel = (row[j] - prev[j]) / dt
            if not abs(vel) <= _v_eff(jt):
                v.append(f"step {t} {jt['name']}: velocity {vel:.2f} rad/s over limit")
            acc = (vel - vel_prev[j]) / dt
            if not abs(acc) <= _a_eff(jt):
                v.append(f"step {t} {jt['name']}: accel {acc:.1f} rad/s^2 over limit")
            vel_prev[j] = vel
        prev = row
    # to-rest edge: decelerating from the final step's velocity to zero
    for j, jt in enumerate(inst["joints"]):
        if not abs(vel_prev[j]) <= _a_eff(jt) * dt:
            v.append(f"final step {jt['name']}: arrives at {vel_prev[j]:.2f} rad/s, cannot stop")
    # target pose
    for j in range(J):
        if not abs(X[-1][j] - inst["target"][j]) <= _POS_EFF:
            v.append(f"{inst['joints'][j]['name']}: final pose misses target")
    # synchronized waypoint at some step
    hit = any(all(abs(X[t][j] - inst["waypoint"][j]) <= _POS_EFF for j in range(J))
              for t in range(T))
    if not hit:
        v.append("waypoint never crossed by all joints simultaneously")
    return len(v) == 0, v


def reward_format(completion: str, inst: dict) -> float:
    return 1.0 if parse_trajectory(completion, inst) is not None else 0.0


def reward_feasibility(completion: str, inst: dict) -> float:
    X = parse_trajectory(completion, inst)
    if X is None:
        return 0.0
    ok, _ = check_feasibility(inst, X)
    return 1.0 if ok else 0.0


def reward_cycle_time(completion: str, inst: dict,
                      t_star: int | None = None) -> float:
    """exp(-5 * (T_yours - T*)/T*), hard-gated on feasibility. A feasible
    schedule shorter than T* is impossible by certificate (T*-1 proven
    infeasible under the grader's own tolerances); if one ever appears it
    indicates a checker/ground-truth divergence and scores 0 (fail closed)."""
    X = parse_trajectory(completion, inst)
    if X is None:
        return 0.0
    ok, _ = check_feasibility(inst, X)
    if not ok:
        return 0.0
    if t_star is None:
        sol = solve_min_cycle(inst)
        if sol is None:
            return 0.0
        t_star = sol["T_star"]
    T_ans = len(X)
    if T_ans < t_star:
        return 0.0  # violates the infeasibility certificate -> fail closed
    gap = (T_ans - t_star) / t_star
    return math.exp(-5.0 * gap)


REWARD_WEIGHTS = {"reward_format": 0.10, "reward_feasibility": 0.40,
                  "reward_cycle_time": 0.50}


# ---------------- prompt + verifiers entry point ------------------------------

def instance_to_prompt(inst: dict) -> str:
    J = len(inst["joints"])
    joints_txt = "\n".join(
        f"  {jt['name']}: max speed {jt['v_max']} rad/s, max acceleration {jt['a_max']} rad/s^2"
        for jt in inst["joints"])
    fmt3 = lambda xs: "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"
    example_row = ", ".join(f"<{jt['name']} rad>" for jt in inst["joints"])
    return f"""You are programming a {J}-axis robot arm for a production cell. Minimize CYCLE TIME.

Joints (proximal axes are slow and strong, distal axes fast and light):
{joints_txt}

Motion (all angles in radians):
  Start pose (at rest):   {fmt3(inst['start'])}
  Waypoint pose:          {fmt3(inst['waypoint'])}
  Target pose (at rest):  {fmt3(inst['target'])}

Discrete time, dt = {inst['dt']} s per step. Output the joint positions at each
step t = 1, 2, ..., T (the start pose is step 0 and is given). Rules:
1. Velocity: each joint moves at most (max speed x dt) per step.
2. Acceleration: each joint's per-step velocity changes by at most
   (max acceleration x dt) between consecutive steps — including speeding up
   from rest at the start and braking to rest at the end (the arm must arrive
   at the target with zero velocity).
3. WAYPOINT SYNCHRONIZATION: there must be at least one step at which EVERY
   joint is simultaneously at its waypoint pose (within 0.01 rad). Joints
   passing the waypoint at different times does not count.
4. The final step must be the target pose (within 0.01 rad per joint).

Your cycle time is T x dt — fewer steps is better, but an infeasible schedule
scores zero. Warning: computing each joint's fastest individual profile and
taking the longest is NOT generally optimal here — the synchronized waypoint
forces joints to coordinate.

Report positions to at least 3 decimal places. Constraints are checked with a
small tolerance (0.01 rad on poses, 0.5% on limits) that the scoring already
accounts for — target exact values.

Respond with your final answer as JSON on the last line, exactly:
{{"trajectory": [[{example_row}], ... one list per step ...]}}"""


def build_dataset(num_examples: int = DEFAULT_NUM_EXAMPLES, seed_offset: int = 0):
    rows = []
    for i in range(num_examples):
        inst = generate_instance(seed_offset + i)
        sol = solve_min_cycle(inst)
        rows.append({
            "question": instance_to_prompt(inst),
            "answer": str(sol["T_star"]),
            "info": {"instance": inst, "t_star": sol["T_star"],
                     "optimal_trajectory": sol["trajectory"]},
        })
    return rows


def load_environment(num_examples: int = DEFAULT_NUM_EXAMPLES,
                     seed_offset: int = 0, **kwargs):
    """seed_offset enables disjoint train/eval datasets."""
    import verifiers as vf
    from datasets import Dataset

    dataset = Dataset.from_list(build_dataset(num_examples, seed_offset=seed_offset))

    def _text(completion):
        if isinstance(completion, str):
            return completion
        parts = []
        for m in completion:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for pc in content:
                    t = pc.get("text") if isinstance(pc, dict) else getattr(pc, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
        return " ".join(parts)

    def fmt(completion, info, **kw):
        return reward_format(_text(completion), info["instance"])

    def feas(completion, info, **kw):
        return reward_feasibility(_text(completion), info["instance"])

    def cyc(completion, info, **kw):
        return reward_cycle_time(_text(completion), info["instance"],
                                 t_star=info["t_star"])

    rubric = vf.Rubric(funcs=[fmt, feas, cyc],
                       weights=list(REWARD_WEIGHTS.values()))
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
