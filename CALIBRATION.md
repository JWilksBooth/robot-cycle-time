# Calibration

Reproduce any number: `python calibration/measure.py --n 200`.

## Metrics

- **sync-premium rate** — fraction of instances where the per-joint
  time-optimal shortcut (max over joints of each joint's solo minimum)
  underestimates the true synchronized minimum T\*. The calibrated difficulty
  metric: these are the instances where the engineer's back-of-envelope
  *number* is wrong, not merely its schedule.
- **naive-schedule-infeasible rate** — measured at 100% across all configs and
  reported as **structural, not calibrated**: independently-planned joints
  essentially never cross the waypoint at the same step, so the shortcut's
  schedule always fails synchronization. It is the *time estimate* metric
  above that discriminates difficulty.
- **rejection rate** — 0.0% in all swept configs: instances are feasible by
  construction (any pose set within joint ranges admits some schedule under
  the generous T cap), so the accepted sample carries no survivor bias.

## Sweep (n = 120 per config)

| config | sync-prem | med premium | mean T\* | reject |
|---|---|---|---|---|
| **default (shipped)** | **65.0%** | **16.0%** | 17.1 | 0.0% |
| detour (0.1,0.7), distal v (2.4,4.5) | 66.7% | 19.4% | 15.0 | 0.0% |
| detour (0.1,0.7), distal v (3.0,6.0) | 57.5% | 20.0% | 14.4 | 0.0% |
| detour (0.2,1.2), distal v (2.4,4.5) | 76.7% | 18.2% | 17.8 | 0.0% |
| detour (0.4,1.6), distal v (2.4,4.5) | 69.2% | 17.6% | 20.8 | 0.0% |
| detour (0.4,1.6), distal v (3.0,6.0) | 57.5% | 14.3% | 20.1 | 0.0% |

Reading: difficulty rises when distal axes are *less* fast relative to
proximal (narrower heterogeneity forces genuine coordination rather than the
wrist simply waiting), and with moderate detours. The default sits mid-band at
65% / 16% with the largest premium variance; 76.7% is reachable
(detour (0.2,1.2) × distal (2.4,4.5)) and is a candidate "frontier" preset —
not shipped unmeasured, per factory rule.

## Shipped configuration (v0.1.0)

`DEFAULT_GEN_CONFIG` in `robot_cycle_time.py`. Validation on dataset seeds
0–199: 0/200 T\* disagreements between the position-space/bisection and
velocity-space/linear-scan formulations, 132/200 (66%) sync-premium, all 8
attack gates pass. Generation ~276 ms/instance including the T\* solve.

## Design iteration record

1. **Grader/ground-truth tolerance divergence (caught by gate 1 on first
   run):** LP solutions sit exactly on constraint boundaries, so decimal
   rounding pushed boundary values epsilon-outside the grader envelope. Fix:
   explicit quantization epsilons (0.0005 rad pose, 0.02 rad/s, 0.2 rad/s²)
   folded into the effective limits used identically by grader, ground truth,
   and cross-check — preserving the T\*−1 infeasibility certificate under the
   grader's exact tolerances — and the published reference trajectory
   re-solved slightly interior to the envelope so rounding cannot fail an
   honest optimum.
2. **Cross-check independence guard:** the cross-check's scan lower bound is
   pure arithmetic (path length / top speed), deliberately not the per-joint
   LP bound, which shares assembly code with the primary formulation.

## Stylizations stated, not hidden

1. **Joint envelopes are tiered stylizations** bracketing published
   collaborative-arm specs, not vendor data — flagged for review by a senior
   industrial-robotics practitioner; his corrections will land here as a
   documented revision.
2. **Kinematic (not dynamic) limits**: velocity/acceleration boxes per joint;
   no torque coupling, no payload dynamics. First-rung isolation of the
   coordination skill; jerk-limited S-curves are the designed v0.2.
3. **Single synchronized waypoint** — one crossing; via-point sequences are
   the v0.3 extension.
4. **dt = 0.1 s discretization** — coarse relative to real controllers (~1–10
   ms) so schedules stay readable/writable by a language model; the physics
   constraints are exact for the discretization stated in the prompt.
