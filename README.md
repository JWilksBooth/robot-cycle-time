# robot-cycle-time

Minimum-cycle-time robot motion scheduling RL environment — rung 1 (motion) of a robotics vertical whose rung 2 is [bimanual-chore-scheduling](https://github.com/JWilksBooth/bimanual-chore-scheduling) (orchestration), built with the same validation discipline as its five power-systems siblings ([economic-dispatch](https://github.com/JWilksBooth/economic-dispatch), [dcopf-grid-verifiers](https://github.com/JWilksBooth/dcopf-grid-verifiers), [multiperiod-dispatch](https://github.com/JWilksBooth/multiperiod-dispatch), [n1-contingency-dispatch](https://github.com/JWilksBooth/n1-contingency-dispatch), [nodal-pricing-lmp](https://github.com/JWilksBooth/nodal-pricing-lmp)). **Cycle time is the metric industrial robotics is paid in.** An N-joint arm must move rest-to-rest from a start pose to a target pose, passing through a **synchronized waypoint** (every joint simultaneously), under per-joint velocity and acceleration limits — in the fewest timesteps.

The naive answer this environment defeats is the engineer's classic back-of-envelope: compute each joint's fastest individual profile and take the longest. With heterogeneous axes (slow strong proximal joints, fast light distal joints — real industrial texture) and a synchronized crossing, that schedule *always* fails the sync constraint (structurally: independent joints never coincide at the waypoint), and on a measured **65% of instances the shortcut's cycle-time estimate itself is wrong** — the true minimum is longer by a **median 16% synchronization premium**.

## Task

Given joint specs (max speed, max acceleration per joint), start/waypoint/target poses (radians), and dt = 0.1 s: output `{"trajectory": [[joint positions] per step]}` — the model chooses its own schedule length T. Rewards: format (0.10), kinematic feasibility (0.40 — positive-proof check of every velocity/acceleration transition including from-rest and to-rest edges, target pose, and the synchronized waypoint), and cycle time (0.50 — `exp(−5·(T−T*)/T*)`, hard-gated on feasibility).

## Ground truth validation

Two **independently assembled** formulations must certify the same minimum cycle T\*:

- **Primary:** bisection over horizon T; feasibility at each T is a position-space LP over joint trajectories (waypoint step free; feasibility is monotone in T since a rest step can always be appended). HiGHS dual simplex.
- **Cross-check:** a velocity-space formulation (variables are per-step joint velocities; poses enter as cumulative sums), linear-scanned from a pure-arithmetic lower bound, HiGHS interior point.

| Metric (200 instances, seeds 0–199) | Result |
|---|---|
| T\* disagreements between formulations | **0 / 200** |
| Cross-check non-convergence | 0 |
| Shortcut's time estimate underestimates T\* | **132 / 200 (66%)** |
| Median synchronization premium (that subset) | 16% |
| Draw rejection | **0.0%** (feasible by construction — no survivor bias) |

**Tolerance-certificate property:** grading tolerances (±0.01 rad poses, 0.5% kinematic grace, plus explicit decimal-quantization epsilons) are identical in the grader, the ground truth, and the cross-check — so **T\*−1 is proven infeasible under the grader's own envelope**, and tolerance-riding cannot buy a shorter certified cycle. The published reference trajectory is re-solved slightly interior to the envelope so decimal rounding cannot fail an honest optimum.

## Anti-reward-hacking (8 regression-tested attack gates)

The unsynchronized per-joint schedule (the shortcut itself) scores 0; the **teleport attack** (jump near the target and dwell) is caught by the velocity/acceleration gate; schedules shorter than T\* are rejected by certificate (fail-closed if one ever passes); `NaN`/`Infinity` literals, >4300-digit integers, and nested-bracket bombs score 0 without crashing; oversized schedules are rejected at parse; padded-but-feasible schedules earn honestly decayed partial credit.

## Usage

```bash
pip install -e .
N_INSTANCES=200 python tests/test_validation.py   # dual-formulation cross-check + attack gates
python calibration/measure.py --n 200             # difficulty distribution sweep

# give frontier models room to reason; -r 1 matches published baselines
vf-eval robot-cycle-time -p anthropic -m <model> -n 50 -r 1 --max-tokens 16000
```

```python
import verifiers as vf
env = vf.load_environment("robot-cycle-time", num_examples=300)
# disjoint train/eval: load_environment(1000) + load_environment(200, seed_offset=1000)
```

Instance generation: ~276 ms per instance including the T\* solve (~13k/hour — RL-scale datasets build in under an hour; dataset rows currently re-solve T\* once more, a documented cacheable optimization). Fully deterministic per seed.

## Instance generation

3–6 joints with position-tiered envelopes bracketing published collaborative-arm specs (proximal 1.4–2.2 rad/s and 3–7 rad/s², distal up to 6 rad/s and 25 rad/s²) — exact tiers are a stylization pending practitioner review, not vendor data. Travels of 0.6–2.6 rad per joint with waypoint detours of 0.2–1.2 rad off the direct path, signed per joint so the synchronized crossing is genuinely awkward for heterogeneous axes. See [CALIBRATION.md](CALIBRATION.md).

## Roadmap

- v0.2: jerk-limited S-curve profiles (what production controllers actually run — third difference, still LP-verifiable)
- v0.3: via-point sequences (multi-waypoint cycles); pick-and-place cycle benchmarks with practitioner-reviewed parameter envelopes
