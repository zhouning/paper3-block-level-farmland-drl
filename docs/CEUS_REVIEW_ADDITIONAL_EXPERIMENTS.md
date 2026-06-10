# CEUS Review Additional Experiment Requirements

This note records the additional experiments implied by the CEUS-style review
of the current anonymous manuscript. It separates experiments that can be run
with existing scripts from those that require code changes before execution.

## Status After macOS Runs

The macOS follow-up experiments have been completed and committed to the
repository.

- Priority 1 completed: Township B depth-3 beam lookahead remains negative on
  `baimu_fang` area (`-103.8` ha), so it does not reproduce the DRL result.
- Priority 2 partially completed: the area-tolerance transition check was run
  as a stress test. Existing DRL block histories were replayed under the
  modified transition but were not retrained under that environment.
- Priority 3 partially completed: a Township B, seed-0 kappa diagnostic was
  generated, but it is not comparable to the main 200k-timestep, five-seed DRL
  experiments and is best treated as a supplemental diagnostic.

Relevant output files:

```text
results/derived_analyses/paper3_lookahead_d3_b8_b_depth3_sensitivity_results.json
results/tables/paper3_lookahead_d3_b8_b_depth3_sensitivity_table_fragment.tex
results/derived_analyses/paper3_area_tolerance_check_results.json
results/tables/paper3_area_tolerance_check_table.tex
results/derived_analyses/paper3_kappa_ablation_b_seed0_results.json
results/tables/paper3_kappa_ablation_b_seed0_table.tex
```

## Priority 1: Township B Depth-3 Lookahead Sensitivity

Purpose: test whether a longer deterministic planner can reproduce the
Township B behavior currently attributed to the trained scheduler. This is the
most direct follow-up to the exact depth-2 result already reported in the
manuscript.

Status: existing script; no DRL retraining required.

Required data: controlled-access `DLTB_with_slope.gpkg`.

Recommended machine: macOS is acceptable if runtime is tolerable; use Colab
Pro+ if the run is too slow.

Command on macOS or Linux:

```bash
export PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 3 --beam-width 8 --suffix b_depth3_sensitivity
```

Expected outputs to commit and report:

```text
results/derived_analyses/paper3_lookahead_d3_b8_b_depth3_sensitivity_results.json
results/tables/paper3_lookahead_d3_b8_b_depth3_sensitivity_table_fragment.tex
```

Interpretation:

- If depth-3 beam search stays negative on `baimu_fang` area, the learned
  scheduler claim becomes stronger under the tested deterministic baselines.
- If depth-3 beam search becomes positive and approaches DRL, revise the
  manuscript to frame the contribution as longer-horizon planning rather than
  RL-specific learning.
- If the result is near zero, report it as ambiguous sensitivity evidence.

## Priority 2: Area-Tolerance Transition Check

Purpose: address the strongest policy-consistency concern: the current engine
conserves parcel count, not farmland area.

Status: requires code changes before running.

Minimum implementation:

- Add an optional area-tolerance rule to `BlockLevelEnv`, for example
  `area_tolerance_pct in {0.5, 1.0}`.
- Replace the parcel-level slope check with an exact area-weighted
  `Delta mean slope` check.
- Rerun Township B first, because it is the critical case for the DRL-vs-greedy
  conclusion.

Recommended staged runs:

1. Deterministic baselines plus Reward-Greedy under the area-tolerant engine.
2. DRL deterministic evaluation using existing block histories, as a stress
   test only.
3. Full DRL retraining on Township B under the modified engine if the stress
   test changes the qualitative outcome.

Recommended machine: macOS can handle deterministic checks; use A100/H100 for
full DRL retraining.

## Priority 3: Parameter Sensitivity Beyond Township A

Purpose: show that the default `kappa=5` and connectivity weights
`gamma=1.0`, `delta=0.5` are not driving the main conclusion by themselves.

Status: partial script exists for `kappa`, but it is currently hard-coded for
Township A and seed 0. Extending it to Township B/C and multiple seeds requires
small code changes.

Recommended minimum:

- Run `kappa in {3, 5, 10}` on Township B, seed 0.
- If results are unstable, run seeds 0--4 on Township B.
- Treat Township C as optional unless the manuscript makes a stronger
  generality claim.

Recommended machine: A100/H100 for multi-seed retraining; macOS may be slow.

## Priority 4: Cross-District Validation

Purpose: address geographic generalizability. This is the most valuable
external-validity experiment, but it requires another controlled cadastral
dataset with comparable slope and land-use attributes.

Status: data-dependent; not required for the current text-only revision unless
the target submission needs stronger generalization evidence.

Minimum acceptable design:

- One additional township or county outside Bishan.
- Same preprocessing and block construction pipeline.
- Report whether block abstraction, Reward-Greedy competitiveness, and the DRL
  scheduler gap persist.
