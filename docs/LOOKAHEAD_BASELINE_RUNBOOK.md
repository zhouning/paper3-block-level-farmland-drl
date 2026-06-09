# Limited-Lookahead Baseline Runbook

This runbook is for the Paper3 robustness experiment that separates three
claims:

- block-level abstraction improves the search space;
- one-step reward-greedy planning is insufficient on Township B;
- the remaining Township-B advantage may or may not require RL training.

The experiment does not retrain DRL. It reuses the trained manuscript
environment and evaluates finite-depth deterministic search under the same
reward and within-block transition model as Reward-Greedy.

## Required Data

Set `PAPER3_DLTB_PATH` to the controlled-access parcel geometry with slope
attributes:

```bash
export PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg
```

On Windows PowerShell:

```powershell
$env:PAPER3_DLTB_PATH = "D:\path\to\DLTB_with_slope.gpkg"
```

## Primary Experiment

Run the exact depth-2 Township-B check first:

```bash
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 2 --beam-width 0
```

`--beam-width 0` means no immediate-reward pruning. For every episode step, the
script evaluates all valid first actions and all valid second actions reachable
from them. This is the cleanest test of whether ordinary two-step planning can
avoid Reward-Greedy's Township-B trap.

Expected outputs:

```text
results/derived_analyses/paper3_lookahead_d2_ball_results.json
results/tables/paper3_lookahead_d2_ball_table_fragment.tex
```

The script also prints an `Interpretation guidance` block after writing the
outputs. Keep that message with the JSON/TEX files because it states whether
the manuscript should frame the Township-B result as RL-specific learning or
as finite-depth multi-step planning.

Runtime note: exact depth-2 search is intentionally more expensive than
Reward-Greedy because it evaluates second-step consequences for every valid
first-step block. It should still be much cheaper than retraining the DRL
models. If it is too slow on a laptop, run the same command on Colab with the
controlled parcel geometry mounted.

## Sensitivity Experiment

If the depth-2 result is near the DRL/Reward-Greedy decision boundary, run a
depth-3 sensitivity check:

```bash
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 3 --beam-width 8 --suffix b_depth3_sensitivity
```

This is not exact exhaustive depth-3 search; it is a pruned beam search. Report
it as a sensitivity analysis, not as a proof of global planning optimality.

## Interpretation Rules

- If exact depth-2 lookahead reaches positive Township-B `baimu_fang` area close
  to DRL, revise the manuscript to say that the scheduler contribution is
  multi-step planning, not RL-specific learning.
- If exact depth-2 lookahead remains negative while DRL is positive, the current
  claim becomes stronger: one-step and two-step deterministic planning do not
  reproduce the learned scheduler's Township-B behavior.
- If depth-3 beam search succeeds but depth-2 fails, report the result as
  evidence that longer-horizon search can substitute for RL at higher planning
  cost.

## Minimal Results To Send Back

After running, provide:

```text
results/derived_analyses/paper3_lookahead_d2_ball_results.json
results/tables/paper3_lookahead_d2_ball_table_fragment.tex
```

Also include the terminal `Interpretation guidance` text printed at the end of
the run.

If the depth-3 sensitivity run was used, also provide:

```text
results/derived_analyses/paper3_lookahead_d3_b8_b_depth3_sensitivity_results.json
results/tables/paper3_lookahead_d3_b8_b_depth3_sensitivity_table_fragment.tex
```
