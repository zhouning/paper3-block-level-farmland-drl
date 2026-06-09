# Paper3 CEUS Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the Paper3 CEUS manuscript by tightening claims, adding a reproducible limited-lookahead baseline scaffold, and verifying the submission package.

**Architecture:** Keep the manuscript revision and new experiment pathway separate. The manuscript edits clarify scope and deployment boundaries without inventing results; the new analysis script produces optional results that can be inserted only after running on controlled-access parcel data.

**Tech Stack:** LaTeX manuscript, Python standard library tests, NumPy, existing `BlockLevelEnv` and Paper3 result path helpers.

---

### Task 1: Add Tested Limited-Lookahead Baseline Utilities

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_paper3_lookahead_baseline.py`
- Create: `scripts/analysis/paper3_lookahead_baseline.py`

- [ ] **Step 1: Write failing behavior tests**

Create a fake tree environment with deterministic rewards. Verify that one-step lookahead chooses the best immediate reward, two-step lookahead can choose a lower immediate reward with better cumulative return, and action selection restores the environment state.

- [ ] **Step 2: Run the tests and confirm failure**

Run: `python -m unittest tests.test_paper3_lookahead_baseline -v`

Expected: import failure because `scripts.analysis.paper3_lookahead_baseline` does not exist yet.

- [ ] **Step 3: Implement the minimal lookahead module**

Add a `LookaheadDecision` dataclass, snapshot/restore helpers for `BlockLevelEnv`, `select_lookahead_action`, `run_lookahead_baseline`, and JSON/LaTeX output writers.

- [ ] **Step 4: Run the tests and confirm pass**

Run: `python -m unittest tests.test_paper3_lookahead_baseline -v`

Expected: all tests pass without requiring restricted cadastral data.

### Task 2: Tighten Manuscript Claims

**Files:**
- Modify: `submission/ceus_anonymous/01_main_document_anonymous/manuscript_ceus_anonymous.tex`
- Sync: `manuscript/ceus_anonymous/manuscript_ceus_anonymous.tex`
- Sync: `manuscript/latex_source/06_latex_source_editable/manuscript_ceus_anonymous.tex`

- [ ] **Step 1: Revise the abstract**

Clarify that the core contribution is block-level spatial abstraction for scenario exploration and that DRL adds a township-specific increment rather than broad dominance.

- [ ] **Step 2: Revise the introduction contribution paragraph**

Frame `baimu fang` as a threshold-based consolidated operating-field metric for international CEUS readers.

- [ ] **Step 3: Revise results and discussion**

Separate observed results from interpretation, explicitly naming Reward-Greedy as a strong alternative and marking the planned limited-lookahead baseline as an open robustness test until results are available.

- [ ] **Step 4: Revise conclusion**

Keep the implication within the evidence: one district, parcel-count conservation, topology-first field metric, and decision-support use.

### Task 3: Verify Build and Submission Artifacts

**Files:**
- Modify if build succeeds: generated `.pdf`, `.aux`, `.log`, `.out` files in the main submission folder.
- Rebuild later after full result update: `CEUS_paper3_latex_source_anonymous.zip`

- [ ] **Step 1: Compile the anonymous manuscript**

Run from `submission/ceus_anonymous/01_main_document_anonymous`: `pdflatex -interaction=nonstopmode manuscript_ceus_anonymous.tex`

Expected: PDF generated. Warnings are acceptable only if they are standard unresolved layout warnings; citation or missing-file errors must be fixed.

- [ ] **Step 2: Inspect git diff**

Run: `git diff --stat` and `git diff --check`

Expected: manuscript edits, test/script additions, and no trailing whitespace errors.

- [ ] **Step 3: Record controlled-data experiment instructions**

Add Mac/Colab run instructions to reproducibility docs only after the baseline script passes local unit tests.
