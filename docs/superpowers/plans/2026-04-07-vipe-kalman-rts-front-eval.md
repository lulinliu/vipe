# ViPE Kalman-RTS Front Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible in-repo experiment that reruns ViPE on 10 fixed-seed front-view videos, applies Kalman+RTS pose smoothing, and compares raw versus smoothed pose metrics.

**Architecture:** Keep ViPE inference untouched and add an external post-processing and evaluation layer under `scripts/`. The pipeline will discover and sample eligible UUIDs, run raw ViPE, smooth only the exported pose artifacts, evaluate both methods with a dataset-aware front-view pose evaluator, and write comparison summaries.

**Tech Stack:** Python 3.10, NumPy, SciPy, standard library `unittest`, existing ViPE CLI/inference artifacts, Parquet via `pyarrow`.

---

### Task 1: Create Documentation And Test Skeleton

**Files:**
- Create: `docs/superpowers/specs/2026-04-07-vipe-kalman-rts-front-eval-design.md`
- Create: `docs/superpowers/plans/2026-04-07-vipe-kalman-rts-front-eval.md`
- Create: `tests/test_vipe_kalman_rts.py`
- Create: `tests/test_vipe_front_pose_eval_ft.py`
- Create: `tests/test_run_vipe_kalman_rts_experiment.py`

- [ ] **Step 1: Write failing skeleton tests**

```python
import unittest


class PlaceholderTest(unittest.TestCase):
    def test_placeholder(self):
        self.fail("replace with real test")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest discover -s tests -v`
Expected: FAIL because placeholder tests intentionally fail.

- [ ] **Step 3: Replace placeholders with import-level failing tests**

```python
from scripts import vipe_kalman_rts  # noqa: F401
```

- [ ] **Step 4: Run tests to verify the modules do not exist yet**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest discover -s tests -v`
Expected: FAIL with `ImportError` for the new modules.

- [ ] **Step 5: Commit docs and test skeleton**

```bash
git add docs/superpowers/specs/2026-04-07-vipe-kalman-rts-front-eval-design.md \
        docs/superpowers/plans/2026-04-07-vipe-kalman-rts-front-eval.md \
        tests/test_vipe_kalman_rts.py tests/test_vipe_front_pose_eval_ft.py \
        tests/test_run_vipe_kalman_rts_experiment.py
git commit -m "docs: add vipe kalman rts experiment design"
```

### Task 2: Implement Pose Smoothing Utility

**Files:**
- Create: `scripts/vipe_kalman_rts.py`
- Modify: `tests/test_vipe_kalman_rts.py`

- [ ] **Step 1: Write failing tests for pose smoothing behavior**

```python
self.assertEqual(smoothed.shape, poses.shape)
self.assertTrue(np.array_equal(out_inds, inds))
self.assertLess(smoothed_error, noisy_error)
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_vipe_kalman_rts -v`
Expected: FAIL because smoothing utilities do not exist yet.

- [ ] **Step 3: Implement minimal smoothing module**

```python
def smooth_pose_sequence(poses: np.ndarray, dt: float = 1.0) -> np.ndarray:
    ...
```

- [ ] **Step 4: Add CLI entry points for reading and writing ViPE pose artifacts**

```bash
python scripts/vipe_kalman_rts.py --input-pose path/to/raw.npz --output-pose path/to/smoothed.npz
```

- [ ] **Step 5: Re-run targeted tests until green**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_vipe_kalman_rts -v`
Expected: PASS.

- [ ] **Step 6: Commit the smoothing utility**

```bash
git add scripts/vipe_kalman_rts.py tests/test_vipe_kalman_rts.py
git commit -m "feat: add vipe pose kalman rts smoothing"
```

### Task 3: Implement Front-View Pose Evaluator For data_ft Layout

**Files:**
- Create: `scripts/vipe_front_pose_eval_ft.py`
- Modify: `tests/test_vipe_front_pose_eval_ft.py`

- [ ] **Step 1: Write failing tests for path resolution and metric output**

```python
self.assertEqual(record.video_path, expected_path)
self.assertIn("ate_se3_rmse", metrics)
self.assertNotIn("intrinsics_rmse", metrics)
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_vipe_front_pose_eval_ft -v`
Expected: FAIL because evaluator module does not exist yet.

- [ ] **Step 3: Implement dataset-aware evaluator**

```python
def evaluate_sequence(uuid_dir: Path, result_dir: Path) -> dict[str, float]:
    ...
```

- [ ] **Step 4: Ensure only pose metrics are summarized**

```python
metric_keys = ["ate_se3_rmse", "ate_sim3_rmse", "rpe_trans_se3_rmse", "rpe_rot_se3_rmse_deg"]
```

- [ ] **Step 5: Re-run targeted tests until green**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_vipe_front_pose_eval_ft -v`
Expected: PASS.

- [ ] **Step 6: Commit evaluator changes**

```bash
git add scripts/vipe_front_pose_eval_ft.py tests/test_vipe_front_pose_eval_ft.py
git commit -m "feat: add front pose evaluator for data_ft"
```

### Task 4: Implement Experiment Orchestrator

**Files:**
- Create: `scripts/run_vipe_kalman_rts_experiment.py`
- Modify: `tests/test_run_vipe_kalman_rts_experiment.py`

- [ ] **Step 1: Write failing tests for deterministic sampling and path selection**

```python
self.assertEqual(sample_a, sample_b)
self.assertEqual(len(sample_a), 10)
self.assertTrue(video_path.name.endswith(".mp4"))
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: FAIL because the orchestrator does not exist yet.

- [ ] **Step 3: Implement discovery, sampling, raw inference, smoothing, and eval wiring**

```python
def run_experiment(...):
    sampled = sample_uuids(...)
    run_raw_vipe(...)
    run_smoothing(...)
    run_eval(...)
```

- [ ] **Step 4: Add summary writers for JSON and Markdown comparison outputs**

```python
def write_compare_markdown(raw_eval: dict, smooth_eval: dict, output_path: Path) -> None:
    ...
```

- [ ] **Step 5: Re-run targeted tests until green**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: PASS.

- [ ] **Step 6: Commit orchestrator changes**

```bash
git add scripts/run_vipe_kalman_rts_experiment.py tests/test_run_vipe_kalman_rts_experiment.py
git commit -m "feat: add vipe kalman rts experiment runner"
```

### Task 5: Run Integration Verification

**Files:**
- Modify: `scripts/run_vipe_kalman_rts_experiment.py`
- Modify: `scripts/vipe_kalman_rts.py`
- Modify: `scripts/vipe_front_pose_eval_ft.py`

- [ ] **Step 1: Run the full unit suite**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 2: Run a one-sequence smoke experiment**

Run:

```bash
/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python scripts/run_vipe_kalman_rts_experiment.py \
  --by-uuid-root /scratch/project/prj-02-phai-lab/lulin/longtail/TrajectoryCrafter/posttrain_symlink_bundle_b0/fullseq_h_576_w_1024/by_uuid \
  --experiment-root /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_front10_smoke \
  --sample-size 1
```

Expected: raw and smoothed result folders plus eval summaries are produced.

- [ ] **Step 3: Fix any integration issues with minimal edits**

```python
# Adjust path resolution, artifact naming, or summary plumbing only as needed.
```

- [ ] **Step 4: Re-run the smoke experiment until green**

Run the same command again.
Expected: PASS with output artifacts present.

- [ ] **Step 5: Commit integration fixes**

```bash
git add scripts/run_vipe_kalman_rts_experiment.py scripts/vipe_kalman_rts.py scripts/vipe_front_pose_eval_ft.py
git commit -m "fix: complete vipe kalman rts integration"
```

### Task 6: Run The Full 10-Sequence Experiment And Capture Results

**Files:**
- No code changes required unless verification exposes a defect.

- [ ] **Step 1: Launch the full experiment**

Run:

```bash
/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python scripts/run_vipe_kalman_rts_experiment.py \
  --by-uuid-root /scratch/project/prj-02-phai-lab/lulin/longtail/TrajectoryCrafter/posttrain_symlink_bundle_b0/fullseq_h_576_w_1024/by_uuid \
  --experiment-root /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_front10 \
  --sample-size 10
```

- [ ] **Step 2: Verify output summaries exist**

Run:

```bash
ls /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_front10/eval
```

Expected: `raw_eval.json`, `kalman_rts_eval.json`, `compare.json`, and `compare.md`.

- [ ] **Step 3: Inspect the final comparison metrics**

Run:

```bash
sed -n '1,200p' /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_front10/eval/compare.md
```

Expected: clear metric comparison between raw ViPE and smoothed ViPE.

- [ ] **Step 4: Commit any final code changes if needed**

```bash
git add <changed-files>
git commit -m "feat: finish vipe kalman rts front-view experiment"
```
