# ViPE Kalman-RTS Front-Tele Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing ViPE Kalman/RTS experiment pipeline so it can evaluate the same 10 UUIDs on front tele, run up to 5 concurrent inference jobs on one GPU, and report raw-versus-smoothed tele pose metrics against GT.

**Architecture:** Generalize the existing wide-specific runner and evaluator into camera-parameterized utilities, then add bounded local concurrency to the experiment runner. Reuse the existing pose smoother unchanged, reuse the previously saved UUID list, and write a separate tele experiment root so tele and wide outputs remain directly comparable.

**Tech Stack:** Python 3.10, NumPy, SciPy, pyarrow, standard library `unittest`, standard library `subprocess`, existing ViPE `run.py` pipeline.

---

### Task 1: Parameterize Camera Selection In The Evaluator

**Files:**
- Modify: `scripts/vipe_front_pose_eval_ft.py`
- Modify: `tests/test_vipe_front_pose_eval_ft.py`

- [ ] **Step 1: Write failing tests for configurable camera paths**

```python
record = eval_mod.discover_front_sequences(base_dir, camera_name="camera_front_tele_30fov_undistorted")[0]
self.assertIn("camera_front_tele_30fov_undistorted", str(record.video_path))
self.assertIn("camera_front_tele_30fov_undistorted", str(record.timestamps_path))
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_vipe_front_pose_eval_ft -v`
Expected: FAIL because the evaluator still assumes wide-only camera names.

- [ ] **Step 3: Implement camera-name parameterization in the evaluator**

```python
def discover_front_sequences(base_dir: Path, camera_name: str = DEFAULT_CAMERA_NAME) -> list[FrontSequenceRecord]:
    ...

def load_front_ground_truth(video_uuid_dir: Path, gt_uuid_dir: Path | None = None, camera_name: str = DEFAULT_CAMERA_NAME) -> np.ndarray:
    ...
```

- [ ] **Step 4: Extend the CLI to accept `--camera-name`**

```python
parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
```

- [ ] **Step 5: Re-run evaluator tests until green**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_vipe_front_pose_eval_ft -v`
Expected: PASS.

- [ ] **Step 6: Commit evaluator parameterization**

```bash
git add scripts/vipe_front_pose_eval_ft.py tests/test_vipe_front_pose_eval_ft.py
git commit -m "feat: parameterize front pose eval camera selection"
```

### Task 2: Parameterize Camera Selection And UUID Reuse In The Runner

**Files:**
- Modify: `scripts/run_vipe_kalman_rts_experiment.py`
- Modify: `tests/test_run_vipe_kalman_rts_experiment.py`

- [ ] **Step 1: Write failing tests for tele path resolution and fixed sample-list reuse**

```python
video_path = runner.resolve_front_video_path(uuid_dir, camera_name="camera_front_tele_30fov_undistorted")
self.assertIn("camera_front_tele_30fov_undistorted", str(video_path))

sampled = runner.load_or_sample_uuids(eligible, sample_list_path=sample_path, sample_size=10, seed=123)
self.assertEqual(sampled, expected_sample)
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: FAIL because the runner still hardcodes wide and always samples internally.

- [ ] **Step 3: Implement camera-name and sample-list support**

```python
def resolve_front_video_path(uuid_dir: Path, camera_name: str = DEFAULT_CAMERA_NAME) -> Path:
    ...

def load_or_sample_uuids(..., sample_list_path: Path | None = None) -> list[str]:
    ...
```

- [ ] **Step 4: Extend the manifest to record `camera_name` and `sample_list_path`**

```python
write_json(experiment_root / "manifest.json", {"camera_name": args.camera_name, ...})
```

- [ ] **Step 5: Re-run runner tests until green**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: PASS.

- [ ] **Step 6: Commit runner parameterization**

```bash
git add scripts/run_vipe_kalman_rts_experiment.py tests/test_run_vipe_kalman_rts_experiment.py
git commit -m "feat: support tele camera experiment inputs"
```

### Task 3: Add Bounded Local Concurrency To The Runner

**Files:**
- Modify: `scripts/run_vipe_kalman_rts_experiment.py`
- Modify: `tests/test_run_vipe_kalman_rts_experiment.py`

- [ ] **Step 1: Write failing tests for concurrent task scheduling**

```python
events = runner.schedule_uuids_for_test(["a", "b", "c"], max_parallel_jobs=2, worker=fake_worker)
self.assertEqual(events["max_active"], 2)
self.assertEqual(events["completed"], ["a", "b", "c"])
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: FAIL because no concurrency helper exists yet.

- [ ] **Step 3: Implement a small local scheduler with `--max-parallel-jobs`**

```python
def run_raw_jobs_concurrently(..., max_parallel_jobs: int) -> list[dict]:
    ...
```

- [ ] **Step 4: Preserve per-UUID logs and continue past failures**

```python
run_log.append({"uuid": uuid, "stage": "raw_vipe", "returncode": proc.returncode, ...})
```

- [ ] **Step 5: Re-run runner tests until green**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: PASS.

- [ ] **Step 6: Commit concurrent runner support**

```bash
git add scripts/run_vipe_kalman_rts_experiment.py tests/test_run_vipe_kalman_rts_experiment.py
git commit -m "feat: add concurrent vipe experiment scheduling"
```

### Task 4: Add Tele Reporting And Cross-Experiment Summary

**Files:**
- Modify: `scripts/run_vipe_kalman_rts_experiment.py`
- Modify: `tests/test_run_vipe_kalman_rts_experiment.py`

- [ ] **Step 1: Write failing tests for tele report metadata**

```python
compare_md = runner.render_compare_markdown(compare, evaluated_count=10, camera_name="camera_front_tele_30fov_undistorted")
self.assertIn("camera_front_tele_30fov_undistorted", compare_md)
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: FAIL because reports do not yet carry camera context.

- [ ] **Step 3: Update JSON/Markdown summaries with camera metadata**

```python
compare_payload = {"camera_name": args.camera_name, ...}
```

- [ ] **Step 4: Re-run runner tests until green**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest tests.test_run_vipe_kalman_rts_experiment -v`
Expected: PASS.

- [ ] **Step 5: Commit reporting updates**

```bash
git add scripts/run_vipe_kalman_rts_experiment.py tests/test_run_vipe_kalman_rts_experiment.py
git commit -m "feat: add tele experiment reporting metadata"
```

### Task 5: Run Full Verification In The ViPE Environment

**Files:**
- Modify only if verification exposes a real defect.

- [ ] **Step 1: Run the full unit suite**

Run: `/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 2: Run a one-sequence tele smoke experiment**

Run:

```bash
/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python scripts/run_vipe_kalman_rts_experiment.py \
  --by-uuid-root /scratch/project/prj-02-phai-lab/lulin/longtail/TrajectoryCrafter/posttrain_symlink_bundle_b0/fullseq_h_576_w_1024/by_uuid \
  --gt-by-uuid-root /scratch/project/prj-02-phai-lab/lulin/longtail/data_ft/by_uuid \
  --sample-list-path /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_front10_eval_gpu/sampled_uuids.json \
  --experiment-root /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_tele10_smoke \
  --sample-size 1 \
  --camera-name camera_front_tele_30fov_undistorted \
  --max-parallel-jobs 1
```

Expected: raw and smoothed tele pose outputs plus eval summaries are produced.

- [ ] **Step 3: Fix any integration issues with minimal edits**

```python
# Adjust camera-name plumbing, scheduling, or report wiring only as needed.
```

- [ ] **Step 4: Re-run the smoke experiment until green**

Run the same command again.
Expected: PASS with tele outputs present.

- [ ] **Step 5: Commit verification fixes**

```bash
git add scripts/run_vipe_kalman_rts_experiment.py scripts/vipe_front_pose_eval_ft.py tests/test_run_vipe_kalman_rts_experiment.py tests/test_vipe_front_pose_eval_ft.py
git commit -m "fix: complete tele experiment integration"
```

### Task 6: Run The Full Tele Experiment And Produce Final Metrics

**Files:**
- Create outside repo: `/scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_tele10_eval_gpu/...`

- [ ] **Step 1: Launch the full tele experiment**

Run:

```bash
/scratch/project/prj-02-phai-lab/lulin/longtail/.conda/envs/vipe/bin/python scripts/run_vipe_kalman_rts_experiment.py \
  --by-uuid-root /scratch/project/prj-02-phai-lab/lulin/longtail/TrajectoryCrafter/posttrain_symlink_bundle_b0/fullseq_h_576_w_1024/by_uuid \
  --gt-by-uuid-root /scratch/project/prj-02-phai-lab/lulin/longtail/data_ft/by_uuid \
  --sample-list-path /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_front10_eval_gpu/sampled_uuids.json \
  --experiment-root /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_tele10_eval_gpu \
  --sample-size 10 \
  --camera-name camera_front_tele_30fov_undistorted \
  --max-parallel-jobs 5
```

- [ ] **Step 2: Verify output summaries exist**

Run:

```bash
ls /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_tele10_eval_gpu/eval
```

Expected: `raw_eval.json`, `kalman_rts_eval.json`, `compare.json`, `compare.md`.

- [ ] **Step 3: Create or refresh a human-readable report**

```text
Summarize mean and median metrics plus notable failure cases in eval/report.md.
```

- [ ] **Step 4: Capture final metrics and compare tele to wide**

Run:

```bash
cat /scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_tele10_eval_gpu/eval/compare.json
```

Expected: tele raw-vs-smoothed aggregate metrics are present and ready for reporting.
