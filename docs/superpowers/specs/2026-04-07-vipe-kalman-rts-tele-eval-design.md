# ViPE Kalman-RTS Front-Tele Evaluation Design

## Goal

Extend the existing front-view ViPE experiment pipeline so it can rerun the same evaluation on the front tele camera and answer whether tele reduces the failure cases seen on front wide.

The tele experiment must:

- reuse the exact same 10 UUIDs as the completed wide experiment,
- run raw ViPE on front tele undistorted videos,
- apply the same Kalman filter + RTS pose smoothing to the ViPE pose artifact,
- evaluate raw and smoothed tele trajectories against GT camera pose, and
- produce tele-only reports plus an easy comparison against the earlier wide results.

## Scope

In scope:

- front tele undistorted camera only,
- reusing the previously sampled 10 UUIDs,
- running ViPE inference on tele videos,
- applying existing pose smoothing unchanged,
- evaluating pose metrics against GT camera pose,
- running inference with up to 5 concurrent jobs on one GPU,
- writing aggregate and per-sequence summaries.

Out of scope:

- resampling a new UUID set,
- retraining or modifying the ViPE model,
- changing the Kalman/RTS math,
- adding non-pose metrics,
- multi-camera fusion.

## Dataset Assumptions

Experiment input root:

`/scratch/project/prj-02-phai-lab/lulin/longtail/TrajectoryCrafter/posttrain_symlink_bundle_b0/fullseq_h_576_w_1024/by_uuid`

Each sampled UUID resolves into `data_ft/by_uuid/<uuid>`.

For all 10 sampled UUIDs, front tele already exists as an undistorted asset under:

- `all_views_undistorted_simplecalib/camera/camera_front_tele_30fov_undistorted/`

This means no extra rectification pipeline is needed. The experiment should consume the existing undistorted tele videos and timestamps directly.

Ground-truth pose reconstruction continues to use:

- `labels/egomotion/<uuid>.egomotion.parquet`
- `calibration/camera_intrinsics/camera_intrinsics.parquet`
- `calibration/sensor_extrinsics/sensor_extrinsics.parquet`

## Fixed UUID Set

The tele run must reuse this exact UUID list from the wide experiment:

- `002b0c9f-cce3-463a-8a2a-1d1661f79f49`
- `00b583bc-0ec5-4f51-acf2-198af454886c`
- `00c28ae0-6d20-4625-a0dd-e2cbca989ba2`
- `00c64711-01b7-4ae8-b438-626d20cf4190`
- `00d90a6a-3b6a-479e-b89f-f33f873d2a19`
- `00ed95a8-920e-4ceb-b1a1-08c9a454733a`
- `01043cf6-15c4-4480-a770-b53cbe6f1dba`
- `013afa9e-a142-4943-a76e-dee3aba9f30f`
- `0146b821-d511-4998-b01e-79b328b5cac8`
- `015be6bb-a4ef-42e0-8447-2482ffa9af57`

The runner may still support fixed-seed sampling generally, but this experiment should pin the tele run to the saved list so wide-vs-tele remains apples-to-apples.

## Recommended Architecture

Keep the existing script split and generalize it to camera-specific execution:

1. `scripts/vipe_kalman_rts.py`
   - no camera-specific changes,
   - continues to smooth a ViPE pose artifact in place-compatible format.

2. `scripts/vipe_front_pose_eval_ft.py`
   - accept a configurable undistorted camera name,
   - load timestamps for the requested camera,
   - select the matching sensor extrinsics row,
   - build GT camera pose for wide or tele with the same metric code.

3. `scripts/run_vipe_kalman_rts_experiment.py`
   - accept `--camera-name`,
   - optionally accept `--sample-list-path` so the tele run can reuse the saved wide UUIDs,
   - resolve videos from the requested camera,
   - run inference with bounded local concurrency,
   - write raw, smoothed, and eval outputs under a camera-specific experiment root.

## Concurrency Design

The tele experiment should support local process-level concurrency on one GPU.

Target behavior:

- keep up to 5 ViPE inference subprocesses active at once,
- each subprocess handles one UUID,
- when one subprocess finishes, immediately launch the next queued UUID,
- smoothing can run after each raw result is produced,
- one failed UUID must not stop the rest of the batch.

Implementation guidance:

- use a small scheduler in the runner rather than shell-level background jobs,
- record one run-log entry per UUID with return code and output tails,
- expose concurrency as a CLI flag such as `--max-parallel-jobs`,
- keep default behavior conservative for old workflows, but allow `5` for this tele run.

If the GPU cannot sustain 5 concurrent tele jobs, the runner should surface the failures clearly instead of silently dropping work. Automatic fallback to lower concurrency is optional, not required.

## Metrics

Keep the same pose metrics as the wide experiment so the reports remain directly comparable:

- `ate_se3_rmse`
- `ate_sim3_rmse`
- `rpe_trans_se3_rmse`
- `rpe_rot_se3_rmse_deg`
- `rpe_trans_sim3_rmse`
- `rpe_rot_sim3_rmse_deg`

Aggregate output should continue to report both mean and median.

## Output Layout

Recommended tele experiment root:

`/scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_tele10_eval_gpu`

Contents:

- `sampled_uuids.json`
- `manifest.json`
- `runs/raw/<uuid>/...`
- `runs/kalman_rts/<uuid>/...`
- `eval/raw_eval.json`
- `eval/kalman_rts_eval.json`
- `eval/compare.json`
- `eval/compare.md`
- `eval/report.md`

The manifest should include the requested camera name and concurrency setting.

## Verification

Before the full tele run:

- add or update unit tests for camera-name parameterization,
- add a test for loading tele GT timestamps and extrinsics,
- add a test for fixed sample-list reuse,
- add a smoke test for the concurrent runner that does not require a GPU.

After implementation:

- run the repo test suite in the ViPE environment,
- run at least one real tele smoke inference,
- run the full 10-sequence tele experiment with `--max-parallel-jobs 5`,
- verify that raw and smoothed evals both report `evaluated_count = 10` unless actual runtime failures occur.

## Success Criteria

This work is successful when:

- the runner can execute the tele experiment without any extra rectification step,
- the tele run reuses the same 10 UUIDs as the wide run,
- raw and smoothed tele results are both evaluated against GT camera pose,
- the run uses a single GPU with up to 5 concurrent ViPE subprocesses,
- the final reports make it easy to compare wide vs tele and raw vs Kalman/RTS.
