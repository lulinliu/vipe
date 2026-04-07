# ViPE Kalman-RTS Front-View Evaluation Design

## Goal

Build a reproducible experiment pipeline inside the `vipe` repository that:

- samples a fixed random set of 10 front-view sequences from the provided `by_uuid` root,
- reruns ViPE on those 10 sequences,
- applies Kalman filter + RTS smoothing to the exported camera pose trajectory,
- evaluates raw ViPE and smoothed ViPE with the same pose metrics, and
- produces a side-by-side comparison summary.

The experiment is front-view only and pose-metrics only.

## Scope

In scope:

- front undistorted camera only,
- fixed-seed random sampling,
- raw ViPE reruns on the sampled videos,
- post-processing on `pose/*.npz`,
- pose-only evaluation against dataset GT,
- aggregate and per-sequence comparison outputs.

Out of scope:

- depth evaluation,
- intrinsics reporting,
- multiview evaluation,
- changes to ViPE model inference internals.

## Dataset Assumptions

Experiment root:

`/scratch/project/prj-02-phai-lab/lulin/longtail/TrajectoryCrafter/posttrain_symlink_bundle_b0/fullseq_h_576_w_1024/by_uuid`

Each UUID entry resolves into:

`/scratch/project/prj-02-phai-lab/lulin/longtail/data_ft/by_uuid/<uuid>`

Front undistorted videos live under one of:

- `all_views_undistorted_simplecalib/camera/camera_front_wide_120fov_undistorted/`
- `all_views_undistorted/camera/camera_front_wide_120fov_undistorted/`

Ground-truth pose reconstruction uses:

- `labels/egomotion/<uuid>.egomotion.parquet`
- `calibration/camera_intrinsics/camera_intrinsics.parquet`
- `calibration/sensor_extrinsics/sensor_extrinsics.parquet`

## Recommended Architecture

Create three new scripts under `scripts/`:

1. `scripts/vipe_kalman_rts.py`
   - reads a ViPE pose artifact,
   - smooths translation and rotation over time,
   - writes a ViPE-compatible smoothed pose artifact.

2. `scripts/vipe_front_pose_eval_ft.py`
   - evaluates pose trajectories against GT for the `data_ft/by_uuid` layout,
   - reports pose metrics only.

3. `scripts/run_vipe_kalman_rts_experiment.py`
   - discovers eligible UUIDs,
   - samples 10 UUIDs with a fixed seed,
   - reruns raw ViPE,
   - runs pose smoothing,
   - evaluates both methods,
   - writes comparison summaries.

## Post-Processing Design

### Input format

ViPE pose artifacts contain:

- `inds`
- `data` with shape `(N, 4, 4)`

### Decomposition

For each pose:

- translation from `pose[:3, 3]`
- rotation from `pose[:3, :3]`

### Translation smoothing

Use a constant-velocity Kalman filter followed by RTS smoothing on:

- `[x, y, z, vx, vy, vz]`

Use the RTS-smoothed translation as final output.

### Rotation smoothing

Use a separate temporal smoothing path for rotation:

- convert rotation matrices to quaternions,
- enforce sign continuity,
- smooth in a linearized representation such as rotation vectors,
- re-normalize,
- convert back to rotation matrices.

Do not average rotation matrices directly.

### Reconstruction

Rebuild final `4x4` poses from smoothed translation and rotation while preserving:

- frame indices,
- artifact naming,
- sequence length.

## Metrics

Report pose metrics only:

- `ate_se3_rmse`
- `ate_sim3_rmse`
- `rpe_trans_se3_rmse`
- `rpe_rot_se3_rmse_deg`
- optional sim3 RPE variants if already computed by the evaluator

## Output Layout

Recommended experiment root:

`/scratch/project/prj-02-phai-lab/lulin/longtail/experiments/vipe_kalman_rts_front10`

Contents:

- `sampled_uuids.json`
- `manifest.json`
- `runs/raw/<uuid>/...`
- `runs/kalman_rts/<uuid>/...`
- `eval/raw_eval.json`
- `eval/kalman_rts_eval.json`
- `eval/compare.json`
- `eval/compare.md`

## Verification

Use lightweight tests that do not require extra dependencies beyond the supplied ViPE environment:

- unit tests for pose smoothing behavior on synthetic trajectories,
- unit tests for evaluator path resolution and metric plumbing,
- a smoke orchestration test for sample selection and result layout,
- one real single-sequence smoke run before the full 10-sequence experiment.

## Success Criteria

This work is successful when:

- the full 10-sequence experiment can be launched from within the `vipe` repo,
- raw and smoothed ViPE use the exact same sampled UUID list,
- both methods are evaluated with the same GT and metrics,
- the final summary clearly shows which method performs better.
