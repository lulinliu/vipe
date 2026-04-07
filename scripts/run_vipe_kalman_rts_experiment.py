#!/usr/bin/env python3

"""Experiment runner for raw ViPE versus Kalman+RTS-smoothed ViPE poses."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys

from pathlib import Path

if __package__ in (None, ""):
    repo_root_for_imports = Path(__file__).resolve().parent.parent
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from scripts.vipe_front_pose_eval_ft import evaluate_sequence, summarize_metrics
from scripts.vipe_kalman_rts import smooth_pose_artifact


UNDIST_CAMERA_NAME = "camera_front_wide_120fov_undistorted"
FRONT_VIEW_ROOT_CANDIDATES = (
    "all_views_undistorted_simplecalib",
    "all_views_undistorted",
)


def resolve_front_video_path(uuid_dir: Path) -> Path:
    uuid_dir = uuid_dir.resolve()
    for root_name in FRONT_VIEW_ROOT_CANDIDATES:
        video_path = (
            uuid_dir
            / root_name
            / "camera"
            / UNDIST_CAMERA_NAME
            / f"{uuid_dir.name}.{UNDIST_CAMERA_NAME}.mp4"
        )
        if video_path.exists():
            return video_path
    raise FileNotFoundError(f"No front undistorted video found under {uuid_dir}")


def has_pose_ground_truth(uuid_dir: Path) -> bool:
    uuid_dir = uuid_dir.resolve()
    required_paths = (
        uuid_dir / "labels" / "egomotion" / f"{uuid_dir.name}.egomotion.parquet",
        uuid_dir / "calibration" / "camera_intrinsics" / "camera_intrinsics.parquet",
        uuid_dir / "calibration" / "sensor_extrinsics" / "sensor_extrinsics.parquet",
    )
    return all(path.exists() for path in required_paths)


def discover_eligible_uuids(by_uuid_root: Path, gt_by_uuid_root: Path | None = None) -> list[str]:
    eligible: list[str] = []
    for uuid_dir in sorted(path for path in by_uuid_root.iterdir() if path.is_dir() or path.is_symlink()):
        try:
            resolved_uuid_dir = uuid_dir.resolve()
            resolve_front_video_path(resolved_uuid_dir)
        except FileNotFoundError:
            continue
        gt_uuid_dir = (gt_by_uuid_root / uuid_dir.name) if gt_by_uuid_root is not None else resolved_uuid_dir
        if not has_pose_ground_truth(gt_uuid_dir):
            continue
        eligible.append(uuid_dir.name)
    return eligible


def sample_uuids(uuids: list[str], sample_size: int, seed: int) -> list[str]:
    if len(uuids) < sample_size:
        raise ValueError(f"Need at least {sample_size} eligible uuids, got {len(uuids)}")
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(uuids), sample_size))


def find_pose_artifact(result_dir: Path) -> Path | None:
    pose_dir = result_dir / "pose"
    artifacts = sorted(pose_dir.glob("*.npz"))
    return artifacts[0] if len(artifacts) == 1 else None


def run_raw_vipe(
    repo_root: Path,
    video_path: Path,
    output_dir: Path,
    python_executable: str | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    python_executable = python_executable or sys.executable
    cmd = [
        python_executable,
        "run.py",
        "pipeline=default",
        "streams=raw_mp4_stream",
        f"streams.base_path={video_path}",
        "pipeline.post.depth_align_model=null",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
        f"pipeline.output.path={output_dir}",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_smoothing(raw_uuid_dir: Path, smooth_uuid_dir: Path, dt: float) -> Path:
    input_pose = find_pose_artifact(raw_uuid_dir)
    if input_pose is None:
        raise FileNotFoundError(f"Expected exactly one pose artifact under {raw_uuid_dir / 'pose'}")
    output_pose = smooth_uuid_dir / "pose" / input_pose.name
    smooth_pose_artifact(input_pose, output_pose, dt=dt)
    return output_pose


def evaluate_result_root(video_by_uuid_root: Path, result_root: Path, uuids: list[str], gt_by_uuid_root: Path | None = None) -> dict:
    rows: list[dict] = []
    missing: list[str] = []
    for uuid in uuids:
        result_dir = result_root / uuid
        if find_pose_artifact(result_dir) is None:
            missing.append(uuid)
            continue
        gt_uuid_dir = (gt_by_uuid_root / uuid) if gt_by_uuid_root is not None else (video_by_uuid_root / uuid)
        rows.append(evaluate_sequence(video_by_uuid_root / uuid, result_dir, gt_uuid_dir=gt_uuid_dir))
    return {
        "base_dir": str(video_by_uuid_root),
        "gt_base_dir": str(gt_by_uuid_root) if gt_by_uuid_root is not None else str(video_by_uuid_root),
        "result_dir": str(result_root),
        "evaluated_count": len(rows),
        "missing_count": len(missing),
        "missing_uuids": missing,
        "aggregate": summarize_metrics(rows),
        "per_sequence": rows,
    }


def compare_aggregate_metrics(raw_eval: dict, smooth_eval: dict) -> dict[str, dict[str, float]]:
    raw_aggregate = raw_eval.get("aggregate", {})
    smooth_aggregate = smooth_eval.get("aggregate", {})
    compare: dict[str, dict[str, float]] = {}
    for key in sorted(set(raw_aggregate) & set(smooth_aggregate)):
        if key == "sequence_count":
            continue
        raw_value = raw_aggregate[key]
        smooth_value = smooth_aggregate[key]
        compare[key] = {
            "raw": raw_value,
            "kalman_rts": smooth_value,
            "delta": smooth_value - raw_value,
        }
    return compare


def render_compare_markdown(compare: dict[str, dict[str, float]], evaluated_count: int) -> str:
    lines = [
        "# ViPE Kalman+RTS Comparison",
        "",
        f"Evaluated sequences: {evaluated_count}",
        "",
        "| Metric | Raw ViPE | Kalman+RTS | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, values in compare.items():
        lines.append(
            f"| {metric} | {values['raw']:.6f} | {values['kalman_rts']:.6f} | {values['delta']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run raw ViPE versus Kalman+RTS-smoothed ViPE on fixed-seed front videos.")
    parser.add_argument("--by-uuid-root", type=Path, required=True, help="Input by_uuid root")
    parser.add_argument("--experiment-root", type=Path, required=True, help="Output experiment root")
    parser.add_argument("--gt-by-uuid-root", type=Path, default=None, help="Optional alternate by_uuid root that contains GT labels/calibration")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of UUIDs to sample")
    parser.add_argument("--seed", type=int, default=20260407, help="Random seed for reproducible sampling")
    parser.add_argument("--dt", type=float, default=1.0, help="Frame interval for smoothing")
    parser.add_argument("--force", action="store_true", help="Re-run raw ViPE even if outputs already exist")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    by_uuid_root = args.by_uuid_root
    gt_by_uuid_root = args.gt_by_uuid_root.resolve() if args.gt_by_uuid_root is not None else None
    experiment_root = args.experiment_root
    raw_root = experiment_root / "runs" / "raw"
    smooth_root = experiment_root / "runs" / "kalman_rts"
    eval_root = experiment_root / "eval"

    eligible_uuids = discover_eligible_uuids(by_uuid_root, gt_by_uuid_root=gt_by_uuid_root)
    sampled_uuids = sample_uuids(eligible_uuids, sample_size=args.sample_size, seed=args.seed)

    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "sampled_uuids.json").write_text(json.dumps(sampled_uuids, indent=2))
    write_json(
        experiment_root / "manifest.json",
        {
            "by_uuid_root": str(by_uuid_root),
            "gt_by_uuid_root": str(gt_by_uuid_root) if gt_by_uuid_root is not None else None,
            "experiment_root": str(experiment_root),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "dt": args.dt,
            "sampled_uuids": sampled_uuids,
        },
    )

    run_log: list[dict] = []
    for uuid in sampled_uuids:
        uuid_dir = by_uuid_root / uuid
        video_path = resolve_front_video_path(uuid_dir)
        raw_uuid_dir = raw_root / uuid
        smooth_uuid_dir = smooth_root / uuid

        pose_artifact = find_pose_artifact(raw_uuid_dir)
        if pose_artifact is None or args.force:
            infer_result = run_raw_vipe(repo_root, video_path, raw_uuid_dir, python_executable=sys.executable)
            run_log.append({"uuid": uuid, "stage": "raw_vipe", **infer_result})
            if infer_result["returncode"] != 0:
                continue
        else:
            run_log.append({"uuid": uuid, "stage": "raw_vipe", "status": "skipped_existing"})

        try:
            output_pose = run_smoothing(raw_uuid_dir, smooth_uuid_dir, dt=args.dt)
            run_log.append({"uuid": uuid, "stage": "kalman_rts", "output_pose": str(output_pose), "status": "ok"})
        except Exception as exc:
            run_log.append({"uuid": uuid, "stage": "kalman_rts", "status": "failed", "error": str(exc)})

    raw_eval = evaluate_result_root(by_uuid_root, raw_root, sampled_uuids, gt_by_uuid_root=gt_by_uuid_root)
    smooth_eval = evaluate_result_root(by_uuid_root, smooth_root, sampled_uuids, gt_by_uuid_root=gt_by_uuid_root)
    write_json(eval_root / "raw_eval.json", raw_eval)
    write_json(eval_root / "kalman_rts_eval.json", smooth_eval)

    compare = compare_aggregate_metrics(raw_eval, smooth_eval)
    compare_payload = {
        "evaluated_count": min(raw_eval["evaluated_count"], smooth_eval["evaluated_count"]),
        "compare": compare,
        "run_log": run_log,
    }
    write_json(eval_root / "compare.json", compare_payload)
    (eval_root / "compare.md").write_text(
        render_compare_markdown(compare, evaluated_count=compare_payload["evaluated_count"])
    )

    print(eval_root / "compare.json")


if __name__ == "__main__":
    main()
