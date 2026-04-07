#!/usr/bin/env python3

"""Front-view pose-only evaluation for data_ft-style ViPE experiments."""

from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scipy.spatial.transform import Rotation, Slerp


UNDIST_CAMERA_NAME = "camera_front_wide_120fov_undistorted"
FRONT_VIEW_ROOT_CANDIDATES = (
    "all_views_undistorted_simplecalib",
    "all_views_undistorted",
)


@dataclass(frozen=True)
class FrontSequenceRecord:
    uuid: str
    uuid_dir: Path
    view_root: Path
    video_path: Path
    timestamps_path: Path
    egomotion_path: Path
    intrinsics_path: Path
    extrinsics_path: Path


def resolve_front_view_root(uuid_dir: Path) -> Path:
    for root_name in FRONT_VIEW_ROOT_CANDIDATES:
        candidate = uuid_dir / root_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported front-view root found under {uuid_dir}")


def resolve_gt_uuid_dir(video_uuid_dir: Path, gt_uuid_dir: Path | None = None) -> Path:
    if gt_uuid_dir is not None:
        return gt_uuid_dir.resolve()
    return video_uuid_dir.resolve()


def discover_front_sequences(base_dir: Path) -> list[FrontSequenceRecord]:
    records: list[FrontSequenceRecord] = []
    for uuid_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        try:
            view_root = resolve_front_view_root(uuid_dir.resolve())
        except FileNotFoundError:
            continue

        video_path = (
            view_root
            / "camera"
            / UNDIST_CAMERA_NAME
            / f"{uuid_dir.name}.{UNDIST_CAMERA_NAME}.mp4"
        )
        timestamps_path = (
            view_root
            / "camera"
            / UNDIST_CAMERA_NAME
            / f"{uuid_dir.name}.{UNDIST_CAMERA_NAME}.timestamps.parquet"
        )
        egomotion_path = uuid_dir.resolve() / "labels" / "egomotion" / f"{uuid_dir.name}.egomotion.parquet"
        intrinsics_path = uuid_dir.resolve() / "calibration" / "camera_intrinsics" / "camera_intrinsics.parquet"
        extrinsics_path = uuid_dir.resolve() / "calibration" / "sensor_extrinsics" / "sensor_extrinsics.parquet"

        if not all(path.exists() for path in (video_path, timestamps_path, egomotion_path, intrinsics_path, extrinsics_path)):
            continue

        records.append(
            FrontSequenceRecord(
                uuid=uuid_dir.name,
                uuid_dir=uuid_dir.resolve(),
                view_root=view_root,
                video_path=video_path,
                timestamps_path=timestamps_path,
                egomotion_path=egomotion_path,
                intrinsics_path=intrinsics_path,
                extrinsics_path=extrinsics_path,
            )
        )
    return records


def _read_parquet_dict(path: Path) -> dict:
    return pq.read_table(path).to_pydict()


def pose_matrix_from_components(qx: float, qy: float, qz: float, qw: float, x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    pose[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return pose


def interpolate_poses(source_timestamps: np.ndarray, source_poses: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_matrix(source_poses[:, :3, :3])
    slerp = Slerp(source_timestamps.astype(np.float64), rotations)
    clamped = np.clip(target_timestamps, source_timestamps[0], source_timestamps[-1]).astype(np.float64)
    interp_rot = slerp(clamped).as_matrix()
    interp_trans = np.stack(
        [
            np.interp(clamped, source_timestamps, source_poses[:, axis, 3])
            for axis in range(3)
        ],
        axis=1,
    )
    out = np.tile(np.eye(4, dtype=np.float64), (len(target_timestamps), 1, 1))
    out[:, :3, :3] = interp_rot
    out[:, :3, 3] = interp_trans
    return out


def compose_camera_trajectory(ego_poses: np.ndarray, camera_extrinsics: np.ndarray) -> np.ndarray:
    return ego_poses @ camera_extrinsics[None, :, :]


def _find_sensor_row(table: dict, sensor_name: str) -> int:
    for idx, value in enumerate(table["sensor_name"]):
        if value == sensor_name:
            return idx
    raise ValueError(f"Sensor {sensor_name} not found")


def _rigid_align_points(source_pts: np.ndarray, target_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_center = source_pts.mean(axis=0)
    tgt_center = target_pts.mean(axis=0)
    src_demean = source_pts - src_center
    tgt_demean = target_pts - tgt_center
    cov = src_demean.T @ tgt_demean / len(source_pts)
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = tgt_center - r @ src_center
    return r, t


def _sim3_align_points(source_pts: np.ndarray, target_pts: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_center = source_pts.mean(axis=0)
    tgt_center = target_pts.mean(axis=0)
    src_demean = source_pts - src_center
    tgt_demean = target_pts - tgt_center
    cov = src_demean.T @ tgt_demean / len(source_pts)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[-1, -1] = -1.0
    r = vt.T @ d @ u.T
    src_var = np.mean(np.sum(src_demean**2, axis=1))
    if src_var <= 1e-12:
        return 1.0, r, tgt_center - r @ src_center
    scale = float(np.sum(s * np.diag(d)) / src_var)
    t = tgt_center - scale * (r @ src_center)
    return scale, r, t


def _apply_rigid_alignment(pred_poses: np.ndarray, gt_poses: np.ndarray) -> np.ndarray:
    r, t = _rigid_align_points(pred_poses[:, :3, 3], gt_poses[:, :3, 3])
    out = pred_poses.copy()
    out[:, :3, :3] = r[None, :, :] @ out[:, :3, :3]
    out[:, :3, 3] = (r @ out[:, :3, 3].T).T + t[None, :]
    return out


def _apply_sim3_alignment(pred_poses: np.ndarray, gt_poses: np.ndarray) -> np.ndarray:
    scale, r, t = _sim3_align_points(pred_poses[:, :3, 3], gt_poses[:, :3, 3])
    out = pred_poses.copy()
    out[:, :3, :3] = r[None, :, :] @ out[:, :3, :3]
    out[:, :3, 3] = scale * (r @ out[:, :3, 3].T).T + t[None, :]
    return out


def _relative_poses(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(poses[:-1]) @ poses[1:]


def _rotation_angle_deg(rot_mats: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot_mats).magnitude() * (180.0 / np.pi)


def _compute_rpe_metrics(aligned_pred_poses: np.ndarray, gt_poses: np.ndarray) -> tuple[float, float]:
    rel_pred = _relative_poses(aligned_pred_poses)
    rel_gt = _relative_poses(gt_poses)
    rel_err = np.linalg.inv(rel_gt) @ rel_pred
    rpe_trans = np.linalg.norm(rel_err[:, :3, 3], axis=1)
    rpe_rot_deg = _rotation_angle_deg(rel_err[:, :3, :3])
    return float(np.sqrt(np.mean(rpe_trans**2))), float(np.sqrt(np.mean(rpe_rot_deg**2)))


def compute_pose_metrics(pred_poses: np.ndarray, gt_poses: np.ndarray) -> dict[str, float]:
    if pred_poses.shape != gt_poses.shape or pred_poses.ndim != 3 or pred_poses.shape[1:] != (4, 4):
        raise ValueError("pred_poses and gt_poses must both have shape (N, 4, 4)")
    if len(pred_poses) < 2:
        raise ValueError("at least two poses are required")

    pred_aligned = _apply_rigid_alignment(pred_poses, gt_poses)
    pred_aligned_sim3 = _apply_sim3_alignment(pred_poses, gt_poses)
    ate = np.linalg.norm(pred_aligned[:, :3, 3] - gt_poses[:, :3, 3], axis=1)
    ate_sim3 = np.linalg.norm(pred_aligned_sim3[:, :3, 3] - gt_poses[:, :3, 3], axis=1)
    rpe_trans_se3_rmse, rpe_rot_se3_rmse_deg = _compute_rpe_metrics(pred_aligned, gt_poses)
    rpe_trans_sim3_rmse, rpe_rot_sim3_rmse_deg = _compute_rpe_metrics(pred_aligned_sim3, gt_poses)
    return {
        "ate_se3_rmse": float(np.sqrt(np.mean(ate**2))),
        "ate_sim3_rmse": float(np.sqrt(np.mean(ate_sim3**2))),
        "rpe_trans_se3_rmse": rpe_trans_se3_rmse,
        "rpe_rot_se3_rmse_deg": rpe_rot_se3_rmse_deg,
        "rpe_trans_sim3_rmse": rpe_trans_sim3_rmse,
        "rpe_rot_sim3_rmse_deg": rpe_rot_sim3_rmse_deg,
    }


def load_front_ground_truth(video_uuid_dir: Path, gt_uuid_dir: Path | None = None) -> np.ndarray:
    video_uuid_dir = video_uuid_dir.resolve()
    gt_uuid_dir = resolve_gt_uuid_dir(video_uuid_dir, gt_uuid_dir)
    view_root = resolve_front_view_root(gt_uuid_dir)
    timestamps = _read_parquet_dict(
        view_root / "camera" / UNDIST_CAMERA_NAME / f"{gt_uuid_dir.name}.{UNDIST_CAMERA_NAME}.timestamps.parquet"
    )
    egomotion = _read_parquet_dict(gt_uuid_dir / "labels" / "egomotion" / f"{gt_uuid_dir.name}.egomotion.parquet")
    extrinsics = _read_parquet_dict(gt_uuid_dir / "calibration" / "sensor_extrinsics" / "sensor_extrinsics.parquet")

    ego_poses = np.stack(
        [
            pose_matrix_from_components(qx, qy, qz, qw, x, y, z)
            for qx, qy, qz, qw, x, y, z in zip(
                egomotion["qx"],
                egomotion["qy"],
                egomotion["qz"],
                egomotion["qw"],
                egomotion["x"],
                egomotion["y"],
                egomotion["z"],
            )
        ],
        axis=0,
    )
    interp_ego = interpolate_poses(
        source_timestamps=np.asarray(egomotion["timestamp"]),
        source_poses=ego_poses,
        target_timestamps=np.asarray(timestamps["timestamp"]),
    )
    extr_idx = _find_sensor_row(extrinsics, UNDIST_CAMERA_NAME)
    camera_extrinsics = pose_matrix_from_components(
        extrinsics["qx"][extr_idx],
        extrinsics["qy"][extr_idx],
        extrinsics["qz"][extr_idx],
        extrinsics["qw"][extr_idx],
        extrinsics["x"][extr_idx],
        extrinsics["y"][extr_idx],
        extrinsics["z"][extr_idx],
    )
    return compose_camera_trajectory(interp_ego, camera_extrinsics)


def find_single_artifact_stem(result_dir: Path) -> str:
    candidates = sorted((result_dir / "pose").glob("*.npz"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one pose artifact in {result_dir / 'pose'}, found {len(candidates)}")
    return candidates[0].stem


def evaluate_sequence(video_uuid_dir: Path, result_dir: Path, gt_uuid_dir: Path | None = None) -> dict[str, float]:
    artifact_stem = find_single_artifact_stem(result_dir)
    pred_pose_npz = np.load(result_dir / "pose" / f"{artifact_stem}.npz")
    pred_pose_inds = pred_pose_npz["inds"]
    pred_poses = pred_pose_npz["data"].astype(np.float64)
    gt_poses_all = load_front_ground_truth(video_uuid_dir, gt_uuid_dir=gt_uuid_dir)
    gt_poses = gt_poses_all[pred_pose_inds]
    metrics = compute_pose_metrics(pred_poses, gt_poses)
    metrics["uuid"] = video_uuid_dir.name
    metrics["num_frames"] = int(len(pred_pose_inds))
    metrics["artifact_stem"] = artifact_stem
    return metrics


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    metric_keys = [
        "ate_se3_rmse",
        "ate_sim3_rmse",
        "rpe_trans_se3_rmse",
        "rpe_rot_se3_rmse_deg",
        "rpe_trans_sim3_rmse",
        "rpe_rot_sim3_rmse_deg",
    ]
    out: dict[str, float] = {"sequence_count": len(rows)}
    for key in metric_keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean()) if len(values) else float("nan")
        out[f"{key}_median"] = float(np.median(values)) if len(values) else float("nan")
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate front-view ViPE pose outputs against data_ft ground truth.")
    parser.add_argument("--base-dir", type=Path, required=True, help="Base by_uuid directory")
    parser.add_argument("--result-dir", type=Path, required=True, help="Result root with one subdir per uuid")
    parser.add_argument("--sample-list-path", type=Path, required=True, help="JSON file containing target uuids")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to write evaluation summary json")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    uuids = json.loads(args.sample_list_path.read_text())
    rows: list[dict] = []
    missing: list[str] = []
    for uuid in uuids:
        uuid_dir = args.base_dir / uuid
        result_subdir = args.result_dir / uuid
        if not (result_subdir / "pose").exists():
            missing.append(uuid)
            continue
        rows.append(evaluate_sequence(uuid_dir, result_subdir))

    summary = {
        "base_dir": str(args.base_dir),
        "result_dir": str(args.result_dir),
        "sample_list_path": str(args.sample_list_path),
        "evaluated_count": len(rows),
        "missing_count": len(missing),
        "missing_uuids": missing,
        "aggregate": summarize_metrics(rows),
        "per_sequence": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(args.output_json)


if __name__ == "__main__":
    main()
