#!/usr/bin/env python3

"""Kalman+RTS pose post-processing for ViPE pose artifacts."""

from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation


def _constant_velocity_kalman_rts(
    observations: np.ndarray,
    dt: float = 1.0,
    process_noise: float = 0.1,
    measurement_noise: float = 1.0,
) -> np.ndarray:
    """Smooth an NxD observation sequence with a constant-velocity Kalman+RTS model."""
    if observations.ndim != 2:
        raise ValueError("observations must have shape (N, D)")
    if len(observations) == 0:
        return observations.copy()
    if len(observations) == 1:
        return observations.copy()

    obs_dim = observations.shape[1]
    state_dim = obs_dim * 2

    F = np.eye(state_dim, dtype=np.float64)
    F[:obs_dim, obs_dim:] = np.eye(obs_dim, dtype=np.float64) * dt

    H = np.zeros((obs_dim, state_dim), dtype=np.float64)
    H[:, :obs_dim] = np.eye(obs_dim, dtype=np.float64)

    Q = np.eye(state_dim, dtype=np.float64) * process_noise
    Q[obs_dim:, obs_dim:] *= 2.0
    R = np.eye(obs_dim, dtype=np.float64) * measurement_noise

    x = np.zeros(state_dim, dtype=np.float64)
    x[:obs_dim] = observations[0]
    x[obs_dim:] = (observations[1] - observations[0]) / max(dt, 1e-6)
    P = np.eye(state_dim, dtype=np.float64) * 10.0

    x_filtered: list[np.ndarray] = []
    P_filtered: list[np.ndarray] = []
    x_predicted: list[np.ndarray] = []
    P_predicted: list[np.ndarray] = []

    for obs in observations:
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        innovation = obs - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ innovation
        P = (np.eye(state_dim, dtype=np.float64) - K @ H) @ P_pred

        x_predicted.append(x_pred.copy())
        P_predicted.append(P_pred.copy())
        x_filtered.append(x.copy())
        P_filtered.append(P.copy())

    x_smoothed = [state.copy() for state in x_filtered]
    P_smoothed = [cov.copy() for cov in P_filtered]

    for idx in range(len(observations) - 2, -1, -1):
        gain = P_filtered[idx] @ F.T @ np.linalg.inv(P_predicted[idx + 1])
        x_smoothed[idx] = x_filtered[idx] + gain @ (x_smoothed[idx + 1] - x_predicted[idx + 1])
        P_smoothed[idx] = P_filtered[idx] + gain @ (P_smoothed[idx + 1] - P_predicted[idx + 1]) @ gain.T

    return np.stack([state[:obs_dim] for state in x_smoothed], axis=0)


def _continuous_quaternions(rotations: Rotation) -> np.ndarray:
    """Flip quaternion signs to keep temporal continuity."""
    quats = rotations.as_quat().copy()
    if len(quats) <= 1:
        return quats

    for idx in range(1, len(quats)):
        if np.dot(quats[idx - 1], quats[idx]) < 0.0:
            quats[idx] *= -1.0
    return quats


def smooth_pose_sequence(
    poses: np.ndarray,
    dt: float = 1.0,
    translation_process_noise: float = 0.1,
    translation_measurement_noise: float = 1.0,
    rotation_process_noise: float = 0.01,
    rotation_measurement_noise: float = 0.1,
) -> np.ndarray:
    """Smooth a pose sequence while preserving SE(3) structure."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    if len(poses) == 0:
        return poses.copy()

    translations = poses[:, :3, 3]
    smoothed_translations = _constant_velocity_kalman_rts(
        translations,
        dt=dt,
        process_noise=translation_process_noise,
        measurement_noise=translation_measurement_noise,
    )

    rotations = Rotation.from_matrix(poses[:, :3, :3])
    continuous_quats = _continuous_quaternions(rotations)
    rotvecs = Rotation.from_quat(continuous_quats).as_rotvec()
    smoothed_rotvecs = _constant_velocity_kalman_rts(
        rotvecs,
        dt=dt,
        process_noise=rotation_process_noise,
        measurement_noise=rotation_measurement_noise,
    )
    smoothed_rotations = Rotation.from_rotvec(smoothed_rotvecs).as_matrix()

    smoothed_poses = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
    smoothed_poses[:, :3, :3] = smoothed_rotations
    smoothed_poses[:, :3, 3] = smoothed_translations
    return smoothed_poses


def smooth_pose_artifact(
    input_pose_path: str | Path,
    output_pose_path: str | Path,
    dt: float = 1.0,
) -> Path:
    """Read a ViPE pose artifact, smooth it, and write the output artifact."""
    input_pose_path = Path(input_pose_path)
    output_pose_path = Path(output_pose_path)

    pose_npz = np.load(input_pose_path)
    if "inds" not in pose_npz or "data" not in pose_npz:
        raise ValueError(f"{input_pose_path} must contain 'inds' and 'data'")

    inds = pose_npz["inds"]
    poses = pose_npz["data"]
    smoothed_poses = smooth_pose_sequence(poses, dt=dt)

    output_pose_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_pose_path, inds=inds, data=smoothed_poses)
    return output_pose_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Kalman+RTS smoothing to a ViPE pose artifact.")
    parser.add_argument("--input-pose", type=Path, required=True, help="Input ViPE pose .npz path")
    parser.add_argument("--output-pose", type=Path, required=True, help="Output smoothed pose .npz path")
    parser.add_argument("--dt", type=float, default=1.0, help="Frame interval used by the temporal model")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    smooth_pose_artifact(args.input_pose, args.output_pose, dt=args.dt)


if __name__ == "__main__":
    main()
