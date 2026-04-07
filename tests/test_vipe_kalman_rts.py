import tempfile
import unittest

import numpy as np

from scripts.vipe_kalman_rts import smooth_pose_artifact, smooth_pose_sequence


def _make_pose(tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return pose


class VipeKalmanRtsTest(unittest.TestCase):
    def test_smooth_pose_sequence_preserves_shape_and_rigid_form(self) -> None:
        poses = np.stack([_make_pose(float(i), 0.0, 0.0) for i in range(6)], axis=0)

        smoothed = smooth_pose_sequence(poses, dt=1.0)

        self.assertEqual(smoothed.shape, poses.shape)
        for pose in smoothed:
            rot = pose[:3, :3]
            self.assertTrue(np.allclose(rot.T @ rot, np.eye(3), atol=1e-5))
            self.assertTrue(np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0])))

    def test_smoothing_reduces_translation_error_on_noisy_track(self) -> None:
        gt_positions = np.array([[float(i), 0.0, 0.0] for i in range(8)], dtype=np.float64)
        noisy_positions = gt_positions.copy()
        noisy_positions[:, 0] += np.array([0.0, 0.6, -0.5, 0.7, -0.4, 0.5, -0.3, 0.0], dtype=np.float64)

        gt_poses = np.stack([_make_pose(*pos) for pos in gt_positions], axis=0)
        noisy_poses = np.stack([_make_pose(*pos) for pos in noisy_positions], axis=0)

        smoothed = smooth_pose_sequence(noisy_poses, dt=1.0)

        noisy_rmse = np.sqrt(np.mean((noisy_poses[:, :3, 3] - gt_poses[:, :3, 3]) ** 2))
        smoothed_rmse = np.sqrt(np.mean((smoothed[:, :3, 3] - gt_poses[:, :3, 3]) ** 2))
        self.assertLess(smoothed_rmse, noisy_rmse)

    def test_smooth_pose_artifact_preserves_inds(self) -> None:
        inds = np.array([2, 4, 6], dtype=np.int64)
        poses = np.stack([_make_pose(float(i), 0.0, 0.0) for i in inds], axis=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = f"{tmpdir}/raw_pose.npz"
            output_path = f"{tmpdir}/smooth_pose.npz"
            np.savez(input_path, inds=inds, data=poses)

            smooth_pose_artifact(input_path, output_path, dt=1.0)

            output = np.load(output_path)
            self.assertTrue(np.array_equal(output["inds"], inds))
            self.assertEqual(output["data"].shape, poses.shape)


if __name__ == "__main__":
    unittest.main()
