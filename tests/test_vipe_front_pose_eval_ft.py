import tempfile
import unittest

import numpy as np

from pathlib import Path

from scripts.vipe_front_pose_eval_ft import (
    FRONT_VIEW_ROOT_CANDIDATES,
    compute_pose_metrics,
    resolve_gt_uuid_dir,
    resolve_front_view_root,
    summarize_metrics,
)


class VipeFrontPoseEvalTest(unittest.TestCase):
    def test_resolve_front_view_root_prefers_simplecalib(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            uuid_dir = Path(tmpdir) / "sample-uuid"
            (uuid_dir / FRONT_VIEW_ROOT_CANDIDATES[1]).mkdir(parents=True)
            (uuid_dir / FRONT_VIEW_ROOT_CANDIDATES[0]).mkdir(parents=True)

            resolved = resolve_front_view_root(uuid_dir)

            self.assertEqual(resolved, uuid_dir / FRONT_VIEW_ROOT_CANDIDATES[0])

    def test_compute_pose_metrics_is_zero_for_identical_poses(self) -> None:
        poses = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))

        metrics = compute_pose_metrics(poses, poses)

        self.assertAlmostEqual(metrics["ate_se3_rmse"], 0.0)
        self.assertAlmostEqual(metrics["ate_sim3_rmse"], 0.0)
        self.assertAlmostEqual(metrics["rpe_trans_se3_rmse"], 0.0)
        self.assertAlmostEqual(metrics["rpe_rot_se3_rmse_deg"], 0.0)

    def test_summarize_metrics_reports_pose_keys_only(self) -> None:
        rows = [
            {
                "ate_se3_rmse": 1.0,
                "ate_sim3_rmse": 0.5,
                "rpe_trans_se3_rmse": 0.25,
                "rpe_rot_se3_rmse_deg": 2.0,
                "rpe_trans_sim3_rmse": 0.2,
                "rpe_rot_sim3_rmse_deg": 1.5,
            },
            {
                "ate_se3_rmse": 2.0,
                "ate_sim3_rmse": 1.5,
                "rpe_trans_se3_rmse": 0.75,
                "rpe_rot_se3_rmse_deg": 4.0,
                "rpe_trans_sim3_rmse": 0.4,
                "rpe_rot_sim3_rmse_deg": 2.5,
            },
        ]

        summary = summarize_metrics(rows)

        self.assertEqual(summary["sequence_count"], 2)
        self.assertIn("ate_se3_rmse_mean", summary)
        self.assertIn("rpe_rot_se3_rmse_deg_median", summary)
        self.assertNotIn("intrinsics_rmse_mean", summary)

    def test_resolve_gt_uuid_dir_prefers_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            video_uuid_dir = base / "video_root" / "uuid-a"
            gt_uuid_dir = base / "gt_root" / "uuid-a"
            gt_uuid_dir.mkdir(parents=True)

            resolved = resolve_gt_uuid_dir(video_uuid_dir, gt_uuid_dir)

            self.assertEqual(resolved, gt_uuid_dir.resolve())

    def test_resolve_front_view_root_supports_front_view_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            uuid_dir = Path(tmpdir) / "sample-uuid"
            (uuid_dir / "front_view_undistorted").mkdir(parents=True)

            resolved = resolve_front_view_root(uuid_dir)

            self.assertEqual(resolved, uuid_dir / "front_view_undistorted")


if __name__ == "__main__":
    unittest.main()
