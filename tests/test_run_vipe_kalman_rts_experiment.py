import subprocess
import sys
import tempfile
import unittest

from pathlib import Path

from scripts.run_vipe_kalman_rts_experiment import (
    compare_aggregate_metrics,
    discover_eligible_uuids,
    has_pose_ground_truth,
    render_compare_markdown,
    resolve_front_video_path,
    sample_uuids,
)


class VipeExperimentRunnerTest(unittest.TestCase):
    def test_sample_uuids_is_reproducible(self) -> None:
        uuids = ["a", "b", "c", "d", "e"]

        sample_a = sample_uuids(uuids, sample_size=3, seed=7)
        sample_b = sample_uuids(uuids, sample_size=3, seed=7)

        self.assertEqual(sample_a, sample_b)
        self.assertEqual(len(sample_a), 3)

    def test_resolve_front_video_path_prefers_simplecalib(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            uuid = "sample-uuid"
            uuid_dir = Path(tmpdir) / uuid
            simplecalib_video = (
                uuid_dir
                / "all_views_undistorted_simplecalib"
                / "camera"
                / "camera_front_wide_120fov_undistorted"
                / f"{uuid}.camera_front_wide_120fov_undistorted.mp4"
            )
            fallback_video = (
                uuid_dir
                / "all_views_undistorted"
                / "camera"
                / "camera_front_wide_120fov_undistorted"
                / f"{uuid}.camera_front_wide_120fov_undistorted.mp4"
            )
            simplecalib_video.parent.mkdir(parents=True)
            fallback_video.parent.mkdir(parents=True)
            simplecalib_video.touch()
            fallback_video.touch()

            resolved = resolve_front_video_path(uuid_dir)

            self.assertEqual(resolved, simplecalib_video)

    def test_compare_aggregate_metrics_computes_delta(self) -> None:
        raw_eval = {"aggregate": {"ate_se3_rmse_mean": 2.0, "rpe_trans_se3_rmse_mean": 1.0}}
        smooth_eval = {"aggregate": {"ate_se3_rmse_mean": 1.5, "rpe_trans_se3_rmse_mean": 0.8}}

        compare = compare_aggregate_metrics(raw_eval, smooth_eval)

        self.assertEqual(compare["ate_se3_rmse_mean"]["raw"], 2.0)
        self.assertEqual(compare["ate_se3_rmse_mean"]["kalman_rts"], 1.5)
        self.assertAlmostEqual(compare["ate_se3_rmse_mean"]["delta"], -0.5)

    def test_has_pose_ground_truth_requires_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            uuid = "sample-uuid"
            uuid_dir = Path(tmpdir) / uuid
            (uuid_dir / "labels" / "egomotion").mkdir(parents=True)
            (uuid_dir / "calibration" / "camera_intrinsics").mkdir(parents=True)
            (uuid_dir / "calibration" / "sensor_extrinsics").mkdir(parents=True)
            (uuid_dir / "labels" / "egomotion" / f"{uuid}.egomotion.parquet").touch()
            (uuid_dir / "calibration" / "camera_intrinsics" / "camera_intrinsics.parquet").touch()
            (uuid_dir / "calibration" / "sensor_extrinsics" / "sensor_extrinsics.parquet").touch()

            self.assertTrue(has_pose_ground_truth(uuid_dir))

            (uuid_dir / "labels" / "egomotion" / f"{uuid}.egomotion.parquet").unlink()
            self.assertFalse(has_pose_ground_truth(uuid_dir))

    def test_discover_eligible_uuids_supports_separate_gt_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_root = Path(tmpdir) / "video_root"
            gt_root = Path(tmpdir) / "gt_root"
            uuid = "sample-uuid"
            video_uuid_dir = video_root / uuid
            gt_uuid_dir = gt_root / uuid
            video_path = (
                video_uuid_dir
                / "all_views_undistorted_simplecalib"
                / "camera"
                / "camera_front_wide_120fov_undistorted"
                / f"{uuid}.camera_front_wide_120fov_undistorted.mp4"
            )
            video_path.parent.mkdir(parents=True)
            video_path.touch()
            (gt_uuid_dir / "labels" / "egomotion").mkdir(parents=True)
            (gt_uuid_dir / "calibration" / "camera_intrinsics").mkdir(parents=True)
            (gt_uuid_dir / "calibration" / "sensor_extrinsics").mkdir(parents=True)
            (gt_uuid_dir / "labels" / "egomotion" / f"{uuid}.egomotion.parquet").touch()
            (gt_uuid_dir / "calibration" / "camera_intrinsics" / "camera_intrinsics.parquet").touch()
            (gt_uuid_dir / "calibration" / "sensor_extrinsics" / "sensor_extrinsics.parquet").touch()

            eligible = discover_eligible_uuids(video_root, gt_by_uuid_root=gt_root)

            self.assertEqual(eligible, [uuid])

    def test_render_compare_markdown_mentions_both_methods(self) -> None:
        compare = {
            "ate_se3_rmse_mean": {
                "raw": 2.0,
                "kalman_rts": 1.5,
                "delta": -0.5,
            }
        }

        markdown = render_compare_markdown(compare, evaluated_count=10)

        self.assertIn("Raw ViPE", markdown)
        self.assertIn("Kalman+RTS", markdown)
        self.assertIn("ate_se3_rmse_mean", markdown)

    def test_runner_script_help_executes_directly(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/run_vipe_kalman_rts_experiment.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("sample-size", proc.stdout)


if __name__ == "__main__":
    unittest.main()
