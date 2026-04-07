import subprocess
import sys
import tempfile
import unittest

from pathlib import Path

from scripts.run_vipe_kalman_rts_experiment import (
    compare_aggregate_metrics,
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
