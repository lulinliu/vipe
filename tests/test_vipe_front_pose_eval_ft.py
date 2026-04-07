import importlib
import unittest


class VipeFrontPoseEvalImportTest(unittest.TestCase):
    def test_module_exists(self) -> None:
        importlib.import_module("scripts.vipe_front_pose_eval_ft")


if __name__ == "__main__":
    unittest.main()
