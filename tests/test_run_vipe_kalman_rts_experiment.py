import importlib
import unittest


class VipeExperimentRunnerImportTest(unittest.TestCase):
    def test_module_exists(self) -> None:
        importlib.import_module("scripts.run_vipe_kalman_rts_experiment")


if __name__ == "__main__":
    unittest.main()
