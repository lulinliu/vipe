import importlib
import unittest


class VipeKalmanRtsImportTest(unittest.TestCase):
    def test_module_exists(self) -> None:
        importlib.import_module("scripts.vipe_kalman_rts")


if __name__ == "__main__":
    unittest.main()
