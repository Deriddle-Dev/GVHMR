import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "explorer_data.py"
spec = importlib.util.spec_from_file_location("explorer_data", SCRIPT)
explorer_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explorer_data)


class ExplorerDataTest(unittest.TestCase):
    def test_selects_frame_series_and_preserves_static_arrays(self):
        prediction = {
            "smpl_params_global": {"body_pose": np.arange(12).reshape(2, 6)},
            "smpl_params_incam": {"transl": np.arange(6).reshape(2, 3)},
            "K_fullimg": np.arange(18).reshape(2, 3, 3),
            "net_outputs": {"model_output": {"pred_cam": np.arange(6).reshape(1, 2, 3)}},
            "calibration": np.array([4.0, 5.0]),
            "empty": np.array([]),
        }
        arrays, count = explorer_data.collect_arrays(prediction)
        self.assertEqual(count, 2)
        data, timed = arrays["net_outputs.model_output.pred_cam"]
        self.assertTrue(timed)
        np.testing.assert_array_equal(explorer_data.component_series(data, timed, 1), [1, 4])
        self.assertFalse(arrays["calibration"][1])
        np.testing.assert_array_equal(explorer_data.component_series(*arrays["calibration"], 0), [4, 5])
        self.assertEqual(arrays["K_fullimg"][0].shape, (2, 3, 3))
        self.assertNotIn("empty", arrays)

    def test_single_frame_batch_does_not_remove_frame_axis(self):
        prediction = {
            "smpl_params_global": {"body_pose": np.zeros((1, 63))},
            "net_outputs": {"model_output": {"pred_cam": np.ones((1, 1, 3))}},
        }
        arrays, count = explorer_data.collect_arrays(prediction)
        self.assertEqual(count, 1)
        self.assertEqual(arrays["smpl_params_global.body_pose"][0].shape, (1, 63))
        self.assertEqual(arrays["net_outputs.model_output.pred_cam"][0].shape, (1, 3))
        np.testing.assert_array_equal(
            explorer_data.component_series(*arrays["net_outputs.model_output.pred_cam"], 2), [1]
        )

    def test_missing_and_empty_predictions_are_rejected(self):
        for prediction in ({}, {"smpl_params_global": {"body_pose": np.zeros((0, 63))}}):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    explorer_data.collect_arrays(prediction)


if __name__ == "__main__":
    unittest.main()
