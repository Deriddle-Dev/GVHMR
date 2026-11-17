import contextlib
import importlib.util
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
AVAILABLE = all(importlib.util.find_spec(name) for name in ("PyQt6", "numpy", "matplotlib"))

if AVAILABLE:
    import numpy as np
    from matplotlib.backend_bases import MouseEvent
    from PyQt6.QtWidgets import QApplication

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
    from explorer import MOTION_JOINTS, OutputExplorer, PredictionWorker


@unittest.skipUnless(AVAILABLE, "GUI dependencies are not installed")
class ExplorerGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate(), "Timed out waiting for the explorer worker")

    def test_select_plot_frame_and_rotate_mesh(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=float)
                self.shape = self.value.shape

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def __getitem__(self, item):
                return Tensor(self.value[item])

        class FakeModel:
            faces = np.array([[0, 1, 2]])

            def __call__(self, **kwargs):
                translation = kwargs["transl"].numpy()
                offset = translation[0, 0]
                vertices = Tensor([[[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]]])
                joints = np.repeat(translation[:, None, :], 22, axis=1)
                joints[:, :, 0] += np.arange(22)
                return types.SimpleNamespace(vertices=vertices, joints=Tensor(joints))

        parameters = {
            "body_pose": Tensor(np.arange(3 * 63).reshape(3, 63) / 100),
            "betas": Tensor(np.zeros((3, 10))),
            "global_orient": Tensor(np.zeros((3, 3))),
            "transl": Tensor([[1, 10, 0], [2, 20, 0], [3, 30, 0]]),
        }
        prediction = {
            "smpl_params_global": parameters,
            "smpl_params_incam": {**parameters, "transl": Tensor([[4, 40, 0], [5, 50, 0], [6, 60, 0]])},
            "net_outputs": {"model_output": {"pred_cam": Tensor(np.arange(9).reshape(1, 3, 3))}},
        }
        fake_torch = types.SimpleNamespace(
            load=lambda *args, **kwargs: prediction,
            inference_mode=contextlib.nullcontext,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            explorer = OutputExplorer(lambda: "/example/repo")
            explorer.worker.model = FakeModel()
            explorer.worker.model_repo = Path("/example/repo")
            try:
                explorer.load_file(Path("/example/prediction.pt"), Path("/example/repo"))
                self.wait_for(lambda: explorer.frame_count == 3 and bool(explorer.mesh_axes.collections)
                              and "smpl_params_global" in explorer.joints_by_coordinate)
                explorer.field.setCurrentText("Position: Left wrist")
                np.testing.assert_array_equal(explorer.plot_axes.lines[0].get_ydata(), [21, 22, 23])
                np.testing.assert_array_equal(explorer.plot_axes.lines[1].get_ydata(), [10, 20, 30])
                np.testing.assert_allclose(explorer.plot_axes.lines[0].get_xdata(), [0, 1 / 30, 2 / 30])
                explorer.slider.setValue(2)
                self.assertIn("X 23.000", explorer.value_label.text())
                np.testing.assert_allclose(explorer.marker.get_xdata(), [2 / 30, 2 / 30])
                explorer.body_mode.setCurrentIndex(1)
                self.wait_for(lambda: "smpl_params_incam" in explorer.joints_by_coordinate)
                np.testing.assert_array_equal(explorer.plot_axes.lines[0].get_ydata(), [24, 25, 26])
                explorer.field.setCurrentText("Position: Head")
                np.testing.assert_array_equal(explorer.plot_axes.lines[0].get_ydata(), [19, 20, 21])
                explorer.field.setCurrentText("Rotation: Head")
                np.testing.assert_allclose(explorer.plot_axes.lines[0].get_ydata(), [0.42, 1.05, 1.68])
                self.assertIn("axis-angle", explorer.value_label.text())
                self.assertFalse(explorer.component.isEnabled())
                explorer.field.setCurrentText("net_outputs.model_output.pred_cam")
                self.assertTrue(explorer.component.isEnabled())
                explorer.component.setValue(1)
                np.testing.assert_array_equal(explorer.plot_axes.lines[0].get_ydata(), [1, 4, 7])
                explorer.slider.setValue(2)
                self.wait_for(lambda: not explorer.mesh_busy)
                self.assertEqual(explorer.frame_label.text(), "Frame: 3 / 3")
                self.assertEqual(list(explorer.marker.get_xdata()), [3, 3])
                self.assertIn("Value: 7.0", explorer.value_label.text())
                explorer.body_mode.setCurrentIndex(1)
                self.wait_for(lambda: not explorer.mesh_busy)
                self.assertEqual(explorer.mesh_axes.get_zlabel(), "-Y (up)")

                for name, joint in MOTION_JOINTS.items():
                    explorer.field.setCurrentText(f"Position: {name}")
                    np.testing.assert_array_equal(
                        explorer.plot_axes.lines[0].get_ydata(), np.array([4, 5, 6]) + joint
                    )

                explorer.mesh_canvas.resize(500, 400)
                explorer.mesh_canvas.draw()
                before = explorer.mesh_axes.azim
                bbox = explorer.mesh_axes.bbox
                x, y = bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2
                for event_name, x_pos, y_pos in (
                    ("button_press_event", x, y),
                    ("motion_notify_event", x + 40, y + 20),
                    ("button_release_event", x + 40, y + 20),
                ):
                    event = MouseEvent(event_name, explorer.mesh_canvas, x_pos, y_pos, button=1)
                    explorer.mesh_canvas.callbacks.process(event_name, event)
                self.assertNotEqual(explorer.mesh_axes.azim, before)
            finally:
                explorer.shutdown()

    def test_long_joint_sequence_is_processed_in_chunks(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)
                self.shape = self.value.shape

            def __getitem__(self, item):
                return Tensor(self.value[item])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        calls = []

        class FakeModel:
            def __call__(self, **kwargs):
                translation = kwargs["transl"].numpy()
                calls.append(len(translation))
                return types.SimpleNamespace(joints=Tensor(np.repeat(translation[:, None, :], 22, axis=1)))

        frames = np.arange(70)
        params = {
            "body_pose": Tensor(np.zeros((70, 63))),
            "betas": Tensor(np.zeros((70, 10))),
            "global_orient": Tensor(np.zeros((70, 3))),
            "transl": Tensor(np.column_stack((frames, frames * 2, frames * 3))),
        }
        worker = PredictionWorker()
        worker.repo = Path("/repo")
        worker.model_repo = worker.repo
        worker.model = FakeModel()
        worker.prediction = {"smpl_params_global": params}
        results = []
        worker.joints_ready.connect(lambda generation, coordinate, joints: results.append((generation, coordinate, joints)))
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            worker.joints(5, "smpl_params_global")
            self.wait_for(lambda: bool(results))
        self.assertEqual(calls, [32, 32, 6])
        self.assertEqual(results[0][0:2], (5, "smpl_params_global"))
        np.testing.assert_array_equal(results[0][2][:, 20, 0], frames)

    def test_invalid_prediction_reports_failure(self):
        def fail_load(*args, **kwargs):
            raise RuntimeError("invalid prediction")

        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(load=fail_load)}):
            explorer = OutputExplorer(lambda: "/example/repo")
            try:
                explorer.load_file(Path("/example/prediction.pt"), Path("/example/repo"))
                self.wait_for(lambda: "invalid prediction" in explorer.status.text())
                self.assertFalse(explorer.field.isEnabled())
            finally:
                explorer.shutdown()


if __name__ == "__main__":
    unittest.main()
