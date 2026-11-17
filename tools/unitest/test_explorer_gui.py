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
    from explorer import OutputExplorer


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
                offset = kwargs["transl"].numpy()[0, 0]
                vertices = Tensor([[[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]]])
                return types.SimpleNamespace(vertices=vertices)

        parameters = {
            "body_pose": Tensor(np.zeros((3, 63))),
            "betas": Tensor(np.zeros((3, 10))),
            "global_orient": Tensor(np.zeros((3, 3))),
            "transl": Tensor([[1, 10, 0], [2, 20, 0], [3, 30, 0]]),
        }
        prediction = {
            "smpl_params_global": parameters,
            "smpl_params_incam": parameters,
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
                self.wait_for(lambda: explorer.frame_count == 3 and bool(explorer.mesh_axes.collections))
                explorer.field.setCurrentText("net_outputs.model_output.pred_cam")
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
