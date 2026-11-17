"""Interactive exploration of saved GVHMR predictions."""

import logging
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from explorer_data import collect_arrays, component_series


class PredictionWorker(QObject):
    loaded = pyqtSignal(int, object, int)
    mesh_ready = pyqtSignal(int, str, int, object, object)
    failed = pyqtSignal(int, str, str)

    def __init__(self):
        super().__init__()
        self.prediction = None
        self.model = None
        self.model_repo = None

    @pyqtSlot(int, str, str)
    def load(self, generation, path, repo):
        self.prediction = None
        try:
            import torch

            # Only tensor/primitive checkpoints are accepted; never unpickle arbitrary objects.
            prediction = torch.load(path, map_location="cpu", weights_only=True)
            arrays, count = collect_arrays(prediction)
            for name in ("smpl_params_global", "smpl_params_incam"):
                for key in ("body_pose", "betas", "global_orient", "transl"):
                    if key not in prediction[name]:
                        raise ValueError(f"Prediction is missing {name}.{key}.")
            self.prediction = prediction
            self.repo = Path(repo)
            self.loaded.emit(generation, arrays, count)
        except Exception as exc:
            logging.exception("Could not load GVHMR predictions")
            self.failed.emit(generation, "load", str(exc))

    @pyqtSlot(int, str, int)
    def mesh(self, generation, coordinate, frame):
        try:
            import torch

            if self.model is None or self.model_repo != self.repo:
                from hmr4d.utils.body_model.body_model_smplx import BodyModelSMPLX

                self.model = BodyModelSMPLX(
                    model_path=self.repo / "inputs/checkpoints/body_models",
                    model_type="smplx",
                    gender="neutral",
                    num_pca_comps=12,
                    flat_hand_mean=False,
                ).eval()
                self.model_repo = self.repo
            parameters = self.prediction[coordinate]
            inputs = {key: parameters[key][frame : frame + 1] for key in (
                "body_pose", "betas", "global_orient", "transl"
            )}
            with torch.inference_mode():
                vertices = self.model(**inputs).vertices[0].detach().cpu().numpy()
            faces = np.asarray(self.model.faces)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError("Body vertices are invalid or contain non-finite values.")
            if (faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0
                    or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError("Body mesh faces are invalid.")
            self.mesh_ready.emit(generation, coordinate, frame, vertices, faces)
        except Exception as exc:
            logging.exception("Could not reconstruct body mesh")
            self.failed.emit(generation, "mesh", str(exc))


class OutputExplorer(QWidget):
    load_requested = pyqtSignal(int, str, str)
    mesh_requested = pyqtSignal(int, str, int)

    def __init__(self, repo_provider, parent=None):
        super().__init__(parent)
        self.repo_provider = repo_provider
        self.arrays = {}
        self.frame_count = 0
        self.generation = 0
        self.mesh_busy = False
        self.mesh_pending = None
        self.mesh_enabled = False
        self.marker = None

        self.thread = QThread(self)
        self.worker = PredictionWorker()
        self.worker.moveToThread(self.thread)
        self.thread.finished.connect(self.worker.deleteLater)
        self.load_requested.connect(self.worker.load)
        self.mesh_requested.connect(self.worker.mesh)
        self.worker.loaded.connect(self._loaded)
        self.worker.mesh_ready.connect(self._mesh_ready)
        self.worker.failed.connect(self._failed)
        self.thread.start()

        self.open_button = QPushButton("Open predictions...")
        self.open_button.clicked.connect(self._choose_file)
        self.body_mode = QComboBox()
        self.body_mode.addItem("Global body", "smpl_params_global")
        self.body_mode.addItem("In-camera body", "smpl_params_incam")
        self.body_mode.currentIndexChanged.connect(self._queue_mesh)
        self.field = QComboBox()
        self.field.currentIndexChanged.connect(self._field_changed)
        self.component = QSpinBox()
        self.component.valueChanged.connect(self._plot)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(QLabel("Body:"))
        controls.addWidget(self.body_mode)
        controls.addWidget(QLabel("Output:"))
        controls.addWidget(self.field, 1)
        controls.addWidget(QLabel("Component:"))
        controls.addWidget(self.component)

        self.mesh_figure = Figure(figsize=(5, 4))
        self.mesh_canvas = FigureCanvasQTAgg(self.mesh_figure)
        self.mesh_axes = self.mesh_figure.add_subplot(111, projection="3d")
        mesh_layout = QVBoxLayout()
        mesh_layout.addWidget(QLabel("Drag to rotate; use the toolbar to zoom or reset."))
        mesh_layout.addWidget(NavigationToolbar2QT(self.mesh_canvas, self))
        mesh_layout.addWidget(self.mesh_canvas, 1)

        self.plot_figure = Figure(figsize=(5, 4), layout="tight")
        self.plot_canvas = FigureCanvasQTAgg(self.plot_figure)
        self.plot_axes = self.plot_figure.add_subplot(111)
        self.value_label = QLabel("Value: -")
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(NavigationToolbar2QT(self.plot_canvas, self))
        plot_layout.addWidget(self.plot_canvas, 1)
        plot_layout.addWidget(self.value_label)
        plots = QHBoxLayout()
        plots.addLayout(mesh_layout, 1)
        plots.addLayout(plot_layout, 1)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._frame_changed)
        self.frame_label = QLabel("Frame: -")
        timeline = QHBoxLayout()
        timeline.addWidget(self.play_button)
        timeline.addWidget(self.slider, 1)
        timeline.addWidget(self.frame_label)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.status = QLabel("Open a prediction file or run GVHMR.")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(plots, 1)
        layout.addLayout(timeline)
        layout.addWidget(self.status)
        self._set_available(False)

    def _set_available(self, enabled):
        for control in (self.body_mode, self.field, self.component, self.slider, self.play_button):
            control.setEnabled(enabled)

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open GVHMR predictions", "", "PyTorch predictions (*.pt)")
        if path:
            self.load_file(Path(path), Path(self.repo_provider()).expanduser().resolve())

    def load_file(self, path, repo):
        self.clear()
        self.status.setText(f"Loading {path}...")
        self.load_requested.emit(self.generation, str(path), str(repo))

    def clear(self):
        self.generation += 1
        self.timer.stop()
        self.play_button.setText("Play")
        self._set_available(False)
        self.arrays = {}
        self.frame_count = 0
        self.mesh_enabled = False
        self.mesh_busy = False
        self.mesh_pending = None
        self.field.clear()
        self.slider.setRange(0, 0)
        self.frame_label.setText("Frame: -")
        self.value_label.setText("Value: -")
        self.mesh_axes.clear()
        self.plot_axes.clear()
        self.mesh_canvas.draw_idle()
        self.plot_canvas.draw_idle()
        self.status.setText("Open a prediction file or run GVHMR.")

    @pyqtSlot(int, object, int)
    def _loaded(self, generation, arrays, count):
        if generation != self.generation:
            return
        self.arrays = arrays
        self.frame_count = count
        self.mesh_enabled = True
        self.field.blockSignals(True)
        self.field.clear()
        self.field.addItems(sorted(arrays))
        self.field.blockSignals(False)
        self.slider.setRange(0, count - 1)
        self.slider.setValue(0)
        self._set_available(True)
        self.status.setText("Select an output and component; drag the body to rotate it.")
        self._field_changed()
        self._frame_changed()

    @pyqtSlot(int, str, str)
    def _failed(self, generation, stage, message):
        if generation != self.generation:
            return
        if stage == "mesh":
            self.mesh_busy = False
            self.mesh_pending = None
            self.mesh_enabled = False
            self.status.setText(f"Body mesh unavailable: {message}. Parameter plots remain available.")
        else:
            self.status.setText(f"Could not load predictions: {message}")

    def _field_changed(self):
        item = self.arrays.get(self.field.currentText())
        if item is None:
            return
        array, timed = item
        count = int(np.prod(array.shape[1:])) if timed else array.size
        self.component.blockSignals(True)
        self.component.setRange(0, max(0, count - 1))
        self.component.setValue(0)
        self.component.blockSignals(False)
        self._plot()

    def _plot(self):
        item = self.arrays.get(self.field.currentText())
        if item is None:
            return
        array, timed = item
        values = component_series(array, timed, self.component.value())
        self.plot_axes.clear()
        positions = np.arange(1, len(values) + 1) if timed else np.arange(len(values))
        self.plot_axes.plot(positions, values)
        self.plot_axes.set_title(self.field.currentText(), fontsize=9)
        self.plot_axes.set_xlabel("Frame" if timed else "Flattened element (static)")
        self.plot_axes.set_ylabel("Value")
        self.marker = self.plot_axes.axvline(self.slider.value() + 1 if timed else self.component.value(), color="red")
        self._show_value()
        self.plot_canvas.draw_idle()

    def _show_value(self):
        item = self.arrays.get(self.field.currentText())
        if item is None:
            return
        array, timed = item
        shape = array.shape[1:] if timed else array.shape
        component = self.component.value()
        index = np.unravel_index(component, shape) if shape else ()
        value = array[(self.slider.value(), *index)] if timed else array[index]
        self.value_label.setText(f"Shape: {shape or 'scalar'}   Component: {index}   Value: {value}")

    def _frame_changed(self):
        if not self.frame_count:
            return
        frame = self.slider.value()
        self.frame_label.setText(f"Frame: {frame + 1} / {self.frame_count}")
        item = self.arrays.get(self.field.currentText())
        if item is not None and item[1] and self.marker is not None:
            self.marker.set_xdata([frame + 1, frame + 1])
            self._show_value()
            self.plot_canvas.draw_idle()
        self._queue_mesh()

    def _queue_mesh(self, *_):
        if not self.frame_count or not self.mesh_enabled:
            return
        self.mesh_pending = (self.body_mode.currentData(), self.slider.value())
        self._dispatch_mesh()

    def _dispatch_mesh(self):
        if self.mesh_busy or self.mesh_pending is None:
            return
        coordinate, frame = self.mesh_pending
        self.mesh_pending = None
        self.mesh_busy = True
        self.mesh_requested.emit(self.generation, coordinate, frame)

    @pyqtSlot(int, str, int, object, object)
    def _mesh_ready(self, generation, coordinate, frame, vertices, faces):
        if generation != self.generation:
            return
        self.mesh_busy = False
        if coordinate == self.body_mode.currentData() and frame == self.slider.value():
            elev, azim = self.mesh_axes.elev, self.mesh_axes.azim
            self.mesh_axes.clear()
            # World Y points up; camera-space Y points down.
            points = vertices[:, [0, 2, 1]].copy()
            if coordinate == "smpl_params_incam":
                points[:, 2] *= -1
            self.mesh_axes.add_collection3d(
                Poly3DCollection(points[faces], facecolor="#8eaec5", edgecolor="none", alpha=1.0)
            )
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            radius = max(float(np.ptp(points, axis=0).max()) * 0.55, 0.1)
            self.mesh_axes.set_xlim(center[0] - radius, center[0] + radius)
            self.mesh_axes.set_ylim(center[1] - radius, center[1] + radius)
            self.mesh_axes.set_zlim(center[2] - radius, center[2] + radius)
            self.mesh_axes.set_xlabel("X")
            self.mesh_axes.set_ylabel("Z")
            self.mesh_axes.set_zlabel("-Y (up)" if coordinate == "smpl_params_incam" else "Y (up)")
            self.mesh_axes.view_init(elev=elev, azim=azim)
            self.mesh_canvas.draw_idle()
        self._dispatch_mesh()

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            if self.slider.value() == self.frame_count - 1:
                self.slider.setValue(0)
            self.timer.start(33)  # GVHMR's demo exports at 30 fps.
            self.play_button.setText("Pause")

    def _advance(self):
        if self.slider.value() >= self.frame_count - 1:
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self.slider.setValue(self.slider.value() + 1)

    def shutdown(self):
        self.timer.stop()
        self.thread.quit()
        self.thread.wait()
