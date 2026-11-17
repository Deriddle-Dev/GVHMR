#!/usr/bin/env python3
"""Small PyQt6 front end for the GVHMR repository's video demo."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Some OpenCV builds change Qt's plugin paths to their own incompatible Qt build.
_qt_env = {key: os.environ.get(key) for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR")}
import cv2

for _key, _value in _qt_env.items():
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value

from explorer import OutputExplorer


class VideoPanel(QWidget):
    """Video preview with independent playback controls."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.capture: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.fps = 30.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self.image = QLabel(f"{title} video will appear here")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(480, 300)
        self.image.setStyleSheet("background: #151515; color: #dddddd;")

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderReleased.connect(self._seek_to_slider)
        self.position = QLabel("00:00 / 00:00")

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.seek, 1)
        controls.addWidget(self.position)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image, 1)
        layout.addLayout(controls)

    def clear_video(self) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame_count = 0
        self.play_button.setText("Play")
        self.seek.setRange(0, 0)
        self.position.setText("00:00 / 00:00")
        self.image.clear()
        self.image.setText(f"{self.title} video will appear here")

    def load_video(self, path: Path) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Could not open video: {path}")

        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        self.seek.setRange(0, max(0, self.frame_count - 1))
        self.seek.setValue(0)
        self._show_frame()
        self._update_position()

    def _toggle_playback(self) -> None:
        if self.capture is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            if self.capture.get(cv2.CAP_PROP_POS_FRAMES) >= self.frame_count:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.timer.start(max(1, round(1000 / self.fps)))
            self.play_button.setText("Pause")

    def _next_frame(self) -> None:
        if self.capture is None:
            return
        current = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        if current >= self.frame_count:
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self._show_frame()

    def _show_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            return
        height, width, channels = frame.shape
        image = QImage(
            frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_BGR888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self.image.setPixmap(
            pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.seek.blockSignals(True)
        self.seek.setValue(max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1))
        self.seek.blockSignals(False)
        self._update_position()

    def _seek_to_slider(self) -> None:
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.seek.value())
        self._show_frame()

    def _update_position(self) -> None:
        seconds = self.seek.value() / self.fps if self.fps else 0
        duration = self.frame_count / self.fps if self.fps else 0
        self.position.setText(f"{self._timestamp(seconds)} / {self._timestamp(duration)}")

    @staticmethod
    def _timestamp(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        if self.capture is not None:
            position = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, position - 1))
            self._show_frame()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        super().closeEvent(event)


class GVHMRWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GVHMR Video Viewer")
        self.resize(1100, 800)
        self.process: QProcess | None = None
        self.run_output_dir: Path | None = None
        self.run_video: Path | None = None
        self.job = ""
        self.stopping = False
        self.audio_path: Path | None = None

        self.repo_path = QLineEdit(str(Path(__file__).resolve().parents[2]))
        self.video_path = QLineEdit()
        self.output_path = QLineEdit(str(Path.home() / "gvhmr_outputs"))
        self.focal_length = QLineEdit()
        self.focal_length.setPlaceholderText("Optional, e.g. 24")
        self.static_camera = QCheckBox("Camera is static (skip visual odometry)")
        self.use_dpvo = QCheckBox("Use DPVO instead of the default SimpleVO")
        self.transcribe = QCheckBox("Transcribe speech locally (downloads Whisper model on first use)")
        self.ocr = QCheckBox("Read visible text with OCR (one frame every 2 seconds)")

        repo_button = QPushButton("Browse…")
        repo_button.clicked.connect(self._browse_repo)
        video_button = QPushButton("Browse…")
        video_button.clicked.connect(self._browse_video)
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(self._browse_output)
        self.input_controls = [
            self.repo_path, self.video_path, self.output_path, self.focal_length,
            self.static_camera, self.use_dpvo, self.transcribe, self.ocr,
            repo_button, video_button, output_button,
        ]

        form = QFormLayout()
        form.addRow("GVHMR repository", self._with_button(self.repo_path, repo_button))
        form.addRow("Input video", self._with_button(self.video_path, video_button))
        form.addRow("Output folder", self._with_button(self.output_path, output_button))
        form.addRow("Focal length (mm)", self.focal_length)
        form.addRow("", self.static_camera)
        form.addRow("", self.use_dpvo)
        form.addRow("", self.transcribe)
        form.addRow("", self.ocr)

        self.run_button = QPushButton("Run GVHMR")
        self.run_button.clicked.connect(self._run_model)
        self.extract_button = QPushButton("Extract audio and text")
        self.extract_button.clicked.connect(self._run_extraction)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_model)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        actions = QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(self.extract_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.progress, 1)

        self.views = QTabWidget()
        self.input_view = VideoPanel("Input")
        self.incam_view = VideoPanel("In-camera reconstruction")
        self.global_view = VideoPanel("Global/world reconstruction")
        self.combined_view = VideoPanel("Combined in-camera + global")
        self.tracking_view = VideoPanel("Person tracking")
        self.keypoints_view = VideoPanel("2D keypoints")
        self.views.addTab(self.input_view, "Input")
        self.views.addTab(self.tracking_view, "Person tracking")
        self.views.addTab(self.keypoints_view, "2D keypoints")
        self.views.addTab(self.incam_view, "In-camera 3D")
        self.views.addTab(self.global_view, "Global 3D")
        self.views.addTab(self.combined_view, "Combined")
        self.explorer = OutputExplorer(lambda: self.repo_path.text())
        self.views.addTab(self.explorer, "Explore outputs")
        self.media_view = QWidget()
        media_layout = QVBoxLayout(self.media_view)
        self.audio_button = QPushButton("Open extracted audio")
        self.audio_button.setEnabled(False)
        self.audio_button.clicked.connect(self._open_audio)
        self.media_text = QTextEdit()
        self.media_text.setReadOnly(True)
        media_layout.addWidget(self.audio_button)
        media_layout.addWidget(self.media_text)
        self.views.addTab(self.media_view, "Audio and text")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(form)
        layout.addLayout(actions)
        layout.addWidget(self.views, 1)
        layout.addWidget(QLabel("Inference log"))
        layout.addWidget(self.log)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Select a GVHMR checkout and input video.")

    @staticmethod
    def _with_button(field: QLineEdit, button: QPushButton) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        layout.addWidget(button)
        return container

    def _browse_repo(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select GVHMR repository")
        if path:
            self.repo_path.setText(path)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video",
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All files (*)",
        )
        if path:
            self.video_path.setText(path)
            try:
                self.input_view.load_video(Path(path))
            except RuntimeError as error:
                QMessageBox.warning(self, "Video preview", str(error))

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_path.setText(path)

    def _run_model(self) -> None:
        repo = Path(self.repo_path.text()).expanduser().resolve()
        video = Path(self.video_path.text()).expanduser().resolve()
        output_root = Path(self.output_path.text()).expanduser().resolve()
        demo_script = repo / "tools" / "demo" / "demo.py"

        if not demo_script.is_file():
            QMessageBox.warning(
                self,
                "GVHMR repository not found",
                "Select the root of a GVHMR checkout containing tools/demo/demo.py.",
            )
            return
        if not video.is_file():
            QMessageBox.warning(self, "Video not found", "Select an existing input video.")
            return
        if self.focal_length.text().strip():
            try:
                focal_length = int(self.focal_length.text().strip())
                if focal_length <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Invalid focal length", "Enter a positive number in mm.")
                return

        try:
            output_root.mkdir(parents=True, exist_ok=True)
            self.run_output_dir = Path(tempfile.mkdtemp(prefix="gvhmr-", dir=output_root))
        except OSError as error:
            QMessageBox.critical(self, "Cannot create output folder", str(error))
            return

        args = [str(demo_script), "--video", str(video), "--output_root", str(self.run_output_dir), "--verbose"]
        if self.static_camera.isChecked():
            args.append("--static_cam")
        if self.use_dpvo.isChecked():
            args.append("--use_dpvo")
        if self.focal_length.text().strip():
            args.extend(["--f_mm", self.focal_length.text().strip()])

        self.run_video = video
        self.explorer.clear()
        for panel in (self.tracking_view, self.keypoints_view, self.incam_view,
                      self.global_view, self.combined_view):
            panel.clear_video()
        self._start_process("model", repo, args)

    def _run_extraction(self) -> None:
        video = Path(self.video_path.text()).expanduser().resolve()
        if not video.is_file():
            QMessageBox.warning(self, "Video not found", "Select an existing input video.")
            return
        output_root = Path(self.output_path.text()).expanduser().resolve()
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            self.run_output_dir = Path(tempfile.mkdtemp(prefix="media-", dir=output_root))
        except OSError as error:
            QMessageBox.critical(self, "Cannot create output folder", str(error))
            return
        args = [str(Path(__file__).resolve().with_name("media_extract.py")), "--video", str(video),
                "--output", str(self.run_output_dir)]
        if self.transcribe.isChecked():
            args.append("--transcript")
        if self.ocr.isChecked():
            args.append("--ocr")
        self.run_video = video
        self.audio_path = None
        self.audio_button.setEnabled(False)
        self.media_text.clear()
        self._start_process("media", Path(__file__).resolve().parents[2], args)

    def _start_process(self, job: str, working_dir: Path, args: list[str]) -> None:
        self.job = job
        self.stopping = False
        self.log.clear()
        self._append_log(f"Starting: {sys.executable} {' '.join(args)}\n")
        self.run_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.explorer.open_button.setEnabled(False)
        for control in self.input_controls:
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.statusBar().showMessage("Working…")

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(working_dir))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.start(sys.executable, args)

    def _read_stdout(self) -> None:
        if self.process is not None:
            self._append_log(bytes(self.process.readAllStandardOutput()).decode(errors="replace"))

    def _read_stderr(self) -> None:
        if self.process is not None:
            self._append_log(bytes(self.process.readAllStandardError()).decode(errors="replace"))

    def _append_log(self, text: str) -> None:
        if text:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()

    def _on_process_error(self, error) -> None:
        if self.process is not None:
            self._append_log(f"\nProcess error: {self.process.errorString()}\n")
            if error == QProcess.ProcessError.FailedToStart:
                self._finish_run(False, "GVHMR could not be started.")

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        self._read_stdout()
        self._read_stderr()
        success = (not self.stopping and exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0)
        message = "Stopped." if self.stopping else ("Finished." if success else f"Failed (exit code {exit_code}).")
        self._finish_run(success, message)

    def _finish_run(self, success: bool, message: str) -> None:
        self.run_button.setEnabled(True)
        self.extract_button.setEnabled(True)
        self.explorer.open_button.setEnabled(True)
        for control in self.input_controls:
            control.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if success else 0)
        self.statusBar().showMessage(message)
        if self.process is not None and self.process.state() == QProcess.ProcessState.NotRunning:
            self.process.deleteLater()
            self.process = None

        if self.run_output_dir is None:
            return
        if self.job == "media":
            self._show_media_results()
            return
        if not success or self.run_video is None:
            return
        video_name = self.run_video.stem
        result_dir = self.run_output_dir / video_name
        videos = (
            (self.tracking_view, result_dir / "preprocess" / "bbx_xyxy_video_overlay.mp4"),
            (self.keypoints_view, result_dir / "preprocess" / "vitpose_video_overlay.mp4"),
            (self.incam_view, result_dir / "1_incam.mp4"),
            (self.global_view, result_dir / "2_global.mp4"),
            (self.combined_view, result_dir / f"{video_name}_3_incam_global_horiz.mp4"),
        )
        missing = [str(path) for _, path in videos if not path.is_file()]
        if missing:
            QMessageBox.warning(
                self,
                "Inference outputs missing",
                "GVHMR returned successfully, but these expected renders were not found:\n"
                + "\n".join(missing),
            )
            return
        try:
            self.input_view.load_video(self.run_video)
            for panel, path in videos:
                panel.load_video(path)
        except RuntimeError as error:
            QMessageBox.warning(self, "Could not load output", str(error))
            return

        self.views.setCurrentWidget(self.incam_view)
        prediction_path = result_dir / "hmr4d_results.pt"
        if prediction_path.is_file():
            self.explorer.load_file(prediction_path, Path(self.repo_path.text()).expanduser().resolve())
        else:
            QMessageBox.warning(self, "Prediction data missing", f"Not found: {prediction_path}")
        self._append_log(f"\nPrediction data: {prediction_path}\n")
        self._append_log(f"Rendered videos: {result_dir}\n")
        QMessageBox.information(self, "GVHMR complete", f"Outputs are in:\n{result_dir}")

    def _show_media_results(self) -> None:
        manifest_path = self.run_output_dir / "manifest.json"
        self.audio_path = None
        self.audio_button.setEnabled(False)
        self.media_text.clear()
        if not manifest_path.is_file():
            self.media_text.setPlainText("No extraction manifest was produced. Check the log.")
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            parts = []
            if manifest["audio"]:
                self.audio_path = self.run_output_dir / manifest["audio"]
                self.audio_button.setEnabled(self.audio_path.is_file())
                parts.append(f"Audio: {self.audio_path}")
            for name in manifest["subtitles"] + [manifest["transcript"], manifest["ocr"]]:
                if name:
                    parts.append(f"--- {name} ---\n{(self.run_output_dir / name).read_text(encoding='utf-8')}")
            parts.extend(manifest["messages"])
            parts.extend(f"Error: {message}" for message in manifest["errors"])
            self.media_text.setPlainText("\n\n".join(parts))
            self.views.setCurrentWidget(self.media_view)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Could not load media results", str(error))

    def _open_audio(self) -> None:
        if self.audio_path and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.audio_path))):
            QMessageBox.warning(self, "Audio", "Could not open the extracted audio in a media player.")

    def _stop_model(self) -> None:
        if self.process is None or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.stopping = True
        self.process.terminate()
        process = self.process
        QTimer.singleShot(3000, lambda: self._kill_if_running(process))

    def _kill_if_running(self, process: QProcess) -> None:
        if self.process is process and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
        self.explorer.shutdown()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = GVHMRWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
