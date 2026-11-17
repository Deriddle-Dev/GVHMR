import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "media_extract.py"
spec = importlib.util.spec_from_file_location("media_extract", SCRIPT)
media_extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media_extract)


class MediaExtractionTest(unittest.TestCase):
    def test_extracts_audio_and_each_subtitle(self):
        streams = [
            {"index": 0, "codec_type": "video"},
            {"index": 1, "codec_type": "audio"},
            {"index": 2, "codec_type": "subtitle"},
            {"index": 3, "codec_type": "subtitle"},
        ]
        calls = []

        def fake_command(args, **kwargs):
            calls.append(args)
            if args[0] == "ffprobe":
                return SimpleNamespace(stdout=json.dumps({"streams": streams}))
            Path(args[-1]).write_bytes(b"data")
            return SimpleNamespace(stdout=b"")

        with tempfile.TemporaryDirectory() as temp, patch.object(media_extract, "command", fake_command):
            output = Path(temp)
            report = media_extract.extract(Path("input.mp4"), output)
            self.assertEqual(report["audio"], "audio.wav")
            self.assertEqual(report["subtitles"], ["subtitles_2.srt", "subtitles_3.srt"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(json.loads((output / "manifest.json").read_text()), report)
            self.assertEqual([args[args.index("-map") + 1] for args in calls[1:]], ["0:1", "0:2", "0:3"])

    def test_missing_streams_are_not_errors(self):
        result = SimpleNamespace(stdout='{"streams":[{"index":0,"codec_type":"video"}]}')
        with tempfile.TemporaryDirectory() as temp, patch.object(media_extract, "command", return_value=result):
            report = media_extract.extract(Path("silent.mp4"), Path(temp), transcript=True)
            self.assertFalse(report["errors"])
            self.assertIn("No audio stream found.", report["messages"])
            self.assertIn("No embedded subtitle streams found.", report["messages"])
            self.assertIn("Transcription skipped: no extracted audio.", report["messages"])

    def test_bad_subtitle_keeps_successful_audio_and_reports_error(self):
        def fake_command(args, **kwargs):
            if args[0] == "ffprobe":
                return SimpleNamespace(stdout=json.dumps({"streams": [
                    {"index": 1, "codec_type": "audio"}, {"index": 2, "codec_type": "subtitle"}
                ]}))
            if args[-1].endswith(".srt"):
                raise subprocess.CalledProcessError(1, args, stderr=b"unsupported subtitle")
            Path(args[-1]).write_bytes(b"audio")
            return SimpleNamespace(stdout=b"")

        with tempfile.TemporaryDirectory() as temp, patch.object(media_extract, "command", fake_command):
            report = media_extract.extract(Path("input.mp4"), Path(temp))
            self.assertEqual(report["audio"], "audio.wav")
            self.assertEqual(report["subtitles"], [])
            self.assertIn("unsupported subtitle", report["errors"][0])
            self.assertTrue((Path(temp) / "manifest.json").is_file())

    def test_probe_failure_produces_error_manifest(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            media_extract, "command", side_effect=FileNotFoundError("ffprobe")
        ):
            report = media_extract.extract(Path("input.mp4"), Path(temp))
            self.assertIn("ffprobe", report["errors"][0])
            self.assertTrue((Path(temp) / "manifest.json").is_file())

    def test_transcription_is_saved_with_timestamps(self):
        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, path, vad_filter):
                self.path = path
                return iter([SimpleNamespace(start=1.5, end=2.75, text=" hello ")]), None

        def fake_command(args, **kwargs):
            if args[0] == "ffprobe":
                return SimpleNamespace(stdout='{"streams":[{"index":1,"codec_type":"audio"}]}')
            Path(args[-1]).write_bytes(b"audio")
            return SimpleNamespace(stdout=b"")

        with tempfile.TemporaryDirectory() as temp, patch.object(media_extract, "command", fake_command), \
                patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeModel)}):
            report = media_extract.extract(Path("input.mp4"), Path(temp), transcript=True)
            self.assertEqual(report["transcript"], "transcript.txt")
            self.assertEqual((Path(temp) / "transcript.txt").read_text(), "[1.50 - 2.75] hello\n")

    def test_ocr_samples_and_deduplicates_visible_text(self):
        class FakeCapture:
            def __init__(self, path):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return 1.0

            def read(self):
                self.index += 1
                return self.index <= 5, object()

            def release(self):
                pass

        image = SimpleNamespace(tobytes=lambda: b"png")
        cv2 = SimpleNamespace(VideoCapture=FakeCapture, CAP_PROP_FPS=5, imencode=lambda *args: (True, image))
        outputs = iter([b"Sign", b"Sign", b"New"])
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"cv2": cv2}), \
                patch.object(media_extract, "command", side_effect=lambda *a, **k: SimpleNamespace(stdout=next(outputs))):
            output = Path(temp) / "ocr.txt"
            media_extract.ocr_video(Path("input.mp4"), output)
            self.assertEqual(output.read_text(), "[0.00s]\nSign\n\n[4.00s]\nNew\n\n")


if __name__ == "__main__":
    unittest.main()
