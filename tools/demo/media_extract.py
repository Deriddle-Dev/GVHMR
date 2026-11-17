"""Extract optional media sidecars from a video. GVHMR does not use these as input."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def command(args, **kwargs):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def extract(video, output, transcript=False, ocr=False):
    output.mkdir(parents=True, exist_ok=True)
    report = {"audio": None, "subtitles": [], "transcript": None, "ocr": None, "messages": [], "errors": []}

    def error(message):
        report["errors"].append(message)
        print(message, file=sys.stderr, flush=True)

    try:
        info = command(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)], text=True
        )
        streams = json.loads(info.stdout)["streams"]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        error(f"Could not inspect media with ffprobe: {exc}")
        (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio:
        try:
            path = output / "audio.wav"
            command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-map",
                     f"0:{audio['index']}", "-vn", "-c:a", "pcm_s16le", str(path)])
            report["audio"] = path.name
        except (OSError, subprocess.CalledProcessError) as exc:
            error(f"Audio extraction failed: {describe(exc)}")
    else:
        report["messages"].append("No audio stream found.")

    subtitles = [stream for stream in streams if stream.get("codec_type") == "subtitle"]
    if not subtitles:
        report["messages"].append("No embedded subtitle streams found.")
    for stream in subtitles:
        index = stream["index"]
        path = output / f"subtitles_{index}.srt"
        try:
            command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-map",
                     f"0:{index}", str(path)])
            report["subtitles"].append(path.name)
        except (OSError, subprocess.CalledProcessError) as exc:
            error(f"Subtitle stream {index} could not be converted to text: {describe(exc)}")

    if transcript:
        if not report["audio"]:
            report["messages"].append("Transcription skipped: no extracted audio.")
        else:
            try:
                from faster_whisper import WhisperModel

                print("Transcribing audio with faster-whisper (small, CPU)...", flush=True)
                model = WhisperModel("small", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(str(output / report["audio"]), vad_filter=True)
                path = output / "transcript.txt"
                with path.open("w", encoding="utf-8") as file:
                    for segment in segments:
                        file.write(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}\n")
                report["transcript"] = path.name
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                error(f"Transcription failed: {exc}")

    if ocr:
        try:
            ocr_video(video, output / "ocr.txt")
            report["ocr"] = "ocr.txt"
        except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            error(f"OCR failed: {describe(exc)}")

    (output / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for message in report["messages"]:
        print(message, flush=True)
    print(f"Media results: {output}", flush=True)
    return report


def describe(exc):
    if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
        return exc.stderr.decode(errors="replace").strip() if isinstance(exc.stderr, bytes) else exc.stderr.strip()
    return str(exc)


def ocr_video(video, destination):
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for OCR: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("Cannot sample OCR frames: video frame rate is unavailable.")
    step = max(1, round(fps * 2))
    index = 0
    previous = None
    try:
        with destination.open("w", encoding="utf-8") as out:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % step == 0:
                    encoded, image = cv2.imencode(".png", frame)
                    if not encoded:
                        raise RuntimeError(f"Could not encode frame {index} for OCR.")
                    result = command(["tesseract", "stdin", "stdout"], input=image.tobytes())
                    text = result.stdout.decode("utf-8", errors="replace").strip()
                    if text and text != previous:
                        out.write(f"[{index / fps:.2f}s]\n{text}\n\n")
                    previous = text
                index += 1
        if index == 0:
            raise RuntimeError("No video frames could be decoded for OCR.")
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transcript", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    report = extract(args.video, args.output, args.transcript, args.ocr)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
