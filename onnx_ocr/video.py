#!/usr/bin/env python3
"""Run complete PP-OCRv6 ONNX OCR on sampled video frames.

The detector and recognizer are separate official ONNX models. The default
configuration uses FP32 detection and recognition. Converted FP16 models are
optional experiments and must be selected explicitly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_DET = HERE / "models" / "PP-OCRv6_medium_det"
DEFAULT_REC = HERE / "models" / "PP-OCRv6_medium_rec"
DEFAULT_INPUT = ROOT / "images" / "VID20260929150008.mp4"
DEFAULT_OUTPUT = HERE / "output-video"

# Keep this runner independent of the Paddle static environment.  Only the
# annotation logic needed by the video script is reproduced here.
NUMERIC_TEXT = re.compile(r"[0-9０-９+＋\-－−./／:：%％#＃×*＝=()（）\[\]【】,，]+")
FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("no CJK font found")


def numeric_only(result: dict) -> dict:
    keep = [i for i, text in enumerate(result["rec_texts"])
            if text and NUMERIC_TEXT.fullmatch(text.strip())]
    filtered = dict(result)
    for key in ("rec_texts", "rec_scores", "rec_polys", "rec_boxes", "dt_polys"):
        if key in result and len(result[key]) == len(result["rec_texts"]):
            filtered[key] = [result[key][i] for i in keep]
    return filtered


def annotate(image: Image.Image, result: dict, font: ImageFont.FreeTypeFont) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    occupied: list[tuple[int, int, int, int]] = []
    for polygon, text, score in zip(result["rec_polys"], result["rec_texts"], result["rec_scores"]):
        points = [tuple(map(int, point)) for point in polygon]
        is_four_digits = re.fullmatch(r"[0-9]{4}", text) is not None
        is_digit_hyphen_digit = re.match(r"^[0-9]-[0-9]", text) is not None
        if is_digit_hyphen_digit:
            fill, line, width = (0, 180, 80, 75), (0, 180, 80, 220), 5
        elif is_four_digits:
            fill, line, width = (0, 110, 255, 75), (0, 110, 255, 220), 5
        else:
            fill, line, width = None, (255, 48, 48, 255), 5
        if fill:
            draw.polygon(points, fill=fill)
        draw.line(points + points[:1], fill=line, width=width)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        box_width, box_height = right - left + 14, bottom - top + 14
        x1, y1 = min(x for x, _ in points), min(y for _, y in points)
        x2, y2 = max(x for x, _ in points), max(y for _, y in points)
        candidates = [(x1, y1 - box_height - 6), (x1, y2 + 6), (x2 + 7, y1), (x1 - box_width - 7, y1)]
        candidates = [(max(0, min(image.width - box_width, x)), max(0, min(image.height - box_height, y))) for x, y in candidates]
        def overlaps(position):
            x, y = position
            return sum(x < a2 and x + box_width > a1 and y < b2 and y + box_height > b1 for a1, b1, a2, b2 in occupied)
        x, y = min(candidates, key=overlaps)
        occupied.append((x, y, x + box_width, y + box_height))
        background = ((0, 140, 65, 180) if is_digit_hyphen_digit else (0, 90, 220, 180) if is_four_digits else (0, 0, 0, 255))
        draw.rectangle((x, y, x + box_width, y + box_height), fill=background)
        draw.text((x + 7, y + 7 - top), text, font=font, fill="white" if score >= 0.5 else "#FFD060")
    return Image.alpha_composite(image, overlay).convert("RGB")


def make_ocr(det_dir: Path, rec_dir: Path, rec_batch_size: int, device="cuda"):
    from pipeline import OCR
    return OCR(det_dir, rec_dir, rec_batch_size, device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, default=DEFAULT_INPUT, nargs="?")
    parser.add_argument("--det-model-dir", type=Path, default=DEFAULT_DET)
    parser.add_argument("--rec-model-dir", type=Path, default=DEFAULT_REC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--font-size", type=int, default=30)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--rec-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--all-text", action="store_true", help="annotate all text, not only numeric text")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"video does not exist: {args.input}")
    if not (args.det_model_dir / "inference.onnx").is_file():
        parser.error(f"detector does not contain inference.onnx: {args.det_model_dir}")
    if not (args.rec_model_dir / "inference.onnx").is_file():
        parser.error(f"recognizer does not contain inference.onnx: {args.rec_model_dir}")
    for model_dir in (args.det_model_dir, args.rec_model_dir):
        if not (model_dir / "inference.yml").is_file():
            parser.error(f"model metadata inference.yml is missing: {model_dir}")
    if args.interval <= 0 or args.font_size <= 0 or args.rec_batch_size <= 0:
        parser.error("interval, font-size and rec-batch-size must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")
    if args.no_frames and args.video_output is None:
        parser.error("--no-frames requires --video-output")
    if args.video_output and args.video_output.resolve() == args.input.resolve():
        parser.error("video output cannot overwrite input")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.video_output:
        args.video_output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        parser.error(f"cannot open video: {args.input}")
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            parser.error("cannot get video FPS")
        step = max(1, round(fps * args.interval))
        ocr = make_ocr(args.det_model_dir, args.rec_model_dir, args.rec_batch_size, args.device)
        font = load_font(args.font_size)
        frame_index = 0
        processed = 0
        total_predict_time = 0.0
        while capture.grab():
            if frame_index % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"frame {frame_index} decode failed")
                start = perf_counter()
                data = ocr.predict(frame)
                elapsed = perf_counter() - start
                total_predict_time += elapsed
                numeric_data = data if args.all_text else numeric_only(data)
                import json
                (args.output_dir / f"frame_{frame_index:06d}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                labeled = annotate(image, numeric_data, font)
                if not args.no_frames:
                    labeled.save(args.output_dir / f"frame_{frame_index:06d}.png")
                if args.video_output:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(args.video_output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps / step,
                            labeled.size,
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot write: {args.video_output}")
                    writer.write(cv2.cvtColor(np.asarray(labeled), cv2.COLOR_RGB2BGR))
                print(
                    f"{frame_index / fps:.2f}s | {len(data.get('rec_texts', []))} OCR items | "
                    f"{len(numeric_data.get('rec_texts', []))} numeric | predict {elapsed:.3f}s",
                    flush=True,
                )
                processed += 1
                if args.max_frames is not None and processed >= args.max_frames:
                    break
            frame_index += 1
        if not processed:
            raise RuntimeError("video has no readable sampled frames")
        print(
            f"完成：{processed} 帧；完整 ONNX OCR 预测总耗时 {total_predict_time:.3f}s，"
            f"平均 {total_predict_time / processed:.3f}s/帧（不含解码、绘制和保存）"
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
