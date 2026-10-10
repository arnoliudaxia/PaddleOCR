#!/usr/bin/env python
"""用 PP-OCRv6 识别图片，并在原图坐标上标注识别文字。"""

import argparse
import os
import re
import sys
from pathlib import Path
from time import perf_counter

# 避免其他 Python 虚拟环境的 CUDA/cuDNN 库抢先被 Paddle 加载。
lib_path = os.environ.get("LD_LIBRARY_PATH", "")
parts = lib_path.split(":") if lib_path else []
clean_parts = [p for p in parts if not (
    "/site-packages/nvidia/" in p and not p.startswith(str(Path(sys.prefix).resolve()) + "/")
)]
if len(clean_parts) != len(parts):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(clean_parts)
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

from PIL import Image, ImageDraw, ImageFont
import paddle
from paddleocr import PaddleOCR


# 只保留完全由数字及常见数字标注符号组成的识别结果；含中英文字母的整行丢弃。
NUMERIC_TEXT = re.compile(r"[0-9０-９+＋\-－−./／:：%％#＃×*＝=()（）\[\]【】,，]+")

FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("未找到可用字体，请安装 Noto CJK 或 Droid Sans Fallback")


def numeric_only(result: dict) -> dict:
    keep = [i for i, text in enumerate(result["rec_texts"])
            if text and NUMERIC_TEXT.fullmatch(text.strip())]
    filtered = dict(result)
    # 这些字段逐项对应 rec_texts，必须一起过滤才能维持坐标和置信度对齐。
    for key in ("rec_texts", "rec_scores", "rec_polys", "rec_boxes",
                "dt_polys", "textline_orientation_angles"):
        if key in result and len(result[key]) == len(result["rec_texts"]):
            filtered[key] = [result[key][i] for i in keep]
    return filtered


def annotate(image: Image.Image, result: dict, font: ImageFont.FreeTypeFont) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    occupied: list[tuple[int, int, int, int]] = []
    for polygon, text, score in zip(
        result["rec_polys"], result["rec_texts"], result["rec_scores"]
    ):
        points = [tuple(map(int, point)) for point in polygon]
        if not text or not NUMERIC_TEXT.fullmatch(text.strip()):
            # 非数字/符号区域保留检测框，但不显示识别文字。
            draw.line(points + points[:1], fill=(180, 180, 180, 170), width=3)
            continue
        is_four_digits = re.fullmatch(r"[0-9]{4}", text) is not None
        is_digit_hyphen_digit = re.match(r"^[0-9]-[0-9]", text) is not None
        if is_digit_hyphen_digit:
            draw.polygon(points, fill=(0, 180, 80, 75))
            draw.line(points + points[:1], fill=(0, 180, 80, 220), width=5)
        elif is_four_digits:
            draw.polygon(points, fill=(0, 110, 255, 75))
            draw.line(points + points[:1], fill=(0, 110, 255, 220), width=5)
        else:
            draw.line(points + points[:1], fill=(255, 48, 48, 255), width=5)
        label = text
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width, height = right - left + 14, bottom - top + 14
        x1, y1 = min(x for x, _ in points), min(y for _, y in points)
        x2, y2 = max(x for x, _ in points), max(y for _, y in points)
        candidates = [(x1, y1 - height - 6), (x1, y2 + 6),
                      (x2 + 7, y1), (x1 - width - 7, y1)]
        candidates = [
            (max(0, min(image.width - width, x)),
             max(0, min(image.height - height, y)))
            for x, y in candidates
        ]

        def overlaps(position: tuple[int, int]) -> int:
            x, y = position
            return sum(x < a2 and x + width > a1 and y < b2 and y + height > b1
                       for a1, b1, a2, b2 in occupied)

        x, y = min(candidates, key=overlaps)
        occupied.append((x, y, x + width, y + height))
        background = ((0, 140, 65, 180) if is_digit_hyphen_digit
                      else (0, 90, 220, 180) if is_four_digits else (0, 0, 0, 255))
        draw.rectangle((x, y, x + width, y + height), fill=background)
        color = "white" if score >= 0.5 else "#FFD060"
        draw.text((x + 7, y + 7 - top), label, font=font, fill=color)
    return Image.alpha_composite(image, overlay).convert("RGB")


def create_ocr() -> PaddleOCR:
    """创建固定使用 PP-OCRv6_small 的 GPU 静态图 FP32 OCR 产线。"""
    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0",
        engine="paddle_static",
        engine_config={"run_mode": "paddle"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="输入图片路径")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output-annotated")
    parser.add_argument("--font-size", type=int, default=30)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"输入图片不存在：{args.input}")
    if args.font_size <= 0:
        parser.error("--font-size 必须为正整数")
    font = load_font(args.font_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ocr = create_ocr()
    image = Image.open(args.input)
    input_path = str(args.input)
    # 首轮触发 GPU 内核/缓存初始化，不计入耗时；仅统计第二轮 predict。
    list(ocr.predict(input_path))
    paddle.device.synchronize()
    start = perf_counter()
    results = list(ocr.predict(input_path))
    paddle.device.synchronize()
    print(f"预测耗时（warm-up 后，仅第二轮 predict）：{perf_counter() - start:.3f} 秒")
    if not results:
        raise RuntimeError(f"OCR 未返回预测结果：{args.input}")
    data = results[0].json["res"]
    image_path = args.output_dir / f"{args.input.stem}_labeled.png"
    annotate(image, data, font).save(image_path)
    print(f"检测 {len(data['rec_texts'])} 个区域，其中数字/符号 {len(numeric_only(data)['rec_texts'])} 个：{image_path}")


if __name__ == "__main__":
    main()
