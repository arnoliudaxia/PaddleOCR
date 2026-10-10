#!/usr/bin/env python
"""抽帧读取视频，复用 ocr_annotate 的 OCR 产线生成标注帧或视频。"""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import paddle
from paddleocr import PaddleOCR

from ocr_annotate import annotate, create_ocr, load_font, numeric_only


def create_video_ocr(model: str) -> PaddleOCR:
    if model == "small":
        return create_ocr()
    return PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{model}_det",
        text_recognition_model_name=f"PP-OCRv6_{model}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0",
        engine="paddle_static",
        engine_config={"run_mode": "paddle"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="输入视频")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output-video")
    parser.add_argument("--video-output", type=Path, help="输出标注视频 MP4（帧率为抽样帧率）")
    parser.add_argument("--no-frames", action="store_true", help="只生成视频，不另存每帧 PNG")
    parser.add_argument("--model", choices=("medium", "small", "tiny"), default="small")
    parser.add_argument("--interval", type=float, default=1.0, help="抽帧时间间隔（秒，默认 1）")
    parser.add_argument("--font-size", type=int, default=30)
    parser.add_argument("--max-frames", type=int, help="最多处理多少个抽样帧，便于快速测试")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"视频不存在：{args.input}")
    if args.interval <= 0 or args.font_size <= 0:
        parser.error("--interval 和 --font-size 必须为正数")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames 必须为正整数")
    if args.no_frames and args.video_output is None:
        parser.error("--no-frames 需要配合 --video-output")
    if args.video_output and args.video_output.resolve() == args.input.resolve():
        parser.error("不能覆盖输入视频")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        parser.error(f"视频无法打开：{args.input}")
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            parser.error("无法获取视频帧率")
        step = max(1, round(fps * args.interval))
        if not args.no_frames:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.video_output:
            args.video_output.parent.mkdir(parents=True, exist_ok=True)
        font = load_font(args.font_size)
        ocr = create_video_ocr(args.model)  # 只初始化一次，所有帧共用同一 GPU 模型。
        index = 0
        processed = 0
        total_predict_time = 0.0
        while capture.grab():
            if index % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"第 {index} 帧解码失败")
                # OpenCV 解码为 BGR，PaddleX 接受 BGR ndarray；显示前转换为 RGB。
                start = perf_counter()
                results = list(ocr.predict(frame))
                paddle.device.synchronize()
                elapsed = perf_counter() - start
                total_predict_time += elapsed
                if not results:
                    raise RuntimeError(f"第 {index} 帧 OCR 未返回结果")
                data = numeric_only(results[0].json["res"])
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = index / fps
                labeled = annotate(image, data, font)
                if not args.no_frames:
                    path = args.output_dir / f"frame_{index:06d}.png"
                    labeled.save(path)
                if args.video_output:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.video_output), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps / step, labeled.size)
                        if not writer.isOpened():
                            raise RuntimeError(f"无法写入 MP4：{args.video_output}")
                    writer.write(cv2.cvtColor(np.asarray(labeled), cv2.COLOR_RGB2BGR))
                print(f"{timestamp:.2f}s | {len(data['rec_texts'])} 项 | predict {elapsed:.3f}s", flush=True)
                processed += 1
                if args.max_frames is not None and processed >= args.max_frames:
                    break
            index += 1
        if not processed:
            raise RuntimeError("视频没有可读取的帧")
        print(f"完成：{processed} 帧；预测总耗时 {total_predict_time:.3f}s，"
              f"平均 {total_predict_time / processed:.3f}s/帧（不含解码、绘制和保存）")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
