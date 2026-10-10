#!/usr/bin/env python
"""PP-OCRv6_small HTTP API；POST /ocr 上传图片，返回 JSON。"""

import argparse
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from time import perf_counter

# 必须先导入此模块：它会处理其他虚拟环境注入的 CUDA 动态库路径。
from ocr_annotate import NUMERIC_TEXT, create_ocr

import cv2
import numpy as np
import paddle

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def extract_image(content_type: str, body: bytes) -> bytes:
    """支持原始图片请求体，或 multipart/form-data 的 image 字段。"""
    if content_type.split(";", 1)[0].strip().lower() != "multipart/form-data":
        if content_type.split(";", 1)[0].strip().lower() not in (
            "image/jpeg", "image/png", "image/webp", "application/octet-stream"
        ):
            raise ValueError("Content-Type 必须是图片，或 multipart/form-data")
        return body

    envelope = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode()
    message = BytesParser(policy=policy.default).parsebytes(envelope + body)
    if not message.is_multipart():
        raise ValueError("multipart/form-data 请求格式无效")
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") == "image":
            data = part.get_payload(decode=True)
            if data:
                return data
            break
    raise ValueError("缺少非空的 image 文件字段")


class OCRHandler(BaseHTTPRequestHandler):
    ocr = None  # 在服务启动时初始化一次；HTTPServer 按请求串行处理。

    def end_headers(self) -> None:
        # 同源访问不需要 CORS；保留这些响应头，便于局域网内调试网页。
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/ocr.html"):
            page = Path(__file__).with_name("ocr.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.path == "/health":
            self.send_json(200, {"status": "ok", "model": "PP-OCRv6_small"})
        else:
            self.send_json(404, {"error": "仅支持 GET /、GET /health 与 POST /ocr"})

    def do_POST(self) -> None:
        if self.path != "/ocr":
            self.send_json(404, {"error": "仅支持 POST /ocr"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self.send_json(413, {"error": "图片请求体为空或超过 25 MiB"})
                return
            data = extract_image(self.headers.get("Content-Type", ""), self.rfile.read(length))
            array = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码图片")
        except (ValueError, TypeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return

        try:
            start = perf_counter()
            predictions = list(self.ocr.predict(image))
            paddle.device.synchronize()
            predict_ms = round((perf_counter() - start) * 1000, 2)
            if not predictions:
                raise RuntimeError("OCR 未返回结果")
            result = predictions[0].json["res"]
            numeric = []
            other_boxes = []
            for text, score, polygon in zip(
                result["rec_texts"], result["rec_scores"], result["rec_polys"]
            ):
                points = [[int(x), int(y)] for x, y in polygon]
                if text and NUMERIC_TEXT.fullmatch(text.strip()):
                    numeric.append({"text": text, "score": float(score), "polygon": points})
                else:
                    other_boxes.append(points)
            self.send_json(200, {
                "model": "PP-OCRv6_small",
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "predict_ms": predict_ms,
                "results": numeric,
                "other_boxes": other_boxes,
            })
        except Exception:
            self.log_exception()
            self.send_json(500, {"error": "OCR 预测失败，详情见服务端日志"})

    def log_exception(self) -> None:
        import traceback
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机可访问")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须介于 1 和 65535 之间")
    OCRHandler.ocr = create_ocr()
    with HTTPServer((args.host, args.port), OCRHandler) as server:
        print(f"OCR API: http://{args.host}:{args.port}/ocr", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
