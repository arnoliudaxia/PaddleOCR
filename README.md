# OCR：两套独立方案

本目录保留两种实现，运行环境和推理链相互独立。

| 方案 | 入口 | 推理与前后处理 |
|---|---|---|
| Paddle Static | `ocr_video.py`、`ocr_annotate.py`、`ocr_api.py` | PaddleOCR/PaddleX + Paddle Static |
| ONNX Runtime | `onnx_ocr/video.py`、`onnx_ocr/run_detection.py` | 直接 ORT + OpenCV/NumPy，本地 DB/CTC 后处理 |

## Paddle 方案

保留现有 Paddle 环境和 API/Web 文件。视频默认 small；可选择 medium。
当前静态图配置为 FP32，关闭文本行方向分类。

```bash
cd PaddleOCR
uv run ocr_video.py ../images/VID20260929150008.mp4 --model medium
```

此命令使用既有 Paddle 环境；不要在其中混装 ONNX 方案依赖。

## 独立 ONNX 方案

依赖、下载模型、命令、限制及测试见 [`onnx_ocr/README.md`](onnx_ocr/README.md)。
ONNX 方案不导入 `paddle`、`paddleocr` 或 `paddlex`，默认检测和识别均为 FP32。

```bash
uv sync --project PaddleOCR/onnx_ocr --python 3.11 --extra cuda
uv run --project PaddleOCR/onnx_ocr python PaddleOCR/onnx_ocr/download_models.py
uv run --project PaddleOCR/onnx_ocr python PaddleOCR/onnx_ocr/video.py \
  images/VID20260929150008.mp4 --max-frames 2
```

上面的命令从 Git 仓库根目录运行。模型权重、运行输出、凭据与虚拟环境不提交。
此目录属于外层 `OCRPackages` Git 仓库，不另建嵌套仓库。
