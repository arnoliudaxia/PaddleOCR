# Independent PP-OCRv6 ONNX Runtime OCR

Direct **ONNX Runtime + NumPy/OpenCV** detection and recognition. No imports or
runtime dependencies on Paddle, PaddleOCR or PaddleX. Original `../ocr_video.py`
and `../ocr_annotate.py` are untouched. Default models are FP32.

## Install

```bash
uv sync --project PaddleOCR/onnx_ocr --python 3.11 --extra cuda
uv run --project PaddleOCR/onnx_ocr python PaddleOCR/onnx_ocr/download_models.py
```

The project depends on `onnxruntime-gpu` (also provides the CPU execution
provider). Do not install CPU `onnxruntime` alongside it. The optional `cuda`
extra provides CUDA 12/cuDNN 9 shared libraries; an NVIDIA driver is still
required. `ort.preload_dlls(directory="")` loads pip CUDA libraries before
session creation. If host libraries conflict, try `env -u LD_LIBRARY_PATH`.
`--device cpu` explicitly uses CPU. CUDA initialization fallback is rejected;
a CUDA provider list alone does **not** prove every node runs on the GPU.

`requirements.lock.txt` now contains only direct pins, **not a resolved
transitive lockfile**. The previous PaddleOCR environment snapshot was removed.

## Required official metadata (important)

Each model directory must contain nonempty `inference.onnx` and `inference.yml`:

```text
models/PP-OCRv6_medium_det/{inference.onnx,inference.yml}
models/PP-OCRv6_medium_rec/{inference.onnx,inference.yml}
```

Downloader repositories:
- `PaddlePaddle/PP-OCRv6_medium_det_onnx`
- `PaddlePaddle/PP-OCRv6_medium_rec_onnx`

The checked-in runner does not ship weights or dictionaries. HF downloads in
this workspace had **zero-byte YAML/JSON/README files** despite valid weights.
The runner rejects missing/empty metadata, and never fabricates a character
mapping. Recognition YAML must provide `PostProcess.character_dict` (ordered
list), `character_list`, or `character_dict_path` relative to the model directory.
A CTC blank is prepended, and space appended according to `use_space_char`
(default true, matching the official decoder). Output class count must match.
For the official medium recognition metadata here, the resulting table has
**18710 entries**. Do not substitute an unrelated Chinese/English dictionary.

If HF metadata is empty, obtain `inference.yml` from the **same official model's
export** (official Paddle model archive/config), not from arbitrary model names.
An explicit local metadata fallback is available, with no Paddle imports:

```bash
uv run --project PaddleOCR/onnx_ocr python PaddleOCR/onnx_ocr/download_models.py \
  --metadata-root /home/arno/.paddlex/official_models
```

It copies `<metadata-root>/<model-name>/inference.yml` only when the downloaded
YAML is empty/missing and validates model name/config structure. Optional
`--revision` pins the HF revision for both repositories. Auxiliary dictionary
files referenced by YAML must be supplied manually. Source path is recorded in
`metadata-source.txt`. Keep model revisions and metadata provenance together.

For this working tree, the empty YAMLs were repaired using existing official
cached `PP-OCRv6_medium_det/rec` exports under `~/.paddlex/official_models`.
Source paths and SHA256 hashes were saved in ignored model directories. This
cache is an optional **data source**, not a Python dependency.

## Pipeline and supported scope

`pipeline.py` implements:
- BGR detector resize to multiples of 32, mean/std normalization and NCHW;
- ORT DB probability map, threshold/dilation, fast/slow box scoring,
  pyclipper unclip, quadrilateral boxes mapped to original resolution;
- reading-order sorting, perspective crops (vertical crops rotated);
- aspect-ratio-sorted batched recognition, per-crop normalization to [-1,1],
  zero padding, width cap 3200, correct greedy CTC duplicate/blank removal;
- result assembly with boxes, detection scores, text and recognition confidence.

Official YAML thresholds are honored. Null `DetResizeForTest` uses the inspected
processor's defaults (short-side minimum 736, max side 4000), **not** the former
PaddleOCR high-level pipeline's possible 960/max overrides. Thus box counts and
cost can differ from old runs. Recognition normalization remains BGR, not RGB.

Practical scope: DB **quad** outputs and CTC recognition, one floating image
input per model, dynamic recognition width. Unknown transforms, non-CTC decoders,
polygon output mode and fixed-width recognition exports are rejected. Label
encoding transforms in inference metadata are training-only and ignored.
No document unwarping or orientation classifier, beam search or word boxes.
Returned confidence assumes the official ONNX output is a probability tensor,
not arbitrary logits. FP16 models are optional experiments, not an accuracy or
speed guarantee. Verify prediction quality on your own frames.

## Video and detection CLI

```bash
uv run --project PaddleOCR/onnx_ocr python PaddleOCR/onnx_ocr/video.py \
  images/VID20260929150008.mp4 --max-frames 2 --rec-batch-size 16 \
  --output-dir PaddleOCR/onnx_ocr/output-video \
  --video-output PaddleOCR/onnx_ocr/output-video/result.mp4

uv run --project PaddleOCR/onnx_ocr python PaddleOCR/onnx_ocr/run_detection.py \
  --warmup 3 --runs 10
```

Video output preserves original resolution. `--interval 1` samples about one
frame per second. MP4 contains sampled annotated frames, at source FPS divided
by sample step, **without audio**. `--no-frames` suppresses PNGs, requires MP4;
JSON results are still saved for each sampled frame. Annotation defaults to
numeric-like text; `--all-text` shows all OCR text. Install `fonts-noto-cjk` for
CJK annotation. `--device cpu` is supported by both CLIs.

Prediction timing includes CPU preprocessing/crop/postprocess and both networks,
not video decode/drawing/writing. Detector benchmark reports separate ORT session
run and full detection means. It does not claim per-node CUDA verification.

Optional FP16 conversion:
```bash
uv sync --project PaddleOCR/onnx_ocr --extra cuda --extra conversion
uv run --project PaddleOCR/onnx_ocr python PaddleOCR/onnx_ocr/convert_fp16.py \
  --model-dir PaddleOCR/onnx_ocr/models/PP-OCRv6_medium_rec \
  --output-dir PaddleOCR/onnx_ocr/models/PP-OCRv6_medium_rec_fp16
```
Select via `--rec-model-dir`/`--det-model-dir`. Conversion keeps FP32 I/O and
copies metadata, refuses existing destination, never modifies source weights.

## Validation

```bash
python -m unittest discover -s PaddleOCR/onnx_ocr -p 'test_*.py' -v
python -m py_compile PaddleOCR/onnx_ocr/*.py
```

Nine synthetic tests cover resize/normalization, BGR channels, padding, CTC
blank/repeat/confidence/class mismatch, perspective crop/rotation, DB boxes,
empty metadata, numeric filtering and annotation resolution. No GPU/models
required and ORT is imported only during session construction.

Local validation with the existing Python 3.11 ORT environment:
- all nine tests and compilation passed;
- CUDA image smoke (`images/image.png`, 2160x3840): 51 boxes/text entries;
  complete detector+recognizer ran without loading paddle/paddleocr/paddlex;
- one video frame: 34 OCR items / 16 numeric; PNG/JSON/MP4 written to
  `/tmp/direct-ort-smoke`. This is a functional smoke, not a warmed speed benchmark.

TODO: reproducible fresh-environment lock, externally pinned metadata provenance,
reference accuracy comparison and warmed end-to-end FP32/FP16 video benchmarks.
The smoke validates execution, **not dictionary semantic correctness against
reference labels or full per-node GPU execution**.

## Algorithm provenance

Preprocessing, DB and CTC patterns inspected from installed PaddleX 3.7.2:
`paddlex/inference/models/text_detection/processors.py` and
`paddlex/inference/models/text_recognition/processors.py` (PaddlePaddle Authors,
Apache-2.0). Implementation is local and does not import those modules.
See `NOTICE` and `LICENSE` for attribution/license of derived algorithm code.
