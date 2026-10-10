#!/usr/bin/env python3
"""Create a separate FP16 ONNX copy of a PaddleOCR detection model.

The source directory is never modified.  Metadata such as inference.yml and
README.md is copied alongside the converted inference.onnx.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "models" / "PP-OCRv6_medium_det"
DEFAULT_DEST = HERE / "models" / "PP-OCRv6_medium_det_fp16"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    source = args.model_dir.resolve()
    destination = args.output_dir.resolve()
    source_model = source / "inference.onnx"
    destination_model = destination / "inference.onnx"
    if not source_model.is_file():
        parser.error(f"missing source model: {source_model}")
    if source == destination:
        parser.error("--output-dir must be different from --model-dir")
    if destination.exists():
        parser.error(f"output already exists; refusing to overwrite: {destination}")

    # Imports are intentionally delayed so --help works without ONNX installed.
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    destination.mkdir(parents=True)
    try:
        for item in source.iterdir():
            if item.name == "inference.onnx":
                continue
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        model = onnx.load(str(source_model))
        converted = convert_float_to_float16(model, keep_io_types=True)
        onnx.save(converted, str(destination_model))
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    print(f"wrote {destination_model}")
    print(f"copied metadata and auxiliary files from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
