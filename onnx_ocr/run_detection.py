#!/usr/bin/env python3
"""Benchmark direct ORT DB detection, excluding image I/O and annotation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter
import cv2
import numpy as np
from pipeline import Detector, detector_input, db_boxes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "PP-OCRv6_medium_det"
DEFAULT_INPUT = ROOT / "images" / "image.png"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        parser.error("warmup must be nonnegative and runs positive")
    image = cv2.imread(str(args.input))
    if image is None:
        parser.error(f"cannot read {args.input}")
    detector = Detector(args.model_dir, args.device)
    for _ in range(args.warmup):
        detector.predict(image)
    total, network = [], []
    for _ in range(args.runs):
        start = perf_counter()
        tensor, shape = detector_input(image, detector.resize, detector.norm)
        raw_start = perf_counter()
        pred = detector.session.run(tensor)
        network.append((perf_counter() - raw_start) * 1000)
        boxes, scores = db_boxes(pred, shape, detector.post)
        total.append((perf_counter() - start) * 1000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"dt_polys": boxes, "dt_scores": scores}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for box in boxes:
        cv2.polylines(image, [np.asarray(box, np.int32)], True, (0, 0, 255), 2)
    if not cv2.imwrite(str(args.output_dir / "annotated.png"), image):
        raise RuntimeError("Failed writing annotation")
    summary = {"box_count": len(boxes), "network_input_shape": list(tensor.shape),
               "mean_total_ms": statistics.mean(total), "mean_session_run_ms": statistics.mean(network),
               "session_providers": detector.session.raw.get_providers(),
               "cuda_node_execution": "unverified: provider list is not per-node profiling evidence"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
