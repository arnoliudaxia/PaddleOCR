#!/usr/bin/env python3
"""Download official FP32 ONNX models and their PaddleOCR metadata."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = HERE / "models"
MODEL_REPOS = {
    "PP-OCRv6_medium_det": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
    "PP-OCRv6_medium_rec": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
}
ALLOW_PATTERNS = ["inference.onnx", "inference.yml", "inference.json", "README.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--revision", help="pin Hugging Face revision (applies to both repositories)")
    parser.add_argument("--metadata-root", type=Path, help="explicit official exported model root; copy model-name/inference.yml when HF metadata is empty")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    for name, repo in MODEL_REPOS.items():
        destination = args.models_dir / name
        snapshot_download(repo_id=repo, repo_type="model", local_dir=str(destination),
                          allow_patterns=ALLOW_PATTERNS, revision=args.revision)
        metadata = destination / "inference.yml"
        if (not metadata.is_file() or not metadata.stat().st_size) and args.metadata_root:
            source = args.metadata_root / name / "inference.yml"
            if source.is_file() and source.stat().st_size:
                shutil.copy2(source, metadata)
                (destination / "metadata-source.txt").write_text(str(source.resolve()) + "\n", encoding="utf-8")
        for required in ("inference.onnx", "inference.yml"):
            required_path = destination / required
            if not required_path.is_file() or required_path.stat().st_size == 0:
                raise RuntimeError(
                    f"missing or empty required model metadata: {required_path}; "
                    "the repository revision may not expose LFS metadata")
        from pipeline import load_config, character_table
        config = load_config(destination)
        if config.get("Global", {}).get("model_name") != name:
            raise RuntimeError(f"metadata model name mismatch: {destination}")
        if name.endswith("_rec"):
            character_table(config, destination)
        print(f"downloaded weights and validated metadata structure: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
