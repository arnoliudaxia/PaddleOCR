"""Lightweight tests: no models or GPU required."""
import unittest
from pathlib import Path

from PIL import Image, ImageFont
import video
import run_detection
import download_models


class VideoTests(unittest.TestCase):
    def test_numeric_filter_preserves_alignment_and_input(self):
        result = {
            "rec_texts": ["1234", "abc", "１－２", "", " 12.5% "],
            "rec_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
            "rec_polys": list(range(5)), "rec_boxes": list(range(5)),
            "dt_polys": list(range(6)),
        }
        filtered = video.numeric_only(result)
        self.assertEqual(filtered["rec_texts"], ["1234", "１－２", " 12.5% "])
        self.assertEqual(filtered["rec_polys"], [0, 2, 4])
        self.assertEqual(filtered["rec_scores"], [0.9, 0.7, 0.5])
        self.assertEqual(filtered["dt_polys"], list(range(6)))
        self.assertEqual(len(result["rec_texts"]), 5)

    def test_annotation_preserves_resolution(self):
        original = Image.new("RGB", (200, 100), "white")
        result = {"rec_polys": [[[20, 30], [80, 30], [80, 60], [20, 60]]],
                  "rec_texts": ["1234"], "rec_scores": [0.9]}
        rendered = video.annotate(original, result, ImageFont.load_default())
        self.assertEqual(rendered.size, original.size)
        self.assertEqual(rendered.mode, "RGB")
        self.assertNotEqual(rendered.tobytes(), original.tobytes())
        self.assertEqual(original.getpixel((20, 30)), (255, 255, 255))

    def test_default_paths_and_fp32(self):
        here = Path(__file__).resolve().parent
        self.assertEqual(video.DEFAULT_DET, here / "models" / "PP-OCRv6_medium_det")
        self.assertEqual(video.DEFAULT_REC, here / "models" / "PP-OCRv6_medium_rec")
        self.assertEqual(run_detection.DEFAULT_MODEL_DIR, video.DEFAULT_DET)
        self.assertEqual(download_models.DEFAULT_MODELS, here / "models")
        self.assertEqual(video.DEFAULT_INPUT.parent, here.parents[1] / "images")


if __name__ == "__main__":
    unittest.main()
