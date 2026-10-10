"""Synthetic CPU tests: no ONNX weights, ORT sessions or GPU required."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from pipeline import (detector_input, recognition_input, ctc_decode,
                      perspective_crop, db_boxes, load_config, character_table)


class PipelineTests(unittest.TestCase):
    def test_detector_resize_normalization(self):
        image = np.full((100, 200, 3), 255, np.uint8)
        tensor, shape = detector_input(image, {"limit_side_len": 96, "limit_type": "max"},
                                       {"scale": "1./255.", "mean": [0.5]*3, "std": [0.5]*3})
        self.assertEqual(tensor.shape, (1, 3, 64, 96))
        self.assertEqual(shape, (100, 200))
        np.testing.assert_allclose(tensor, 1)
        self.assertEqual(tensor.dtype, np.float32)

    def test_recognition_padding_and_bgr(self):
        image = np.zeros((24, 48, 3), np.uint8)
        image[:, :, 0] = 255
        tensor = recognition_input([image], [3, 48, 320])
        self.assertEqual(tensor.shape, (1, 3, 48, 320))
        np.testing.assert_allclose(tensor[0, 0, :, :96], 1)
        np.testing.assert_allclose(tensor[0, 1, :, :96], -1)
        np.testing.assert_allclose(tensor[:, :, :, 96:], 0)

    def test_ctc_duplicate_blank_confidence(self):
        pred = np.zeros((2, 6, 3), np.float32)
        for t, idx in enumerate([1, 1, 0, 1, 2, 2]):
            pred[0, t, idx] = 0.8
        pred[1, :, 0] = 1
        result = ctc_decode(pred, ["", "a", "b"])
        self.assertEqual(result[0][0], "aab")
        self.assertAlmostEqual(result[0][1], .8)
        self.assertEqual(result[1], ("", 0.))
        with self.assertRaises(ValueError):
            ctc_decode(pred, ["", "a"])

    def test_crop_and_vertical_rotation(self):
        image = np.full((100, 120, 3), 42, np.uint8)
        crop = perspective_crop(image, [[10, 10], [90, 10], [90, 30], [10, 30]])
        self.assertEqual(crop.shape, (20, 80, 3))
        np.testing.assert_array_equal(crop, 42)
        crop = perspective_crop(image, [[10, 10], [30, 10], [30, 90], [10, 90]])
        self.assertEqual(crop.shape, (20, 80, 3))
        with self.assertRaises(ValueError):
            perspective_crop(image, [[0, 0]] * 4)

    def test_db_map_coordinates(self):
        pred = np.zeros((1, 1, 64, 64), np.float32)
        pred[0, 0, 20:35, 10:40] = .95
        boxes, scores = db_boxes(pred, (128, 256), {"box_thresh": .5, "unclip_ratio": 1.4})
        self.assertEqual(len(boxes), 1)
        self.assertGreater(scores[0], .9)
        self.assertGreater(max(p[0] for p in boxes[0]), 150)
        empty, _ = db_boxes(np.zeros_like(pred), (128, 256), {})
        self.assertEqual(empty, [])

    def test_empty_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "inference.yml").write_text("")
            with self.assertRaisesRegex(ValueError, "Missing/empty"):
                load_config(tmp)
        with self.assertRaises(ValueError):
            character_table({"PostProcess": {}}, ".")
        self.assertEqual(character_table({"PostProcess": {"character_dict": ["a", "b"], "use_space_char": True}}, "."), ["", "a", "b", " "])


if __name__ == "__main__":
    unittest.main()
