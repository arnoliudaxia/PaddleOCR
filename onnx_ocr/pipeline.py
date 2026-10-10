# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2026 OCRPackages contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# Local adaptations of preprocessing, DB and CTC algorithm patterns.
"""Direct ORT OCR. Algorithms follow PaddleX Apache-2.0 processors (see README).

No guessed dictionary or model-specific defaults: valid inference.yml is required.
"""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
import yaml


def load_config(directory):
    path = Path(directory) / "inference.yml"
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read official metadata {path}: {exc}") from exc
    if not isinstance(config, dict) or not config.get("PreProcess") or not config.get("PostProcess"):
        raise ValueError(f"Missing/empty official PreProcess/PostProcess metadata: {path}; see README")
    return config


def operators(config):
    ops = config["PreProcess"].get("transform_ops", [])
    if not ops:
        raise ValueError("Metadata has no transform_ops")
    return {name: value or {} for op in ops for name, value in op.items()}


def validate_image(image):
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("Expected nonempty HWC BGR image with three channels")


def scale_value(value):
    if isinstance(value, str):
        parts = value.split("/")
        if len(parts) == 2:
            return float(parts[0]) / float(parts[1])
    return float(value)


def detector_input(image, resize, norm):
    validate_image(image)
    h, w = image.shape[:2]
    if h + w < 64:
        padded = np.zeros((max(32, h), max(32, w), 3), np.uint8)
        padded[:h, :w] = image
        image = padded
    ih, iw = image.shape[:2]
    if "image_shape" in resize:
        rh, rw = resize["image_shape"]
        if resize.get("keep_ratio", False):
            rw = math.ceil(iw * rh / ih / 32) * 32
    elif "resize_long" in resize:
        ratio = resize["resize_long"] / max(ih, iw)
        rh, rw = [math.ceil(int(v * ratio) / 128) * 128 for v in (ih, iw)]
    else:
        limit = resize.get("limit_side_len", 736)
        kind = resize.get("limit_type", "min")
        if kind == "max":
            ratio = min(1., limit / max(ih, iw))
        elif kind == "min":
            ratio = max(1., limit / min(ih, iw))
        elif kind == "resize_long":
            ratio = limit / max(ih, iw)
        else:
            raise ValueError(f"Unsupported detector limit_type: {kind}")
        rh, rw = int(ih * ratio), int(iw * ratio)
        cap = resize.get("max_side_limit", 4000)
        if max(rh, rw) > cap:
            ratio = cap / max(rh, rw)
            rh, rw = int(rh * ratio), int(rw * ratio)
        rh, rw = max(32, round(rh / 32) * 32), max(32, round(rw / 32) * 32)
    image = cv2.resize(image, (int(rw), int(rh))).astype(np.float32)
    mean = np.asarray(norm.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(norm.get("std", [0.229, 0.224, 0.225]), np.float32)
    image = (image * scale_value(norm.get("scale", 1 / 255)) - mean) / std
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None]), (h, w)


def mini_box(contour):
    rect = cv2.minAreaRect(np.asarray(contour, np.float32))
    p = sorted(cv2.boxPoints(rect), key=lambda x: x[0])
    left = sorted(p[:2], key=lambda x: x[1])
    right = sorted(p[2:], key=lambda x: x[1])
    return np.asarray([left[0], right[0], right[1], left[1]], np.float32), min(rect[1])


def db_boxes(probability, original_shape, config):
    import pyclipper
    pred = np.asarray(probability)
    if pred.ndim != 4 or pred.shape[:2] != (1, 1):
        raise ValueError(f"Expected DB probability map [1,1,H,W], got {pred.shape}")
    pred = pred[0, 0]
    bitmap = (pred > config.get("thresh", 0.3)).astype(np.uint8)
    if config.get("use_dilation", False):
        bitmap = cv2.dilate(bitmap, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    h, w = original_shape
    for contour in contours[:config.get("max_candidates", 1000)]:
        box, side = mini_box(contour)
        if side < 3:
            continue
        mask = np.zeros(pred.shape, np.uint8)
        score_points = contour if config.get("score_mode", "fast") == "slow" else box
        cv2.fillPoly(mask, [np.asarray(score_points, np.int32).reshape(-1, 2)], 1)
        score = cv2.mean(pred, mask)[0]
        if score < config.get("box_thresh", 0.7):
            continue
        perimeter = cv2.arcLength(box, True)
        if perimeter <= 0:
            continue
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(cv2.contourArea(box) * config.get("unclip_ratio", 2.) / perimeter)
        if len(expanded) != 1 or len(expanded[0]) < 3:
            continue
        box, side = mini_box(expanded[0])
        if side < 5:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] * w / pred.shape[1]), 0, w - 1)
        box[:, 1] = np.clip(np.round(box[:, 1] * h / pred.shape[0]), 0, h - 1)
        if np.linalg.norm(box[0] - box[1]) <= 3 or np.linalg.norm(box[0] - box[3]) <= 3:
            continue
        boxes.append(box.tolist())
        scores.append(float(score))
    ordered = sorted(zip(boxes, scores), key=lambda item: (item[0][0][1], item[0][0][0]))
    # Official OCR reading-order correction for boxes on the same line.
    for i in range(1, len(ordered)):
        j = i
        while j > 0 and abs(ordered[j][0][0][1] - ordered[j-1][0][0][1]) < 10 and ordered[j][0][0][0] < ordered[j-1][0][0][0]:
            ordered[j], ordered[j-1] = ordered[j-1], ordered[j]
            j -= 1
    return [x[0] for x in ordered], [x[1] for x in ordered]


def perspective_crop(image, box):
    validate_image(image)
    points = np.asarray(box, np.float32)
    if points.shape != (4, 2):
        raise ValueError("Perspective crop requires ordered TL,TR,BR,BL quadrilateral")
    width = int(max(np.linalg.norm(points[0]-points[1]), np.linalg.norm(points[2]-points[3])))
    height = int(max(np.linalg.norm(points[0]-points[3]), np.linalg.norm(points[1]-points[2])))
    if min(width, height) <= 0:
        raise ValueError("Degenerate crop")
    target = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(np.rot90(crop) if height / width >= 1.5 else crop)


def recognition_input(images, shape):
    c, h, base_w = shape
    if c != 3 or min(h, base_w) <= 0 or not images:
        raise ValueError("Invalid recognition shape or empty batch")
    normalized = []
    for image in images:
        validate_image(image)
        ih, iw = image.shape[:2]
        width = min(3200, int(h * max(base_w / h, iw / ih)))
        resized_w = min(width, math.ceil(h * iw / ih))
        array = cv2.resize(image, (resized_w, h)).astype(np.float32).transpose(2, 0, 1) / 127.5 - 1
        padded = np.zeros((3, h, width), np.float32)
        padded[:, :, :resized_w] = array
        normalized.append(padded)
    max_width = max(x.shape[2] for x in normalized)
    return np.stack([np.pad(x, ((0, 0), (0, 0), (0, max_width-x.shape[2]))) for x in normalized])


def character_table(config, directory):
    post = config["PostProcess"]
    chars = post.get("character_dict", post.get("character_list"))
    if chars is None and post.get("character_dict_path"):
        path = Path(directory) / post["character_dict_path"]
        chars = path.read_text(encoding="utf-8").splitlines()
    if not isinstance(chars, list) or not chars or not all(isinstance(x, str) and x for x in chars):
        raise ValueError("Official ordered character_dict/list (or dictionary file) missing; cannot decode safely")
    return [""] + chars + ([" "] if post.get("use_space_char", True) else [])


def ctc_decode(predictions, characters):
    pred = np.asarray(predictions)
    if pred.ndim != 3 or pred.shape[-1] != len(characters):
        raise ValueError(f"CTC class count mismatch: output {pred.shape}, dictionary {len(characters)}")
    indices, probabilities = pred.argmax(-1), pred.max(-1)
    result = []
    for ids, probs in zip(indices, probabilities):
        selection = np.ones(len(ids), bool)
        selection[1:] = ids[1:] != ids[:-1]
        selection &= ids != 0
        result.append(("".join(characters[i] for i in ids[selection]), float(probs[selection].mean()) if selection.any() else 0.))
    return result


class Session:
    def __init__(self, directory, device):
        import onnxruntime as ort
        path = Path(directory) / "inference.onnx"
        if not path.is_file():
            raise ValueError(f"Missing ONNX model: {path}")
        if device == "cuda":
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA provider unavailable; use --device cpu explicitly")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.raw = ort.InferenceSession(str(path), providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.raw.get_providers():
            raise RuntimeError("ORT fell back to CPU during CUDA session creation")
        inputs = self.raw.get_inputs()
        if len(inputs) != 1 or inputs[0].type not in ("tensor(float)", "tensor(float16)"):
            raise ValueError("Expected one floating-point ONNX image input")
        self.input = inputs[0]

    def run(self, tensor):
        dtype = np.float16 if self.input.type == "tensor(float16)" else np.float32
        return self.raw.run(None, {self.input.name: tensor.astype(dtype)})[0]


class Detector:
    def __init__(self, directory, device="cuda"):
        config = load_config(directory)
        ops = operators(config)
        if config["PostProcess"].get("name") != "DBPostProcess" or config["PostProcess"].get("box_type", "quad") != "quad":
            raise ValueError("Only DBPostProcess quad metadata is supported")
        allowed = {"DecodeImage", "DetLabelEncode", "DetResizeForTest", "NormalizeImage", "ToCHWImage", "KeepKeys"}
        if set(ops) - allowed or not {"DetResizeForTest", "NormalizeImage"} <= set(ops):
            raise ValueError(f"Unsupported detector transform_ops: {list(ops)}")
        if ops.get("DecodeImage", {}).get("img_mode", "BGR") != "BGR":
            raise ValueError("Only official BGR input metadata supported")
        self.resize, self.norm, self.post = ops["DetResizeForTest"], ops["NormalizeImage"], config["PostProcess"]
        self.session = Session(directory, device)

    def predict(self, image):
        tensor, shape = detector_input(image, self.resize, self.norm)
        return db_boxes(self.session.run(tensor), shape, self.post)


class OCR:
    def __init__(self, det_dir, rec_dir, batch_size=64, device="cuda"):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # Validate both metadata files before allocating GPU sessions.
        config = load_config(rec_dir)
        ops = operators(config)
        if config["PostProcess"].get("name") != "CTCLabelDecode":
            raise ValueError("Only CTCLabelDecode supported")
        allowed = {"DecodeImage", "MultiLabelEncode", "RecResizeImg", "KeepKeys"}
        if set(ops) - allowed or "RecResizeImg" not in ops:
            raise ValueError(f"Unsupported recognizer transform_ops: {list(ops)}")
        if ops.get("DecodeImage", {}).get("img_mode", "BGR") != "BGR":
            raise ValueError("Only BGR recognition metadata supported")
        self.shape = ops["RecResizeImg"].get("image_shape")
        if not self.shape or len(self.shape) != 3:
            raise ValueError("RecResizeImg.image_shape missing")
        self.characters = character_table(config, rec_dir)
        self.detector = Detector(det_dir, device)
        self.recognizer = Session(rec_dir, device)
        shape = self.recognizer.input.shape
        if len(shape) != 4 or isinstance(shape[3], int):
            raise ValueError("This runner requires dynamic-width NCHW recognition export")
        output_shape = self.recognizer.raw.get_outputs()[0].shape
        if isinstance(output_shape[-1], int) and output_shape[-1] != len(self.characters):
            raise ValueError("Official dictionary class count does not match ONNX output")
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError("Fixed recognition batches larger than one are unsupported")
        self.batch_size = 1 if isinstance(shape[0], int) else batch_size

    def predict(self, image):
        boxes, scores = self.detector.predict(image)
        crops = [perspective_crop(image, box) for box in boxes]
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
        decoded = [None] * len(crops)
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start+self.batch_size]
            tensor = recognition_input([crops[i] for i in ids], self.shape)
            results = ctc_decode(self.recognizer.run(tensor), self.characters)
            for i, result in zip(ids, results):
                decoded[i] = result
        return {"dt_polys": boxes, "dt_scores": scores, "rec_polys": boxes,
                "rec_texts": [x[0] for x in decoded], "rec_scores": [x[1] for x in decoded]}
