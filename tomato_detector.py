import json
import threading
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "tomato_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
DATASET_TRAIN_DIR = BASE_DIR / "dataset" / "images" / "Train"
IMAGE_SIZE = (224, 224)
DEFAULT_CLASS_NAMES = ["Ripe", "Unripe"]


class TomatoDetectorError(Exception):
    """Raised when prediction cannot be completed safely."""


def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
            names = json.load(file)
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return names
        raise TomatoDetectorError(f"Invalid class metadata in {CLASS_NAMES_PATH}")

    if DATASET_TRAIN_DIR.exists():
        names = sorted(path.name for path in DATASET_TRAIN_DIR.iterdir() if path.is_dir())
        if names:
            return names

    return DEFAULT_CLASS_NAMES


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def open_image(source: str | Path | BinaryIO) -> Image.Image:
    try:
        image = Image.open(source)
        image.load()
        return image
    except FileNotFoundError as exc:
        raise TomatoDetectorError(f"Image file not found: {source}") from exc
    except (OSError, UnidentifiedImageError) as exc:
        raise TomatoDetectorError("The uploaded file is not a valid image.") from exc


class TomatoDetector:
    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self._lock = threading.RLock()
        self.model_path = Path(model_path)
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise TomatoDetectorError(f"Model file not found: {self.model_path}")

        self.model = load_model(self.model_path)
        class_names = load_class_names()

        output_shape = self.model.output_shape
        output_count = output_shape[-1] if isinstance(output_shape, tuple) else None
        if output_count and output_count != len(class_names):
            if len(class_names) > output_count:
                class_names = class_names[:output_count]
            else:
                class_names = [
                    *class_names,
                    *[f"Class {index + 1}" for index in range(len(class_names), output_count)],
                ]

        self.class_names = class_names

    def reload(self) -> None:
        with self._lock:
            self._load()

    def classify_image(self, image: Image.Image) -> dict:
        with self._lock:
            processed_image = preprocess_image(image)
            raw_prediction = np.asarray(self.model.predict(processed_image, verbose=0))[0]

        return self._format_prediction(raw_prediction)

    def predict(self, source: str | Path | BinaryIO) -> dict:
        image = open_image(source)
        result = self.classify_image(image)
        result["detections"] = [
            {
                "id": 1,
                "bbox": {
                    "x": 0,
                    "y": 0,
                    "width": image.width,
                    "height": image.height,
                },
                **result,
            }
        ]
        result["summary"] = self._summarize_detections(result["detections"])
        return result

    def _format_prediction(self, raw_prediction: np.ndarray) -> dict:
        if raw_prediction.size == 1:
            positive_confidence = float(raw_prediction[0])
            predicted_index = 1 if positive_confidence >= 0.5 else 0
            confidence = positive_confidence if predicted_index == 1 else 1 - positive_confidence
            scores = {
                self.class_names[0]: round((1 - positive_confidence) * 100, 2),
                self.class_names[1]: round(positive_confidence * 100, 2),
            }
        else:
            predicted_index = int(np.argmax(raw_prediction))
            confidence = float(raw_prediction[predicted_index])
            scores = {
                class_name: round(float(score) * 100, 2)
                for class_name, score in zip(self.class_names, raw_prediction)
            }

        label = self.class_names[predicted_index]
        return {
            "prediction": label,
            "ripeness": "ripe" if label.lower() == "ripe" else "unripe",
            "confidence": round(confidence * 100, 2),
            "scores": scores,
        }

    def detect_frame(self, source: str | Path | BinaryIO) -> dict:
        image = open_image(source).convert("RGB")
        candidates = self._find_tomato_candidates(image)

        detections = []
        for index, bbox in enumerate(candidates, start=1):
            x, y, width, height = bbox
            crop = image.crop((x, y, x + width, y + height))
            prediction = self.classify_image(crop)
            detections.append(
                {
                    "id": index,
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                    },
                    **prediction,
                }
            )

        if not detections:
            whole_image = self.classify_image(image)
            detections.append(
                {
                    "id": 1,
                    "bbox": {
                        "x": 0,
                        "y": 0,
                        "width": image.width,
                        "height": image.height,
                    },
                    **whole_image,
                }
            )

        primary = max(detections, key=lambda item: item["confidence"])
        return {
            "prediction": primary["prediction"],
            "ripeness": primary["ripeness"],
            "confidence": primary["confidence"],
            "scores": primary["scores"],
            "image": {"width": image.width, "height": image.height},
            "detections": detections,
            "summary": self._summarize_detections(detections),
        }

    def _find_tomato_candidates(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        masks = [
            cv2.inRange(hsv, (0, 45, 45), (12, 255, 255)),
            cv2.inRange(hsv, (168, 45, 45), (180, 255, 255)),
            cv2.inRange(hsv, (15, 40, 45), (38, 255, 255)),
            cv2.inRange(hsv, (38, 35, 35), (88, 255, 255)),
        ]
        mask = masks[0]
        for item in masks[1:]:
            mask = cv2.bitwise_or(mask, item)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = image.width * image.height
        min_area = max(450, image_area * 0.002)
        max_area = image_area * 0.75

        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            aspect = width / max(height, 1)
            extent = area / max(width * height, 1)
            if not 0.45 <= aspect <= 2.1:
                continue
            if extent < 0.24:
                continue

            pad = int(max(width, height) * 0.18)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(image.width, x + width + pad)
            y1 = min(image.height, y + height + pad)
            boxes.append((x0, y0, x1 - x0, y1 - y0))

        return self._merge_boxes(boxes)

    def _merge_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        merged: list[tuple[int, int, int, int]] = []
        for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
            if all(self._iou(box, existing) < 0.35 for existing in merged):
                merged.append(box)
        return merged[:20]

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_w = max(0, min(ax2, bx2) - max(ax, bx))
        inter_h = max(0, min(ay2, by2) - max(ay, by))
        intersection = inter_w * inter_h
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0

    def _summarize_detections(self, detections: list[dict]) -> dict:
        ripe = sum(1 for item in detections if item["prediction"].lower() == "ripe")
        unripe = sum(1 for item in detections if item["prediction"].lower() == "unripe")
        average_confidence = (
            round(sum(item["confidence"] for item in detections) / len(detections), 2)
            if detections
            else 0
        )
        return {
            "total": len(detections),
            "ripe": ripe,
            "unripe": unripe,
            "average_confidence": average_confidence,
        }
