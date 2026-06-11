import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from tomato_detector import CLASS_NAMES_PATH, DATASET_TRAIN_DIR, IMAGE_SIZE, MODEL_PATH, load_class_names


LEARNING_DIR = Path(__file__).resolve().parent / "learning_samples"
ANNOTATION_DB_PATH = LEARNING_DIR / "annotations.jsonl"


class LearningManager:
    def __init__(self, reload_callback=None) -> None:
        self.reload_callback = reload_callback
        self._lock = threading.Lock()
        self.status = {
            "state": "idle",
            "message": "Waiting for learning samples.",
            "last_trained_at": None,
            "queued_samples": self.count_samples(),
            "annotation_records": self.count_annotation_records(),
        }

    def count_samples(self) -> int:
        if not LEARNING_DIR.exists():
            return 0
        return len([
            path
            for path in LEARNING_DIR.rglob("*")
            if path.is_file() and path.name != ANNOTATION_DB_PATH.name
        ])

    def count_annotation_records(self) -> int:
        if not ANNOTATION_DB_PATH.exists():
            return 0
        with ANNOTATION_DB_PATH.open("r", encoding="utf-8") as file:
            return sum(1 for line in file if line.strip())

    def labels(self) -> list[str]:
        labels = set(load_class_names())
        for root in (DATASET_TRAIN_DIR, LEARNING_DIR):
            if root.exists():
                labels.update(path.name for path in root.iterdir() if path.is_dir())
        return sorted(labels)

    def ensure_label(self, label: str) -> str:
        normalized = label.strip()
        if not normalized:
            raise ValueError("Label cannot be empty.")

        safe_label = "".join(ch for ch in normalized if ch.isalnum() or ch in (" ", "-", "_")).strip()
        if not safe_label:
            raise ValueError("Label must contain letters or numbers.")

        existing = next((item for item in self.labels() if item.lower() == safe_label.lower()), safe_label)
        labels = self.labels()
        if existing not in labels:
            labels.append(existing)

        (DATASET_TRAIN_DIR / existing).mkdir(parents=True, exist_ok=True)
        (LEARNING_DIR / existing).mkdir(parents=True, exist_ok=True)
        return existing

    def save_sample(self, source_path: Path, label: str) -> Path:
        normalized_label = self.ensure_label(label)

        destination_dir = LEARNING_DIR / normalized_label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{int(time.time())}_{source_path.name}"
        shutil.copy2(source_path, destination)
        self.status["queued_samples"] = self.count_samples()
        self.status["annotation_records"] = self.count_annotation_records()
        self.status["message"] = f"Saved learning sample as {normalized_label}."
        return destination

    def save_image(self, image, label: str, filename: str, metadata: dict | None = None) -> Path:
        normalized_label = self.ensure_label(label)
        destination_dir = LEARNING_DIR / normalized_label
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).stem or "annotation"
        destination = destination_dir / f"{int(time.time() * 1000)}_{safe_name}.jpg"
        image.convert("RGB").save(destination, format="JPEG", quality=94)
        if metadata is not None:
            self.record_annotation(
                {
                    **metadata,
                    "label": normalized_label,
                    "crop_path": str(destination.relative_to(LEARNING_DIR.parent)),
                }
            )
        self.status["queued_samples"] = self.count_samples()
        self.status["annotation_records"] = self.count_annotation_records()
        self.status["message"] = f"Saved annotation crop as {normalized_label}."
        return destination

    def record_annotation(self, metadata: dict) -> None:
        LEARNING_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        with ANNOTATION_DB_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def annotation_history(self, limit: int = 50) -> list[dict]:
        if not ANNOTATION_DB_PATH.exists():
            return []

        records: list[dict] = []
        with ANNOTATION_DB_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-limit:]

    def train_async(self) -> bool:
        if self._lock.locked():
            return False

        thread = threading.Thread(target=self._train, daemon=True)
        thread.start()
        return True

    def _train(self) -> None:
        with self._lock:
            sample_count = self.count_samples()
            if sample_count < 2:
                self.status.update(
                    {
                        "state": "idle",
                        "message": "Need at least 2 learning samples before retraining.",
                        "queued_samples": sample_count,
                        "annotation_records": self.count_annotation_records(),
                    }
                )
                return

            self.status.update(
                {
                    "state": "training",
                    "message": f"Retraining with {sample_count} new samples.",
                    "queued_samples": sample_count,
                    "annotation_records": self.count_annotation_records(),
                }
            )

            try:
                training_root = self._combined_training_root()
                class_names = self.labels()
                dataset = tf.keras.utils.image_dataset_from_directory(
                    training_root,
                    image_size=IMAGE_SIZE,
                    batch_size=8,
                    shuffle=True,
                    class_names=class_names,
                )
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

                model = self._load_or_build_model(len(class_names))
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
                model.fit(dataset, epochs=2, verbose=0)
                model.save(MODEL_PATH)
                CLASS_NAMES_PATH.parent.mkdir(exist_ok=True)
                with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
                    json.dump(class_names, file, indent=2)

                for class_dir in LEARNING_DIR.iterdir():
                    if not class_dir.is_dir():
                        continue
                    train_dir = DATASET_TRAIN_DIR / class_dir.name
                    train_dir.mkdir(parents=True, exist_ok=True)
                    for sample in class_dir.iterdir():
                        if sample.is_file():
                            shutil.copy2(sample, train_dir / sample.name)

                if self.reload_callback:
                    self.reload_callback()

                self.status.update(
                    {
                        "state": "idle",
                        "message": "Retraining complete. Model reloaded.",
                        "last_trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "queued_samples": self.count_samples(),
                        "annotation_records": self.count_annotation_records(),
                    }
                )
            except Exception as exc:
                self.status.update(
                    {
                        "state": "error",
                        "message": f"Retraining failed: {exc}",
                        "queued_samples": self.count_samples(),
                        "annotation_records": self.count_annotation_records(),
                    }
                )

    def _load_or_build_model(self, class_count: int):
        try:
            model = keras.models.load_model(MODEL_PATH)
            output_shape = model.output_shape
            if isinstance(output_shape, tuple) and output_shape[-1] == class_count:
                return model
        except Exception:
            pass

        return keras.Sequential(
            [
                keras.Input(shape=IMAGE_SIZE + (3,)),
                keras.layers.Rescaling(1.0 / 255),
                keras.layers.Conv2D(32, 3, activation="relu"),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(64, 3, activation="relu"),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(128, 3, activation="relu"),
                keras.layers.MaxPooling2D(),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.25),
                keras.layers.Dense(class_count, activation="softmax"),
            ]
        )

    def _combined_training_root(self) -> Path:
        combined_root = Path(__file__).resolve().parent / "training_cache"
        if combined_root.exists():
            shutil.rmtree(combined_root)
        combined_root.mkdir(parents=True, exist_ok=True)

        for source_root in (DATASET_TRAIN_DIR, LEARNING_DIR):
            if not source_root.exists():
                continue
            for class_dir in source_root.iterdir():
                if not class_dir.is_dir():
                    continue
                target_dir = combined_root / class_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)
                for image_path in class_dir.iterdir():
                    if image_path.is_file() and image_path.name != ".DS_Store":
                        target = target_dir / image_path.name
                        if target.exists():
                            target = target_dir / f"{source_root.name}_{image_path.name}"
                        shutil.copy2(image_path, target)

        return combined_root
