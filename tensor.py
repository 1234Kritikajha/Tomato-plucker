import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

TRAIN_DIR = Path("dataset/images/Train")
VAL_DIR = Path("dataset/images/val")
TEST_DIR = Path("dataset/images/Test")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tomato_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


def count_images(class_dir: Path) -> int:
    return len([path for path in class_dir.iterdir() if path.is_file() and path.name != ".DS_Store"])


for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if not directory.exists():
        raise FileNotFoundError(f"Dataset folder not found: {directory}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

CLASS_NAMES = train_ds.class_names
print("\nDetected Classes:")
print(CLASS_NAMES)

if set(CLASS_NAMES) != {"Ripe", "Unripe"}:
    raise ValueError(f"Expected dataset classes Ripe and Unripe, found: {CLASS_NAMES}")

MODEL_DIR.mkdir(exist_ok=True)
with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
    json.dump(CLASS_NAMES, file, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ]
)

try:
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    print("\nUsing MobileNetV2 with ImageNet weights.")
except Exception as exc:
    print(f"\nImageNet weights unavailable, training MobileNetV2 from scratch: {exc}")
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None,
    )
    base_model.trainable = True

model = keras.Sequential(
    [
        keras.Input(shape=IMG_SIZE + (3,)),
        data_augmentation,
        keras.applications.mobilenet_v2.preprocess_input,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
)

class_counts = {class_name: count_images(TRAIN_DIR / class_name) for class_name in CLASS_NAMES}
total_count = sum(class_counts.values())
class_weight = {
    index: total_count / (len(CLASS_NAMES) * class_counts[class_name])
    for index, class_name in enumerate(CLASS_NAMES)
}

print("\nClass counts:")
print(class_counts)
print("\nClass weights:")
print(class_weight)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight,
)

print("\nEvaluating on Test Dataset...\n")
loss, accuracy = model.evaluate(test_ds)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")
print("==============================")
print("\nModel Saved:")
print(MODEL_PATH)
print("\nClass Metadata Saved:")
print(CLASS_NAMES_PATH)
