import json
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 35
EPOCHS_STAGE2 = 20
INIT_LR = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_LAYERS = 30

# Reproducibility
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ==============================
# DATASET LOADING
# ==============================
train_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_raw.class_names
NUM_CLASSES = len(class_names)
print(f"Found {NUM_CLASSES} classes")
print("Classes:", class_names)

# Save class labels for Flask API
labels_path = os.path.join(SAVE_DIR, "class_labels.txt")
with open(labels_path, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")
print("Saved class labels:", labels_path)


# ==============================
# CLASS IMBALANCE HANDLING
# ==============================
def compute_class_weight_map(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in dataset.unbatch():
        counts[int(labels.numpy())] += 1

    total = np.sum(counts)
    class_weights = {}
    for idx, count in enumerate(counts):
        if count > 0:
            class_weights[idx] = float(total / (num_classes * count))

    return counts, class_weights


class_counts, class_weight = compute_class_weight_map(train_raw, NUM_CLASSES)
print("\nClass distribution in training split:")
for idx, cls in enumerate(class_names):
    print(f"  {cls:20s} -> {class_counts[idx]:4d} | weight={class_weight[idx]:.3f}")

stats_path = os.path.join(SAVE_DIR, "training_class_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "seed": SEED,
            "class_names": class_names,
            "train_class_counts": class_counts.tolist(),
            "class_weight": {str(k): v for k, v in class_weight.items()},
        },
        f,
        indent=2,
    )
print("Saved class stats:", stats_path)


# ==============================
# DATA PIPELINE
# ==============================
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.15),
    ],
    name="data_augmentation",
)

AUTOTUNE = tf.data.AUTOTUNE


def prepare_train(images, labels):
    images = tf.cast(images, tf.float32)
    images = data_augmentation(images, training=True)
    images = preprocess_input(images)
    labels = tf.one_hot(labels, NUM_CLASSES)
    return images, labels


def prepare_val(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    labels = tf.one_hot(labels, NUM_CLASSES)
    return images, labels


train_ds = (
    train_raw.map(prepare_train, num_parallel_calls=AUTOTUNE)
    .shuffle(2000, seed=SEED)
    .prefetch(AUTOTUNE)
)

val_ds = val_raw.map(prepare_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# ==============================
# MODEL
# ==============================
def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )


def make_callbacks(stage_name):
    return [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-7),
        ModelCheckpoint(
            os.path.join(SAVE_DIR, f"best_model_{stage_name}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]


model, base_model = build_model()
compile_model(model, INIT_LR)

print("\n>>> STAGE 1: Training head")
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=make_callbacks("stage1"),
    class_weight=class_weight,
)

with open(os.path.join(SAVE_DIR, "history_stage1.pkl"), "wb") as f:
    pickle.dump(history_stage1.history, f)


# ==============================
# FINE TUNING
# ==============================
print("\n>>> STAGE 2: Fine-tuning")
base_model.trainable = True

for layer in base_model.layers[:-FINE_TUNE_UNFREEZE_LAYERS]:
    layer.trainable = False

# Keep BatchNorm frozen for stable fine-tuning
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

compile_model(model, FINE_TUNE_LR)

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2,
    initial_epoch=EPOCHS_STAGE1,
    callbacks=make_callbacks("stage2"),
    class_weight=class_weight,
)

with open(os.path.join(SAVE_DIR, "history_stage2.pkl"), "wb") as f:
    pickle.dump(history_stage2.history, f)


# ==============================
# SAVE + EVALUATE
# ==============================
final_model_path = os.path.join(SAVE_DIR, "my_model.keras")
model.save(final_model_path)
print("Saved final model:", final_model_path)

val_metrics = model.evaluate(val_ds, return_dict=True)
print("\nValidation metrics:")
for k, v in val_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

metrics_path = os.path.join(SAVE_DIR, "final_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(val_metrics, f, indent=2)
print("Saved metrics:", metrics_path)

# Quick summary for terminal
best_val_acc_stage1 = max(history_stage1.history.get("val_accuracy", [0]))
best_val_acc_stage2 = max(history_stage2.history.get("val_accuracy", [0]))
best_val_top3_stage2 = max(history_stage2.history.get("val_top3_acc", [0]))

print("\n=== Training Summary ===")
print(f"Best val accuracy (stage1): {best_val_acc_stage1:.2%}")
print(f"Best val accuracy (stage2): {best_val_acc_stage2:.2%}")
print(f"Best val top-3 acc (stage2): {best_val_top3_stage2:.2%}")
