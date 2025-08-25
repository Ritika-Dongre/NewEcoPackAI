import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle
import os

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50
FINE_TUNE_EPOCHS = 10
MODEL_DIR = "saved_model/my_model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# LOAD DATASETS
# ==============================
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Found {NUM_CLASSES} classes: {class_names}")

# Save class labels for Flask
with open(os.path.join(MODEL_DIR, "class_labels.txt"), "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ==============================
# DATA AUGMENTATION & NORMALIZATION
# ==============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./127.5, offset=-1),  # scale [-1,1]
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# ==============================
# OPTIMIZE PERFORMANCE
# ==============================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================
# BASE MODEL
# ==============================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze backbone

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs, outputs)

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# CALLBACKS
# ==============================
initial_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model_initial.keras"), save_best_only=True)
]

# ==============================
# TRAIN
# ==============================
print("\n>>> Starting initial training (feature extraction) ...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=initial_callbacks
)

# Save final model for Flask
model.save(MODEL_DIR)
print(f"Final model saved in {MODEL_DIR}")

# Save training history
with open(os.path.join(MODEL_DIR, "history_initial.pkl"), "wb") as f:
    pickle.dump(history.history, f)
print("Initial training history saved.")

# ==============================
# OPTIONAL FINE-TUNE
# ==============================
print("\n>>> Starting optional fine-tuning ...")
base_model.trainable = True
for layer in base_model.layers[:-30]:  # unfreeze last 30 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model_fine_tuned.keras"), save_best_only=True)
]

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_tune_callbacks
)

model.save(os.path.join(MODEL_DIR, "fine_tuned_model.keras"))
print("Fine-tuned model saved.")

with open(os.path.join(MODEL_DIR, "history_fine_tuned.pkl"), "wb") as f:
    pickle.dump(fine_tune_history.history, f)
print("Fine-tuning history saved.")
