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
BATCH_SIZE = 16        # smaller batch size
IMG_SIZE = (224, 224)
EPOCHS = 30
FINE_TUNE_EPOCHS = 10
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# LOAD DATASETS (train/validation split)
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

# Save class names for Flask app
with open("class_labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ==============================
# PREPROCESSING (normalize first)
# ==============================
preprocessing = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./127.5, offset=-1)  # normalize [-1,1]
])

# ==============================
# DATA AUGMENTATION
# ==============================
data_augmentation = tf.keras.Sequential([
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
base_model.trainable = False  # freeze backbone initially

inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocessing(inputs)
x = data_augmentation(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

# ==============================
# COMPILE (initial training)
# ==============================
model.compile(
    optimizer=Adam(learning_rate=3e-4),  # lower LR
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# CALLBACKS
# ==============================
initial_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "best_model_initial.keras"), save_best_only=True)
]

# ==============================
# INITIAL TRAINING (feature extraction)
# ==============================
print("\n>>> Starting initial training (feature extraction) ...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=initial_callbacks
)

# Save model for Flask app
model.save(os.path.join(SAVE_DIR, "my_model.keras"))
print(f"\nModel saved at {os.path.join(SAVE_DIR, 'my_model.keras')}")

with open("history_initial.pkl", "wb") as f:
    pickle.dump(history.history, f)

# ==============================
# OPTIONAL: FINE-TUNING
# ==============================
print("\n>>> Starting optional fine-tuning ...")
base_model.trainable = True
for layer in base_model.layers[:-30]:  # freeze early layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # smaller LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "best_model_fine_tuned.keras"), save_best_only=True)
]

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_tune_callbacks
)

# Save fine-tuned model in same folder
model.save(os.path.join(SAVE_DIR, "my_model.keras"))
print(f"\nFine-tuned model saved at {os.path.join(SAVE_DIR, 'my_model.keras')}")
with open("history_fine_tuned.pkl", "wb") as f:
    pickle.dump(fine_tune_history.history, f)
