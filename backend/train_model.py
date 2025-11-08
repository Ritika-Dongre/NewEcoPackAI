import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"
BATCH_SIZE = 32              # increased batch size
IMG_SIZE = (224, 224)        # MobileNetV2 recommended size
EPOCHS = 50                  # more training
FINE_TUNE_EPOCHS = 30
INIT_LR = 0.001             # initial learning rate

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# LOAD DATASETS
# ==============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
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

# Save class labels for Flask API
with open(os.path.join(SAVE_DIR, "class_labels.txt"), "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

# ==============================
# DATA AUGMENTATION (stronger)
# ==============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Lambda(preprocess_input),  # Model-specific preprocessing
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.2, 0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.GaussianNoise(0.1),
    tf.keras.layers.RandomCrop(IMG_SIZE[0], IMG_SIZE[1])
])

AUTOTUNE = tf.data.AUTOTUNE

def prepare_data(image, label):
    """Prepare data with one-hot labels"""
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# Apply preprocessing and augmentation
train_ds = train_ds.map(prepare_data, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(2000)
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# Only preprocess validation data
val_ds = val_ds.map(prepare_data, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ==============================
# MODEL
# ==============================
def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    
    return Model(inputs, outputs)

model = build_model()

# Compile with categorical crossentropy (one-hot labels)
model.compile(
    optimizer=Adam(learning_rate=INIT_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

# ==============================
# CALLBACKS
# ==============================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        os.path.join(SAVE_DIR, "best_model_initial.keras"),
        save_best_only=True,
        monitor='val_accuracy'
    )
]

# ==============================
# INITIAL TRAINING
# ==============================
print("\n>>> TRAINING MODEL ...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

with open(os.path.join(SAVE_DIR, "history_initial.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# ==============================
# FINE TUNING
# ==============================
print("\n>>> FINE TUNING ...")
base_model = model.layers[1]  # Get the base model
base_model.trainable = True

# Freeze earlier layers, unfreeze later layers
for layer in base_model.layers[:-20]:  # Keep more layers frozen
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

callbacks_ft = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        os.path.join(SAVE_DIR, "best_model_fine_tuned.keras"),
        save_best_only=True,
        monitor='val_accuracy'
    )
]

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks_ft
)

# Save final model
model.save(os.path.join(SAVE_DIR, "my_model.keras"))
print("Fine-tuned model saved!")

with open(os.path.join(SAVE_DIR, "history_fine_tuned.pkl"), "wb") as f:
    pickle.dump(fine_history.history, f)

# Print final metrics
final_val_accuracy = max(fine_history.history['val_accuracy'])
final_val_top3 = max(fine_history.history['val_top3_acc'])
print(f"\nBest validation accuracy: {final_val_accuracy:.2%}")
print(f"Best validation top-3 accuracy: {final_val_top3:.2%}")
