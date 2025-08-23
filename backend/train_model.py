'''import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"  
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
EPOCHS = 50

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
print(f" Found {NUM_CLASSES} classes: {class_names}")

# Save class names for Flask app
with open("class_labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ==============================
# OPTIMIZE PERFORMANCE
# ==============================
AUTOTUNE = tf.data.AUTOTUNE
# MobileNetV2 expects -1 to 1 range
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
# BASE MODEL
# ==============================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze backbone first

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
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

# ==============================
# TRAIN
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================
# SAVE MODELS & HISTORY
# ==============================
model.save("model.keras")   
print("Final model saved as model.keras")
print("Best model saved as best_model.keras")

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("Training history saved as history.pkl")

# ==============================
# OPTIONAL: FINE-TUNE
# ==============================
print("\n>>> Starting optional fine-tuning ...")
base_model.trainable = True
for layer in base_model.layers[:-30]:  # keep most frozen, unfreeze last 30 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

model.save("fine_tuned_model.keras")
print("Fine-tuned model saved as fine_tuned_model.keras")'''


import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"  
BATCH_SIZE = 32 # Increased for more stable training
IMG_SIZE = (224, 224)
EPOCHS = 50
FINE_TUNE_EPOCHS = 10

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
# DATA AUGMENTATION & NORMALIZATION
# ==============================
data_augmentation = tf.keras.Sequential([
    # Input images are in [0, 255] range. MobileNetV2 expects [-1, 1].
    tf.keras.layers.Rescaling(1./127.5, offset=-1), 
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
base_model.trainable = False  # freeze backbone first

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
    tf.keras.callbacks.ModelCheckpoint("best_model_initial.keras", save_best_only=True)
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

print("\nFinal model saved as model.keras")
model.save("model.keras") 
print("Best model from initial training saved as best_model_initial.keras")

with open("history_initial.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("Initial training history saved as history_initial.pkl")

# ==============================
# OPTIONAL: FINE-TUNE
# ==============================
print("\n>>> Starting optional fine-tuning ...")
base_model.trainable = True
# Unfreeze the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model_fine_tuned.keras", save_best_only=True)
]

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_tune_callbacks
)

model.save("fine_tuned_model.keras")
print("Final fine-tuned model saved as fine_tuned_model.keras")

with open("history_fine_tuned.pkl", "wb") as f:
    pickle.dump(fine_tune_history.history, f)
print("Fine-tuning history saved as history_fine_tuned.pkl")
