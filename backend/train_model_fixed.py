# train_model_efficientnet_mixup.py
import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (300, 300)   # EfficientNetB3 recommended
EPOCHS = 40
FINE_TUNE_EPOCHS = 20

MIXUP_ALPHA = 0.2       # MixUp strength
WEIGHT_DECAY = 1e-5

# ==============================
# UTIL: MixUp
# ==============================
def sample_beta_distribution(batch_size, alpha):
    """Sample lambda from Beta(alpha, alpha) for MixUp"""
    gamma1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lam = gamma1 / (gamma1 + gamma2)
    return lam

def mixup_batch(images, labels, alpha=MIXUP_ALPHA):
    """Apply MixUp to a batch of images and one-hot labels.
       images: [B,H,W,C], labels: [B,NUM_CLASSES]
    """
    if alpha <= 0:
        return images, labels

    batch_size = tf.shape(images)[0]
    lam = sample_beta_distribution(batch_size, alpha)  # shape [B]
    lam_x = tf.reshape(lam, (batch_size, 1, 1, 1))
    lam_y = tf.reshape(lam, (batch_size, 1))

    # shuffle the batch
    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)

    mixed_images = images * lam_x + images_shuffled * (1.0 - lam_x)
    mixed_labels = labels * lam_y + labels_shuffled * (1.0 - lam_y)
    return mixed_images, mixed_labels

# ==============================
# LOAD DATASETS (Keras utility)
# ==============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
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

# Save class labels for your Flask app
with open(os.path.join(SAVE_DIR, "class_labels.txt"), "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

# ==============================
# AUGMENTATION & PREPROCESSING
# ==============================
data_augmentation = tf.keras.Sequential([
    # Use EfficientNet's preprocessing instead of manual rescaling
    tf.keras.layers.Lambda(preprocess_input),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomTranslation(0.15, 0.15),
])

AUTOTUNE = tf.data.AUTOTUNE

# Convert labels to one-hot and apply augmentation
def prepare_train(image_batch, label_batch):
    image_batch = tf.cast(image_batch, tf.float32)
    image_batch = data_augmentation(image_batch)
    label_batch = tf.one_hot(label_batch, NUM_CLASSES)
    return image_batch, label_batch

def prepare_val(image_batch, label_batch):
    image_batch = tf.cast(image_batch, tf.float32)
    # Use EfficientNet's preprocessing for validation too
    image_batch = preprocess_input(image_batch)
    label_batch = tf.one_hot(label_batch, NUM_CLASSES)
    return image_batch, label_batch

train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(2000)

# Apply MixUp at dataset level (map per-batch)
def mixup_map(images, labels):
    return mixup_batch(images, labels, alpha=MIXUP_ALPHA)

train_ds = train_ds.map(lambda x, y: mixup_map(x, y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.map(prepare_val, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==============================
# MODEL (EfficientNetB3)
# ==============================
# First create base model without preprocessing (we handle it ourselves)
base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False

# Build the complete model
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

# Cosine LR schedule with restarts
initial_lr = 1e-3
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=initial_lr,
    first_decay_steps=2000,
    t_mul=2.0,
    m_mul=0.9
)

optimizer = Adam(learning_rate=lr_schedule)

# Use categorical crossentropy since labels are one-hot (MixUp)
loss_fn = CategoricalCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")]
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ModelCheckpoint(os.path.join(SAVE_DIR, "best_efficientnet_mixup.keras"), 
                   save_best_only=True, 
                   monitor="val_accuracy")
]

# ==============================
# STAGE 1: Train head
# ==============================
print("\n>>> STAGE 1: Training head ...")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

with open(os.path.join(SAVE_DIR, "history_stage1.pkl"), "wb") as f:
    pickle.dump(history1.history, f)

# ==============================
# STAGE 2: Fine-tune deeper layers
# ==============================
print("\n>>> STAGE 2: Fine tuning ...")
base_model.trainable = True

# Freeze most layers, unfreeze last blocks
for layer in base_model.layers[:-80]:
    layer.trainable = False

# Recompile with lower LR
optimizer_finetune = Adam(learning_rate=1e-5)
model.compile(
    optimizer=optimizer_finetune,
    loss=loss_fn,
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")]
)

callbacks_ft = [
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    ModelCheckpoint(os.path.join(SAVE_DIR, "best_efficientnet_mixup_finetuned.keras"),
                   save_best_only=True,
                   monitor="val_accuracy")
]

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks_ft
)

# Save final model
model.save(os.path.join(SAVE_DIR, "my_model.keras"))
print("Saved final model to:", os.path.join(SAVE_DIR, "my_model.keras"))

with open(os.path.join(SAVE_DIR, "history_finetune.pkl"), "wb") as f:
    pickle.dump(fine_history.history, f)