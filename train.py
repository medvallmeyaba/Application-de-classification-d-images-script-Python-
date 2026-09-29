"""
Script d'entraînement — Transfer Learning avec MobileNetV2
------------------------------------------------------------------------------
Utilise ce script si tu veux entraîner un modèle sur TES PROPRES images.

Structure attendue du dataset :
    dataset/
    ├── train/
    │   ├── classe_1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   ├── classe_2/
    │   └── ...
    └── val/
        ├── classe_1/
        ├── classe_2/
        └── ...

Lancement :
    python train.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # cache les logs INFO/WARNING

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5        # entraînement de la tête seulement
EPOCHS_FINE_TUNE = 5   # fine-tuning des dernières couches
MODEL_OUTPUT_PATH = "model/mon_modele.h5"

# ----------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ----------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes détectées : {class_names}")

# Prétraitement + augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

train_ds = train_ds.map(lambda x, y: (preprocess_input(data_augmentation(x)), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ----------------------------------------------------------------------------
# CONSTRUCTION DU MODÈLE (base pré-entraînée + tête personnalisée)
# ----------------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # on gèle la base au départ

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------------------------------------------------------
# ÉTAPE 1 : entraînement de la tête seulement
# ----------------------------------------------------------------------------
print("\n=== Entraînement de la tête de classification ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# ----------------------------------------------------------------------------
# ÉTAPE 2 : fine-tuning (dégel des dernières couches de la base)
# ----------------------------------------------------------------------------
print("\n=== Fine-tuning ===")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # dégèle les 30 dernières couches
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR plus faible
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE_TUNE)

# ----------------------------------------------------------------------------
# SAUVEGARDE
# ----------------------------------------------------------------------------
import os
os.makedirs("model", exist_ok=True)
model.save(MODEL_OUTPUT_PATH)
print(f"\n✅ Modèle sauvegardé : {MODEL_OUTPUT_PATH}")
print(f"⚠️  Pense à mettre à jour CUSTOM_CLASS_NAMES dans app.py avec : {class_names}")
