"""
Classification d'images — Script autonome (Deep Learning / TensorFlow)
------------------------------------------------------------------------------
Lancement :
    python app.py

Fonctionnement :
    1. Une fenêtre de dialogue s'ouvre pour choisir une image sur ton PC.
    2. Le modèle TensorFlow (MobileNetV2, pré-entraîné sur ImageNet) analyse
       l'image et prédit sa classe.
    3. Une fenêtre s'ouvre affichant l'image avec la prédiction écrite dessus.

Pour utiliser TON PROPRE modèle (entraîné avec train.py) :
    - Passe USE_CUSTOM_MODEL à True
    - Renseigne CUSTOM_CLASS_NAMES avec tes classes
"""

import tkinter as tk
from tkinter import filedialog
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # cache les logs INFO/WARNING


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
DISPLAY_SIZE = (500, 500)          # taille d'affichage de l'image résultat
USE_CUSTOM_MODEL = False
CUSTOM_MODEL_PATH = "model/mon_modele.h5"
CUSTOM_CLASS_NAMES = ["classe_0", "classe_1", "classe_2"]  # à adapter


# ----------------------------------------------------------------------------
# CHARGEMENT DU MODÈLE
# ----------------------------------------------------------------------------
def load_model():
    print("Chargement du modèle...")
    if USE_CUSTOM_MODEL:
        model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    else:
        model = MobileNetV2(weights="imagenet")
    print("Modèle chargé.")
    return model


# ----------------------------------------------------------------------------
# PRÉTRAITEMENT + PRÉDICTION
# ----------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(IMG_SIZE)
    array = tf.keras.preprocessing.image.img_to_array(resized)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def predict(model, image: Image.Image, top_k: int = 3):
    array = preprocess_image(image)
    preds = model.predict(array, verbose=0)

    if USE_CUSTOM_MODEL:
        probs = preds[0]
        top_indices = probs.argsort()[-top_k:][::-1]
        results = [(CUSTOM_CLASS_NAMES[i], float(probs[i])) for i in top_indices]
    else:
        decoded = decode_predictions(preds, top=top_k)[0]
        results = [(label, float(prob)) for (_, label, prob) in decoded]

    return results


# ----------------------------------------------------------------------------
# AFFICHAGE : image + prédiction écrite dessus
# ----------------------------------------------------------------------------
def draw_prediction_on_image(image: Image.Image, results: list) -> Image.Image:
    image = image.convert("RGB").resize(DISPLAY_SIZE)

    try:
        font_big = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Bandeau semi-transparent en bas pour la lisibilité du texte
    band_height = 30 + 22 * len(results)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(0, image.height - band_height), (image.width, image.height)],
        fill=(0, 0, 0, 160),
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    top_label, top_prob = results[0]
    y = image.height - band_height + 8
    draw.text((10, y), f"{top_label} ({top_prob * 100:.1f}%)", fill="white", font=font_big)

    for label, prob in results[1:]:
        y += 24
        draw.text((10, y), f"{label} — {prob * 100:.1f}%", fill="white", font=font_small)

    return image


# ----------------------------------------------------------------------------
# INTERFACE TKINTER
# ----------------------------------------------------------------------------
def show_result_window(image_with_text: Image.Image):
    window = tk.Toplevel()
    window.title("Résultat de la classification")

    tk_image = ImageTk.PhotoImage(image_with_text)
    label = tk.Label(window, image=tk_image)
    label.image = tk_image  # évite le garbage collection
    label.pack()

    close_btn = tk.Button(window, text="Fermer", command=window.destroy)
    close_btn.pack(pady=8)


def main():
    while(True):
        root = tk.Tk()
        root.withdraw()  # cache la fenêtre principale vide

        file_path = filedialog.askopenfilename(
            title="Choisis une image à classifier",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )

        if not file_path:
            print("Aucune image sélectionnée. Fin du programme.")
            return

        model = load_model()

        original_image = Image.open(file_path)
        results = predict(model, original_image, top_k=3)

        print("\n=== Résultats ===")
        for label, prob in results:
            print(f"{label} : {prob * 100:.2f}%")

        image_with_text = draw_prediction_on_image(original_image, results)
        show_result_window(image_with_text)


if __name__ == "__main__":
    main()
