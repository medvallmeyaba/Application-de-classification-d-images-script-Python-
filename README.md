# Classification d'images — Script Python (TensorFlow + Tkinter)

## 🚀 Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Lancer l'application (modèle pré-entraîné ImageNet)

```bash
python app.py
```

Une fenêtre de dialogue s'ouvre : choisis une image sur ton ordinateur. Le modèle
(MobileNetV2, TensorFlow) l'analyse, puis une fenêtre s'affiche avec l'image et
la prédiction (top 3 classes + score de confiance) écrite dessus. Le résultat est
aussi affiché dans le terminal.

## 🎓 Entraîner ton propre modèle

1. Organise ton dataset :
   ```
   dataset/
   ├── train/
   │   ├── chat/
   │   ├── chien/
   │   └── ...
   └── val/
       ├── chat/
       ├── chien/
       └── ...
   ```
2. Lance l'entraînement :
   ```bash
   python train.py
   ```
3. Dans `app.py`, mets :
   ```python
   USE_CUSTOM_MODEL = True
   CUSTOM_CLASS_NAMES = ["chat", "chien", ...]  # ordre donné par train.py
   ```
4. Relance `python app.py`.

## 📁 Structure du projet

```
.
├── app.py              # Script principal : sélection + classification + affichage
├── train.py             # Entraînement (transfer learning)
├── requirements.txt
└── model/
    └── mon_modele.h5    # généré après entraînement
```

## ℹ️ Remarque

Ce script utilise **Tkinter** pour la fenêtre de dialogue et l'affichage — il est donc
inclus nativement avec Python et ne nécessite pas d'installation supplémentaire
(sauf sur certaines distributions Linux : `sudo apt install python3-tk`).
